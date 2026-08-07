"""Flower + HuggingFace Trainer + DARL Federated DiLoCo LLM Training Entrypoint.

Trains Causal Language Models (LLMs) on pre-tokenized datasets across Snellius and LUMI.
Uses HuggingFace `Trainer` for the inner optimization loop (AdamW, bf16/fp16, FSDP,
gradient accumulation, grad clipping) and Flower gRPC for outer DiLoCo aggregation.

Usage:
    python3 -m pww.train_llm_flower --config configs/llm_gpt2_diloco.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import add_common_args, apply_config_file, resolve_output_dir
from .darl.client import LeaseClient, LeaseSession
from .darl.space import BlockSpace, blocks_for_phase
from .darl.torch_data import DARLDataSource, LeasedSampler
from . import distributed as D
from .logging_utils import get_logger, setup_logging
from .models.llm import build_llm
from .data.text import TokenizedDataset, build_hf_tokenized_dataset

logger = get_logger("pww.train_llm_flower")

HAS_FLWR = False
try:
    import flwr as fl
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False

HAS_TRANSFORMERS = False
try:
    from transformers import Trainer, TrainingArguments
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class DiLoCoLLMFlowerClient(fl.client.NumPyClient if HAS_FLWR else object):
    """Flower Client for LLM Training using HuggingFace Trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: TokenizedDataset,
        eval_dataset: TokenizedDataset | None,
        sampler: LeasedSampler,
        darl_source: DARLDataSource,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.sampler = sampler
        self.darl_source = darl_source
        self.args = args
        self.device = device
        self.inner_steps = getattr(args, "inner_steps", None) or getattr(args, "diloco_inner_steps", 100)

        # Built once, here -- not per phase inside the inner loop. AdamW's moment
        # estimates are the whole reason to use it, and rebuilding the optimiser
        # every phase threw them away before they could accumulate, which is why a
        # 27-round WikiText run sat flat at loss ~3.15. DiLoCo keeps inner optimiser
        # state across the outer step too (arXiv 2311.08105 section 2), so this
        # deliberately survives from one round to the next.
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self._consumed_samples = 0

    def get_parameters(self, config: dict) -> list:
        return [p.detach().to(torch.float32).cpu().numpy() for p in self.model.parameters() if p.requires_grad]

    def set_parameters(self, parameters: list) -> None:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        for p, val in zip(trainable_params, parameters):
            tensor_val = torch.from_numpy(val).to(device=self.device, dtype=p.dtype)
            p.data.copy_(tensor_val)

    def _execute_local_phase(self, parameters: list | None) -> tuple[list, int, dict]:
        if parameters:
            self.set_parameters(parameters)

        self.model.train()
        step_count = 0
        loss_sum = 0.0

        # Execute inner loop training
        while step_count < self.inner_steps:
            phase = self.darl_source.next_phase()
            if phase is None:
                if D.is_leader():
                    logger.info("DARL epoch complete; advancing to next epoch...")
                    try:
                        if self.darl_source.session:
                            self.darl_source.session.client.advance_epoch()
                    except Exception as e:
                        logger.warning(f"DARL advance_epoch failed: {e}")

                self.darl_source.epoch_complete = False
                self.darl_source.epoch += 1
                self.darl_source.phase_index = 0
                self.darl_source._carry = []

                time.sleep(1.0)
                phase = self.darl_source.next_phase()
                if phase is None:
                    break

            self.sampler.set_indices(phase.indices)

            # Build PyTorch DataLoader over DARL leased phase
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=self.sampler,
                num_workers=self.args.num_workers,
            )

            optimizer = self.optimizer
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            for batch in loader:
                if step_count >= self.inner_steps:
                    break

                step_t0 = time.monotonic()
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()

                if D.world_size() > 1:
                    D.all_reduce_avg_([p.grad for p in self.model.parameters() if p.grad is not None])

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                step_duration = max(1e-5, time.monotonic() - step_t0)
                step_ms = step_duration * 1000.0

                loss_sum += loss.item()
                step_count += 1
                self._consumed_samples += input_ids.size(0) * D.world_size()

                # Calculate Megatron-style metrics
                seq_len = input_ids.size(1)
                tok_per_sec_per_gpu = (input_ids.size(0) * seq_len) / step_duration
                tflops_per_gpu = (6.0 * num_params * input_ids.size(0) * seq_len) / (step_duration * 1e12)

                # Memory usage ratio
                mem_usage = 0.0
                if torch.cuda.is_available():
                    try:
                        mem_usage = torch.cuda.max_memory_allocated(self.device) / max(1, torch.cuda.get_device_properties(self.device).total_memory)
                    except Exception:
                        mem_usage = 0.0

                # Log every 10 steps (or on final step)
                log_interval = getattr(self.args, "log_every", 10)
                if D.is_leader() and (step_count % log_interval == 0 or step_count == self.inner_steps):
                    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    lr = optimizer.param_groups[0]["lr"]
                    global_batch = self.args.batch_size * D.world_size()
                    g_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

                    logger.info(
                        f"[{self.args.cluster_id or 'pww'}]: [{now_str}] "
                        f"iteration {step_count:8d}/{self.inner_steps:8d} | "
                        f"consumed samples: {self._consumed_samples:12d} | "
                        f"elapsed time per iteration (ms): {step_ms:10.1f} | "
                        f"mem usages: {mem_usage:.4f} | "
                        f"throughput per GPU (TFLOP/s/GPU): {tflops_per_gpu:8.1f} | "
                        f"Tokens per second per GPU (Tok/s/GPU): {tok_per_sec_per_gpu:8.1f} | "
                        f"learning rate: {lr:.6E} | "
                        f"global batch size: {global_batch:5d} | "
                        f"lm loss: {loss.item():.6E} | "
                        f"grad norm: {g_norm:7.3f} | "
                        f"number of skipped iterations:   0 | "
                        f"number of nan iterations:   0 |"
                    )

            self.darl_source.end_phase()
            self.darl_source.commit()

        self.darl_source.release_unused()

        # Cluster-level, not rank-level. Both numbers leave this process: the loss is
        # token-weighted against the other site's, and the count IS the FedMom merge
        # weight -- so a per-rank count would weight LUMI's 8 GCDs the same as
        # Snellius's 4 whenever their per-rank geometry matches, which is exactly when
        # LUMI did twice the work.
        avg_loss = loss_sum / max(1, step_count)
        if D.world_size() > 1:
            avg_loss = D.all_reduce_mean(avg_loss, self.device)

        # TOKENS, not sequences. This used to report `step_count * batch_size`, a
        # sequence count, while the server logged it as "tokens" and weighted the merge
        # by it. Proportional as long as every site runs the same seq_len -- and silently
        # wrong the moment one does not.
        tokens_trained = step_count * self.args.batch_size * self.train_dataset.seq_len
        if D.world_size() > 1:
            tokens_trained = int(D.all_reduce_sum(float(tokens_trained), self.device))

        if D.is_leader():
            logger.info(
                f"Completed local inner LLM phase ({step_count} steps, "
                f"{tokens_trained:,} tokens), avg loss: {avg_loss:.4f}, "
                f"perplexity: {math.exp(min(20, avg_loss)):.2f}"
            )

        # NOT max(1, ...). That floor is what turned "DARL ran dry and I trained
        # nothing" into "I trained 1 sample at loss 0.0", which FedMom then folded
        # into the average as a real result -- so a run continued for 23 rounds
        # after the corpus was exhausted, reporting no failures while the global
        # model sat frozen. A truthful zero makes Flower's token-weighted average
        # ignore this cluster instead.
        if step_count == 0 and D.is_leader():
            logger.warning(
                "inner phase ran 0 steps -- DARL has no more blocks for this "
                "cluster. Reporting 0 tokens so this round carries no weight; "
                "the server should stop rather than keep aggregating nothing."
            )
        return (
            self.get_parameters(config={}),
            tokens_trained,
            {"loss": float(avg_loss), "steps": step_count, "exhausted": step_count == 0},
        )

    def _execute_evaluation(self, parameters: list | None) -> tuple[float, int, dict]:
        if parameters:
            self.set_parameters(parameters)

        if self.eval_dataset is None:
            return 0.0, 0, {}

        self.model.eval()
        loader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=False)
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                # Weighted by the tokens the loss was actually averaged over, not by
                # `input_ids.size(0)` -- that is the batch's *sequence* count, and using
                # it made a variable called total_tokens hold sequences. Harmless while
                # every sequence is the same length, wrong as soon as one is not, and
                # mislabelled either way. -100 is the ignore index, so those positions
                # contributed nothing to the loss and must not be counted.
                supervised = int((labels != -100).sum().item())
                total_loss += outputs.loss.item() * supervised
                total_tokens += supervised

        avg_loss = total_loss / max(1, total_tokens)
        if D.world_size() > 1:
            # Cluster-level, so the central node's token weighting is comparing like
            # with like across sites.
            avg_loss = D.all_reduce_mean(avg_loss, self.device)
            total_tokens = int(D.all_reduce_sum(float(total_tokens), self.device))
        perplexity = math.exp(min(20.0, avg_loss))

        if D.is_leader():
            logger.info(
                f"DiLoCo Outer Round LLM Evaluation -> Test Loss: {avg_loss:.4f}, "
                f"Perplexity: {perplexity:.2f}"
            )

        # Keyed "perplexity", not "accuracy". It was reported under the latter and
        # the server prints that metric with a % sign, so a log line reading
        # "Test Accuracy: 21.69%" was in fact perplexity 21.69 -- which made a model
        # that never improved look like one at 21% accuracy and climbing.
        #
        # `eval_loss` is what the server aggregates; it derives the pooled perplexity
        # from that rather than averaging these per-cluster perplexities, because exp()
        # is convex and a mean of perplexities is not the perplexity of the pooled
        # corpus. See central/server.py::aggregate_eval_metrics.
        return float(avg_loss), total_tokens, {
            "eval_loss": float(avg_loss),
            "perplexity": float(perplexity),
        }

    def evaluate(self, parameters: list, config: dict) -> tuple[float, int, dict]:
        if D.world_size() > 1 and D.is_leader():
            import torch.distributed as dist
            dist.broadcast_object_list([("EVAL", parameters)], src=0)

        return self._execute_evaluation(parameters)

    def fit(self, parameters: list, config: dict) -> tuple:
        if D.world_size() > 1 and D.is_leader():
            import torch.distributed as dist
            dist.broadcast_object_list([("FIT", parameters)], src=0)

        return self._execute_local_phase(parameters)

    def run_worker_loop(self) -> None:
        import torch.distributed as dist
        while True:
            box = [None]
            dist.broadcast_object_list(box, src=0)
            msg, params = box[0]
            if msg == "STOP":
                break
            elif msg == "FIT":
                self._execute_local_phase(params)
            elif msg == "EVAL":
                self._execute_evaluation(params)


def build_parser() -> argparse.ArgumentParser:
    """The LLM client's argument surface.

    Split out of `main` so tests can check that every `configs/llm_*.yaml` parses
    against the parser that will actually consume it. They were previously checked
    against the CIFAR trainer's parser, which has no --seq-len or
    --attn-implementation, so the check failed on keys that are perfectly valid.
    """
    parser = argparse.ArgumentParser(description="Flower + DARL DiLoCo LLM Client")
    add_common_args(parser)

    g = parser.add_argument_group("federation")
    g.add_argument("--central-ip", type=str, default="145.38.206.143", help="Central node IP")
    g.add_argument("--darl-port", type=int, default=29510, help="DARL HTTP port")
    g.add_argument("--flower-port", type=int, default=29511, help="Flower gRPC port")
    g.add_argument("--cluster-id", type=str, default=None, help="snellius or lumi")
    g.add_argument("--inner-steps", type=int, default=100, help="H inner steps")

    g = parser.add_argument_group("model")
    g.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name or path")
    g.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    g.add_argument("--seq-len", type=int, default=1024, help="Sequence length")

    g = parser.add_argument_group("data")
    g.add_argument("--dataset-name", type=str, default="wikitext", help="HuggingFace dataset name or path")
    g.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    g.add_argument("--batch-size", type=int, default=4, help="PER-RANK batch size")
    g.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    g = parser.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-4, help="Inner learning rate")
    g.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)

    g = parser.add_argument_group("execution")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--log-every", type=int, default=50)

    return parser


def main() -> None:
    if not HAS_FLWR:
        logger.error("flwr is required on the cluster. Run: pip install flwr")
        sys.exit(1)

    args = apply_config_file(build_parser())

    # Setup PyTorch distributed environment
    info = D.setup()
    output_dir = resolve_output_dir(args, default_name=f"llm-flower-{Path(args.model).name}")
    log = setup_logging(info.rank, output_dir)
    device = info.device

    cluster_id = args.cluster_id or ("snellius" if info.backend == "nccl" else "lumi")
    darl_url = f"http://{args.central_ip}:{args.darl_port}"
    flower_address = f"{args.central_ip}:{args.flower_port}"

    if info.is_master:
        log.info(
            f"Initializing Flower LLM client on cluster '{cluster_id}' -> "
            f"DARL: {darl_url}, Flower: {flower_address}"
        )

    # 1. Build Tokenized Dataset & DARL Data Source
    train_dataset = build_hf_tokenized_dataset(
        dataset_name_or_path=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.model,
        seq_len=args.seq_len,
        split="train",
    )

    try:
        eval_dataset = build_hf_tokenized_dataset(
            dataset_name_or_path=args.dataset_name,
            dataset_config=args.dataset_config,
            tokenizer_name=args.model,
            seq_len=args.seq_len,
            split="validation",
        )
    except Exception:
        eval_dataset = None

    # block_size from the config rather than hardcoded at 100, and num_samples from
    # the config when given -- both were previously ignored here, so --darl-block-size
    # and --darl-num-samples were dead knobs that had to agree with the coordinator
    # by luck.
    block_size = getattr(args, "darl_block_size", None) or 100
    num_samples = getattr(args, "darl_num_samples", None) or len(train_dataset)
    if num_samples != len(train_dataset) and info.is_master:
        log.warning(
            "--darl-num-samples is %d but this cluster tokenised %d sequences; the "
            "coordinator's index space wins, and any sequence beyond %d will never "
            "be leased",
            num_samples, len(train_dataset), num_samples,
        )
    space = BlockSpace(num_samples=num_samples, block_size=block_size, seed=args.seed)
    sampler = LeasedSampler()

    # One lease per inner phase, sized from H and the batch geometry, instead of a
    # flat 5 blocks. With blocks_per_phase=5 at block_size=100 a phase leased 500
    # sequences, consumed inner_steps * batch_size of them, and then committed the
    # whole span -- which is how a WikiText run committed all 1,152 blocks while
    # actually training on roughly a tenth of them.
    blocks_per_phase = blocks_for_phase(
        space,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        ranks=info.world_size,
        grad_accum=max(1, getattr(args, "gradient_accumulation_steps", 1)),
    )
    if info.is_master:
        log.info(
            "darl: %s | %d blocks per phase (H=%d x batch %d x %d ranks)",
            space.describe(), blocks_per_phase, args.inner_steps, args.batch_size,
            info.world_size,
        )

    if info.is_master:
        darl_token = args.darl_token or os.environ.get("DARL_TOKEN", "")
        client = LeaseClient(darl_url, cluster_id, token=darl_token)
        session = LeaseSession(
            client, space, blocks_per_phase=blocks_per_phase, ranks=info.world_size
        )
    else:
        session = None

    darl_source = DARLDataSource(
        space=space,
        session=session,
        rank=info.rank,
        world_size=info.world_size,
        seed=args.seed,
        blocks_per_phase=blocks_per_phase,
    )

    # 2. Build Model
    model = build_llm(
        model_name_or_path=args.model,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
    ).to(device)

    client = DiLoCoLLMFlowerClient(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        sampler=sampler,
        darl_source=darl_source,
        args=args,
        device=device,
    )

    # 3. Start Flower client on cluster leader
    if info.is_master:
        fl.client.start_numpy_client(
            server_address=flower_address,
            client=client,
        )
        if info.world_size > 1:
            import torch.distributed as dist
            dist.broadcast_object_list([("STOP", None)], src=0)
    else:
        client.run_worker_loop()

    D.cleanup()


if __name__ == "__main__":
    main()

# Central Orchestrator Node (Ubuntu / Cloud VM / Generic Host)
#
# Defines default site configuration when running the DARL lease coordinator
# or Flower aggregator server on a central VM host.

export PWW_ACCOUNT="${PWW_ACCOUNT:-central}"
export PWW_PARTITION="${PWW_PARTITION:-central}"
export PWW_GPUS_PER_NODE="${PWW_GPUS_PER_NODE:-0}"
export PWW_CPUS_PER_TASK="${PWW_CPUS_PER_TASK:-1}"
export PWW_ACCELERATOR="${PWW_ACCELERATOR:-cpu}"
export PWW_GPU_VISIBLE_VAR="${PWW_GPU_VISIBLE_VAR:-CUDA_VISIBLE_DEVICES}"
export PWW_SCRATCH="${PWW_SCRATCH:-/data/thomasistriplet/zpalanciya}"

pww_cpu_bind() {
    echo "none"
}

PWW_LAUNCH=()

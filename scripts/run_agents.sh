#!/usr/bin/env bash
# =============================================================================
# run_agents.sh  -  SLURM wrapper for run_pipeline.py
#
# Runs ALL cases found under <input_dir>/<dataset>/{free_hpo,text,vcf,image}/
# by default. In a SLURM job array, each task automatically receives one
# deterministic, non-overlapping shard; the final task receives the remainder.
# Use --case_id only if you want to debug a single non-array case.
#
# Submit example:
#
#   sbatch -p gpuq \
#          --gres=gpu:a100:1 \
#          --cpus-per-gpu=1 \
#          --mem=64G \
#          --time=5-00:00:00 \
#          --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
#                    --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                    --dataset HMS \
#                    --output_dir /home/nguyenqm/projects/RareMind/outputs \
#                    --config /home/nguyenqm/projects/RareMind/configs/default.yaml \
#                    --text_model Qwen/Qwen3-8B"
#
# Interactive test:
#   bash scripts/run_agents.sh --input_dir inputs --dataset HMS --case_id case001 --verbose
#
# Five-way array (no extra sharding flags are needed):
#   sbatch --array=1-5 ... --wrap="bash scripts/run_agents.sh --input_dir inputs --dataset HMS"
#
# Split one fixed 30-shard run across two clusters (shared inputs/outputs):
#   cluster A: sbatch --array=1-18  ... --wrap="bash scripts/run_agents.sh ... --n_shards 30"
#   cluster B: sbatch --array=19-30 ... --wrap="bash scripts/run_agents.sh ... --n_shards 30"
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults - override via command-line arguments
# ---------------------------------------------------------------------------
CONFIG="configs/default.yaml"
DATASET=""             # REQUIRED: e.g. HMS
INPUT_DIR="inputs"     # parent of the dataset folder, e.g. .../inputs
OUTPUT_DIR="outputs"
CASE_ID=""             # leave empty to run ALL cases
LIMIT=""               # leave empty for no limit
SHARD_INDEX=""         # one-based; normally derived from SLURM_ARRAY_TASK_ID
NUM_SHARDS=""          # normally derived from SLURM_ARRAY_TASK_COUNT

GENOMICS_RESULTS_DIR=""
GENOMICS_COMMAND=""
TEXT_MODEL=""          # e.g. Qwen/Qwen3-8B

# Boolean flags
VERBOSE=""
NO_VISION_PREFETCH=""
OVERWRITE_VISION_CACHE=""
NO_STAGE1_PREFETCH=""
OVERWRITE_STAGE1_CACHE=""
RESUME=""
CONTROL=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash run_agents.sh [OPTIONS]

Required:
  --dataset NAME               Dataset folder name inside input_dir, e.g. HMS
  --input_dir PATH             Parent directory of the dataset folder.
                               Pipeline reads <input_dir>/<dataset>/free_hpo|text|vcf|image/

Optional:
  --output_dir PATH            Where to write results   (default: outputs)
  --config PATH                Config YAML path         (default: configs/default.yaml)
  --limit N                    Run only first N cases   (default: all)
  --case_id ID                 Debug: run ONE case only (default: run all)
  --shard_index N              One-based global shard number
  --num_shards N               Fixed total number of global shards
  --n_shards N                 Alias for --num_shards
                               In a SLURM array, supplying only --num_shards
                               uses SLURM_ARRAY_TASK_ID as the global index.
  --genomics_results_dir PATH  Dir with Exomiser/RankVar outputs
  --genomics_command CMD       Async command template for missing VCF results
  --text_model NAME            Override cfg.models.text_llm.model_name
                               e.g. Qwen/Qwen3-8B
  --no_vision_prefetch         Skip vision prefetch; run inline instead
  --overwrite_vision_cache     Re-run vision even if cached JSON exists
  --no_stage1_prefetch         Skip batched Stage 1 text extraction
  --overwrite_stage1_cache     Re-run Stage 1 text even if cached JSON exists
  --resume                     Continue incomplete cases from saved checkpoints
  --control                    Stop after Stage 3 ranking and Stage 3.5 classification
  --verbose                    Enable debug-level logging
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                 CONFIG="$2";                shift 2 ;;
        --dataset)                DATASET="$2";               shift 2 ;;
        --input_dir)              INPUT_DIR="$2";             shift 2 ;;
        --output_dir)             OUTPUT_DIR="$2";            shift 2 ;;
        --case_id)                CASE_ID="$2";               shift 2 ;;
        --limit)                  LIMIT="$2";                 shift 2 ;;
        --shard_index|--shard-index) SHARD_INDEX="$2";        shift 2 ;;
        --num_shards|--num-shards|--n_shards|--n-shards)
                                   NUM_SHARDS="$2";           shift 2 ;;
        --genomics_results_dir)   GENOMICS_RESULTS_DIR="$2";  shift 2 ;;
        --genomics_command)       GENOMICS_COMMAND="$2";      shift 2 ;;
        --text_model)             TEXT_MODEL="$2";            shift 2 ;;
        --no_vision_prefetch)     NO_VISION_PREFETCH="1";     shift   ;;
        --overwrite_vision_cache) OVERWRITE_VISION_CACHE="1"; shift   ;;
        --no_stage1_prefetch)     NO_STAGE1_PREFETCH="1";     shift   ;;
        --overwrite_stage1_cache) OVERWRITE_STAGE1_CACHE="1"; shift   ;;
        --resume)                  RESUME="1";                 shift   ;;
        --control)                 CONTROL="1";                shift   ;;
        --verbose)                VERBOSE="1";                shift   ;;
        -h|--help)                usage; exit 0               ;;
        *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "[ERROR] --dataset is required."
    usage
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# SLURM array -> deterministic pipeline shard
# ---------------------------------------------------------------------------
# With an explicitly supplied global shard count, preserve the physical SLURM
# task ID. This lets disjoint array ranges on separate clusters cover one fixed
# partition, for example 1-18 and 19-30 with --n_shards 30. Without an explicit
# total, retain the convenient behavior of mapping the current array to 1..N.
if [[ -z "$SHARD_INDEX" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ -n "$NUM_SHARDS" ]]; then
        SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
    else
        ARRAY_MIN="${SLURM_ARRAY_TASK_MIN:-1}"
        ARRAY_STEP="${SLURM_ARRAY_TASK_STEP:-1}"
        ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-}"
        if [[ -z "$ARRAY_COUNT" ]]; then
            ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-$SLURM_ARRAY_TASK_ID}"
            ARRAY_COUNT=$(( (ARRAY_MAX - ARRAY_MIN) / ARRAY_STEP + 1 ))
        fi
        ARRAY_OFFSET=$(( SLURM_ARRAY_TASK_ID - ARRAY_MIN ))
        if (( ARRAY_OFFSET < 0 || ARRAY_OFFSET % ARRAY_STEP != 0 )); then
            echo "[ERROR] Invalid SLURM array metadata: task=$SLURM_ARRAY_TASK_ID min=$ARRAY_MIN step=$ARRAY_STEP"
            exit 2
        fi
        SHARD_INDEX=$(( ARRAY_OFFSET / ARRAY_STEP + 1 ))
        NUM_SHARDS="$ARRAY_COUNT"
    fi
fi

if [[ -n "$SHARD_INDEX" || -n "$NUM_SHARDS" ]]; then
    if [[ -z "$SHARD_INDEX" || -z "$NUM_SHARDS" ]]; then
        echo "[ERROR] --shard_index and --num_shards must be supplied together."
        exit 2
    fi
    if [[ -n "$CASE_ID" ]]; then
        echo "[ERROR] --case_id cannot be used in a SLURM array or with explicit sharding."
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "======================================================="
echo " RareMind Pipeline"
echo " Dataset : $DATASET"
echo " Input   : $INPUT_DIR/$DATASET"
echo " Output  : $OUTPUT_DIR/$DATASET"
echo " Repo    : $REPO_ROOT"
echo " Started : $(date)"
echo " Host    : $(hostname)"
echo " SLURM job ID: ${SLURM_JOB_ID:-<interactive>}"
echo " SLURM RAM/node (MB): ${SLURM_MEM_PER_NODE:-<not set>}"
echo " SLURM RAM/CPU  (MB): ${SLURM_MEM_PER_CPU:-<not set>}"
if [[ -n "$SHARD_INDEX" ]]; then
    echo " Shard   : $SHARD_INDEX / $NUM_SHARDS"
else
    echo " Shard   : none (entire dataset)"
fi
if [[ -n "$CONTROL" ]]; then
    echo " Mode    : control (stop after Stage 3.5)"
else
    echo " Mode    : full pipeline"
fi
echo "======================================================="

if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "-----------"
fi

# ---------------------------------------------------------------------------
# Build the python argument list
# ---------------------------------------------------------------------------
PYTHON_ARGS=(
    --config     "$CONFIG"
    --input_dir  "$INPUT_DIR"
    --dataset    "$DATASET"
    --output_dir "$OUTPUT_DIR"
)

[[ -n "$CASE_ID"              ]] && PYTHON_ARGS+=(--case_id             "$CASE_ID")
[[ -n "$LIMIT"                ]] && PYTHON_ARGS+=(--limit               "$LIMIT")
[[ -n "$SHARD_INDEX"          ]] && PYTHON_ARGS+=(--shard_index         "$SHARD_INDEX")
[[ -n "$NUM_SHARDS"           ]] && PYTHON_ARGS+=(--num_shards          "$NUM_SHARDS")
[[ -n "$GENOMICS_RESULTS_DIR" ]] && PYTHON_ARGS+=(--genomics_results_dir "$GENOMICS_RESULTS_DIR")
[[ -n "$GENOMICS_COMMAND"     ]] && PYTHON_ARGS+=(--genomics_command    "$GENOMICS_COMMAND")
[[ -n "$TEXT_MODEL"           ]] && PYTHON_ARGS+=(--text_model          "$TEXT_MODEL")

[[ -n "$NO_VISION_PREFETCH"     ]] && PYTHON_ARGS+=(--no_vision_prefetch)
[[ -n "$OVERWRITE_VISION_CACHE" ]] && PYTHON_ARGS+=(--overwrite_vision_cache)
[[ -n "$NO_STAGE1_PREFETCH"     ]] && PYTHON_ARGS+=(--no_stage1_prefetch)
[[ -n "$OVERWRITE_STAGE1_CACHE" ]] && PYTHON_ARGS+=(--overwrite_stage1_cache)
[[ -n "$RESUME"                 ]] && PYTHON_ARGS+=(--resume)
[[ -n "$CONTROL"                ]] && PYTHON_ARGS+=(--control)
[[ -n "$VERBOSE"                ]] && PYTHON_ARGS+=(--verbose)

echo ""
echo "Command: python scripts/run_pipeline.py ${PYTHON_ARGS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
python scripts/run_pipeline.py "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

echo ""
echo "======================================================="
echo " Finished : $(date)"
echo " Exit code: $EXIT_CODE"
echo "======================================================="
exit $EXIT_CODE

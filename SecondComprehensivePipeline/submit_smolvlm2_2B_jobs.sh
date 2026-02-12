#!/bin/bash
# Submit all SmolVLM2 2.2B training strategies as separate SLURM jobs
# Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
# WandB project: 23ComprehensivePipeline
# Account: master, Time: 71h
#
# Usage:
#   ./submit_smolvlm2_2B_jobs.sh                    # Submit all jobs
#   ./submit_smolvlm2_2B_jobs.sh --dry-run          # Show what would be submitted without actually submitting
#   ./submit_smolvlm2_2B_jobs.sh --limit 5          # Only submit first 5 jobs (for testing)

set -e

# Parse arguments
DRY_RUN=false
LIMIT=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--limit N]"
            exit 1
            ;;
    esac
done

# Configuration
CONFIG_DIR="config/individual"
JOB_SCRIPT="job_template.sh"
LOG_DIR="izarlogs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Check if job template exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi

# SmolVLM2 2.2B config files (12 total)
CONFIG_FILES=(
    # Full FT Benchmark (3)
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_ocrbench.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_docvqa.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_chartqa.yaml"
    # Full FT ERP (6)
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_chosen_nova.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_chosen_gemini.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_qcm_nova.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_qcm_gemini.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_qcm_procedure1.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_full_sft_qcm_procedure2.yaml"
    # QLoRA Benchmark (3)
    "$CONFIG_DIR/conf_smolvlm2_2B_qlora_sft_ocrbench.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_qlora_sft_docvqa.yaml"
    "$CONFIG_DIR/conf_smolvlm2_2B_qlora_sft_chartqa.yaml"
)

# Filter to only existing files
VALID_FILES=()
for f in "${CONFIG_FILES[@]}"; do
    if [ -f "$f" ]; then
        VALID_FILES+=("$f")
    fi
done

if [ ${#VALID_FILES[@]} -eq 0 ]; then
    echo "Error: No SmolVLM2 2.2B config files found"
    exit 1
fi

echo "=========================================="
echo "SmolVLM2 2.2B Training Job Submission"
echo "=========================================="
echo "Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct"
echo "Found ${#VALID_FILES[@]} config files"
echo "Job script: $JOB_SCRIPT"
echo "Config directory: $CONFIG_DIR"
echo "WandB project: 23ComprehensivePipeline"
echo "SLURM account: master, time: 71h"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
fi

if [ $LIMIT -gt 0 ]; then
    echo "LIMIT: Only submitting first $LIMIT jobs"
fi

echo "=========================================="
echo ""

# List configs
echo "Configs to submit:"
for config_file in "${VALID_FILES[@]}"; do
    strategy_name=$(basename "$config_file" | sed 's/conf_//' | sed 's/.yaml//')
    echo "  - $strategy_name"
done
echo ""

# Submit jobs
SUBMITTED=0
SKIPPED=0

for config_file in "${VALID_FILES[@]}"; do
    # Extract strategy name from config filename
    strategy_name=$(basename "$config_file" | sed 's/conf_//' | sed 's/.yaml//')

    echo "[$((SUBMITTED + SKIPPED + 1))/${#VALID_FILES[@]}] $strategy_name"

    # Check limit
    if [ $LIMIT -gt 0 ] && [ $SUBMITTED -ge $LIMIT ]; then
        echo "  -> Skipping (limit reached)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Submit or show command
    if [ "$DRY_RUN" = true ]; then
        echo "  -> Would submit: sbatch --job-name=$strategy_name $JOB_SCRIPT $config_file"
        SUBMITTED=$((SUBMITTED + 1))
    else
        # Submit with custom job name
        job_output=$(sbatch --job-name="$strategy_name" "$JOB_SCRIPT" "$config_file" 2>&1)

        if [ $? -eq 0 ]; then
            job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
            echo "  -> Submitted: Job ID $job_id"
            SUBMITTED=$((SUBMITTED + 1))

            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        else
            echo "  -> Failed: $job_output"
            SKIPPED=$((SKIPPED + 1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total configs: ${#VALID_FILES[@]}"
echo "Submitted: $SUBMITTED"
echo "Skipped: $SKIPPED"
echo "=========================================="

if [ "$DRY_RUN" = false ] && [ $SUBMITTED -gt 0 ]; then
    echo ""
    echo "Monitor jobs with: squeue -u \$USER"
    echo "Cancel all jobs: scancel -u \$USER"
    echo "View logs in: $LOG_DIR/"
fi

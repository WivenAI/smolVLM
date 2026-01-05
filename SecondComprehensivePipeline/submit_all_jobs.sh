#!/bin/bash
# Submit all training strategies as separate SLURM jobs
#
# Usage:
#   ./submit_all_jobs.sh                    # Submit all jobs
#   ./submit_all_jobs.sh --dry-run          # Show what would be submitted without actually submitting
#   ./submit_all_jobs.sh --limit 5          # Only submit first 5 jobs (for testing)

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
    echo "Please run 'python generate_configs.py' first"
    exit 1
fi

# Check if job template exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi

# Find all config files
CONFIG_FILES=($(ls "$CONFIG_DIR"/conf_*.yaml | grep -v summary))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run 'python generate_configs.py' first"
    exit 1
fi

echo "=========================================="
echo "SLURM Job Submission"
echo "=========================================="
echo "Found ${#CONFIG_FILES[@]} config files"
echo "Job script: $JOB_SCRIPT"
echo "Config directory: $CONFIG_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
fi

if [ $LIMIT -gt 0 ]; then
    echo "LIMIT: Only submitting first $LIMIT jobs"
fi

echo "=========================================="
echo ""

# Submit jobs
SUBMITTED=0
SKIPPED=0

for config_file in "${CONFIG_FILES[@]}"; do
    # Extract strategy name from config filename
    # Example: config/individual/conf_baseline.yaml -> baseline
    strategy_name=$(basename "$config_file" | sed 's/conf_//' | sed 's/.yaml//')

    echo "[$((SUBMITTED + SKIPPED + 1))/${#CONFIG_FILES[@]}] $strategy_name"

    # Skip full DPO strategies
    if [[ "$strategy_name" == "full_ft_dpo_gemini" ]] || [[ "$strategy_name" == "full_ft_dpo_nova" ]]; then
        echo "  ⊘ Skipping (full DPO excluded)"
        ((SKIPPED++))
        continue
    fi

    # Check limit
    if [ $LIMIT -gt 0 ] && [ $SUBMITTED -ge $LIMIT ]; then
        echo "  ⊘ Skipping (limit reached)"
        ((SKIPPED++))
        continue
    fi

    # Submit or show command
    if [ "$DRY_RUN" = true ]; then
        echo "  → Would submit: sbatch --job-name=wiven_$strategy_name $JOB_SCRIPT $config_file"
        ((SUBMITTED++))
    else
        # Submit with custom job name
        job_output=$(sbatch --job-name="wiven_$strategy_name" "$JOB_SCRIPT" "$config_file" 2>&1)

        if [ $? -eq 0 ]; then
            job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
            echo "  ✓ Submitted: Job ID $job_id"
            ((SUBMITTED++))

            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        else
            echo "  ✗ Failed: $job_output"
            ((SKIPPED++))
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total configs: ${#CONFIG_FILES[@]}"
echo "Submitted: $SUBMITTED"
echo "Skipped: $SKIPPED"
echo "=========================================="

if [ "$DRY_RUN" = false ] && [ $SUBMITTED -gt 0 ]; then
    echo ""
    echo "Monitor jobs with: squeue -u \$USER"
    echo "Cancel all jobs: scancel -u \$USER"
    echo "View logs in: $LOG_DIR/"
fi

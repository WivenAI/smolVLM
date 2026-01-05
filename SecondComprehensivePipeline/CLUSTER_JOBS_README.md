# Running Training Strategies as Separate Cluster Jobs

This guide explains how to run each training strategy as a separate SLURM job on the cluster.

## Overview

Instead of running all training strategies in a single long job, this system allows you to:
- Run each strategy as an independent job
- Submit all jobs at once or selectively
- Better utilize cluster resources
- Easier to track and debug individual strategies
- Restart failed jobs without rerunning everything

## Quick Start

### 1. Generate Individual Config Files

First, generate separate config files for each enabled strategy:

```bash
python generate_configs.py
```

This will create individual config files in `config/individual/`:
- `conf_baseline.yaml`
- `conf_full_ft_docvqa.yaml`
- `conf_full_ft_ocrbench.yaml`
- ... (one for each enabled strategy)

### 2. Submit All Jobs

Submit all training jobs to the cluster:

```bash
chmod +x submit_all_jobs.sh
./submit_all_jobs.sh
```

### 3. Monitor Jobs

Check your submitted jobs:

```bash
squeue -u $USER
```

View logs (as jobs run):

```bash
tail -f izarlogs/output_wiven_*
```

## Advanced Usage

### Dry Run (Preview Without Submitting)

See what jobs would be submitted without actually submitting them:

```bash
./submit_all_jobs.sh --dry-run
```

### Submit Limited Number of Jobs (Testing)

Submit only the first 5 jobs (useful for testing):

```bash
./submit_all_jobs.sh --limit 5
```

### Submit a Single Strategy

Submit just one specific strategy:

```bash
sbatch --job-name=wiven_baseline job_template.sh config/individual/conf_baseline.yaml
```

### Custom Config Source

Generate configs from a different base config:

```bash
python generate_configs.py --config config/conf_custom.yaml --output config/custom_individual
```

## File Structure

```
.
├── config/
│   ├── conf.yaml                    # Main config with all strategies
│   └── individual/                  # Generated individual configs
│       ├── conf_baseline.yaml
│       ├── conf_full_ft_docvqa.yaml
│       └── ...
│
├── generate_configs.py              # Generate individual configs
├── job_template.sh                  # SLURM job template (accepts config path)
├── submit_all_jobs.sh               # Submit all jobs at once
└── job.sh                           # Original job script (still works)
```

## Managing Jobs

### Cancel All Jobs

```bash
scancel -u $USER
```

### Cancel Specific Job

```bash
scancel <job_id>
```

### Cancel Jobs by Name Pattern

```bash
scancel --name=wiven_full_ft*
```

### Check Job Status

```bash
squeue -u $USER -o "%.10i %.9P %.30j %.8T %.10M %.6D %R"
```

### View Job Details

```bash
scontrol show job <job_id>
```

## Logs

All logs are saved in `izarlogs/`:
- `output_wiven_<strategy_name>_<job_id>.txt` - Standard output
- `error_wiven_<strategy_name>_<job_id>.txt` - Error output

## Customization

### Modify Job Parameters

Edit `job_template.sh` to change:
- Time limit: `#SBATCH --time 71:00:00`
- GPUs: `#SBATCH --gres=gpu:1`
- CPUs: `#SBATCH --cpus-per-task 5`
- Memory: Add `#SBATCH --mem=32G`
- Partition: Add `#SBATCH --partition=gpu`

### Selective Strategy Submission

To submit only specific strategies:

1. Edit `config/conf.yaml` and set `enabled: false` for strategies you don't want
2. Regenerate configs: `python generate_configs.py`
3. Submit: `./submit_all_jobs.sh`

Or manually submit individual configs:

```bash
sbatch --job-name=wiven_strategy1 job_template.sh config/individual/conf_strategy1.yaml
sbatch --job-name=wiven_strategy2 job_template.sh config/individual/conf_strategy2.yaml
```

## Workflow Example

Complete workflow for running experiments:

```bash
# 1. Configure your strategies in config/conf.yaml
vim config/conf.yaml

# 2. Generate individual configs
python generate_configs.py

# 3. Preview what will be submitted
./submit_all_jobs.sh --dry-run

# 4. Test with a few jobs first
./submit_all_jobs.sh --limit 3

# 5. Check they're running correctly
squeue -u $USER
tail -f izarlogs/output_wiven_*

# 6. If all looks good, submit remaining jobs
./submit_all_jobs.sh

# 7. Monitor progress
watch -n 60 squeue -u $USER

# 8. When complete, check results
ls results/
```

## Troubleshooting

### "No config files found"

Run `python generate_configs.py` first to generate the configs.

### Jobs fail immediately

Check error logs:
```bash
cat izarlogs/error_wiven_*
```

Common issues:
- Wrong paths in config (check absolute paths match cluster location)
- Missing dependencies
- CUDA/GPU issues

### Out of disk space

Check your scratch space usage:
```bash
du -sh /scratch/izar/dlacour/*
```

Clean old cache:
```bash
rm -rf /scratch/izar/dlacour/hf_cache/*
```

### Too many jobs queued

The cluster may have limits on total jobs. Submit in batches:
```bash
./submit_all_jobs.sh --limit 10
# Wait for some to complete, then submit more
```

## Tips

1. **Start small**: Use `--limit` to test with a few jobs first
2. **Monitor closely**: Keep an eye on first few jobs to catch config issues early
3. **Use wandb**: Results are logged to WandB for easy comparison
4. **Clean up**: Remove old model weights and caches periodically
5. **Baseline first**: Submit baseline job first to verify setup works

## Contact

If you encounter issues, check:
- SLURM documentation: `man sbatch`
- Cluster documentation
- Pipeline logs in `logs/`

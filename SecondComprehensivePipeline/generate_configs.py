#!/usr/bin/env python3
"""
Generate individual config files for each enabled training strategy.
This allows running each strategy as a separate cluster job.

Usage:
    python generate_configs.py
    python generate_configs.py --config config/conf.yaml
"""

import yaml
import argparse
from pathlib import Path
import copy


def generate_individual_configs(config_path: str, output_dir: str = "config/individual"):
    """Generate individual config files for each enabled strategy"""

    # Load main config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all enabled strategies
    strategies = base_config.get("training", {}).get("strategies", [])
    enabled_strategies = [s for s in strategies if s.get("enabled", True)]

    print(f"Found {len(enabled_strategies)} enabled strategies")
    print(f"Generating individual configs in: {output_path}")
    print("-" * 80)

    generated_configs = []

    for strategy in enabled_strategies:
        strategy_name = strategy["name"]

        # Create a copy of the base config
        strategy_config = copy.deepcopy(base_config)

        # Disable all strategies except this one
        for s in strategy_config["training"]["strategies"]:
            s["enabled"] = (s["name"] == strategy_name)

        # Generate output filename
        config_filename = f"conf_{strategy_name}.yaml"
        config_filepath = output_path / config_filename

        # Save the config
        with open(config_filepath, 'w') as f:
            yaml.dump(strategy_config, f, default_flow_style=False, sort_keys=False)

        generated_configs.append({
            "name": strategy_name,
            "type": strategy["type"],
            "config_file": str(config_filepath)
        })

        print(f"✓ {strategy_name:40s} -> {config_filename}")

    print("-" * 80)
    print(f"Generated {len(generated_configs)} config files")

    # Save a summary file
    summary_file = output_path / "configs_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(generated_configs, f, default_flow_style=False)

    print(f"\nSummary saved to: {summary_file}")

    return generated_configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual config files for each training strategy"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/conf.yaml",
        help="Path to main config file (default: config/conf.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/individual",
        help="Output directory for generated configs (default: config/individual)"
    )

    args = parser.parse_args()

    # Generate configs
    generate_individual_configs(args.config, args.output)


if __name__ == "__main__":
    main()

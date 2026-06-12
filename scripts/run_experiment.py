import argparse
from pathlib import Path

from src.utils.config import load_config, print_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run semantic PEFT FL experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print_config(config)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed set to: {seed}")
    print(f"Output directory created: {output_dir}")

    print("\nStep 1 completed successfully.")
    print("Next step will be dataset loading and client partitioning.")


if __name__ == "__main__":
    main()

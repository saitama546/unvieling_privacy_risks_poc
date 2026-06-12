import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path so imports from src work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import models
from src.models import global_model
from src.utils.config import load_config, print_config
from src.utils.seed import set_seed
from src.datasets.dataset_loader import build_federated_dataloaders
from src.models.global_model import build_all_global_models, print_model_summary



def check_batch(batch_images: torch.Tensor, batch_labels: torch.Tensor) -> None:
    """
    Basic sanity checks for one batch.
    """
    assert isinstance(batch_images, torch.Tensor), "Images must be a torch.Tensor"
    assert isinstance(batch_labels, torch.Tensor), "Labels must be a torch.Tensor"

    assert batch_images.ndim == 4, (
        f"Expected images shape [B, C, H, W], got {batch_images.shape}"
    )

    assert batch_images.shape[1] == 3, (
        f"Expected 3 image channels, got {batch_images.shape[1]}"
    )

    assert batch_labels.ndim == 1, (
        f"Expected labels shape [B], got {batch_labels.shape}"
    )

    assert batch_images.shape[0] == batch_labels.shape[0], (
        "Batch size mismatch between images and labels"
    )

    assert torch.is_floating_point(batch_images), "Images must be floating point tensors"
    assert batch_labels.dtype == torch.long, "Labels must be torch.long"

    assert batch_images.min() >= 0.0, "Images should be normalized to [0, 1]"
    assert batch_images.max() <= 1.0, "Images should be normalized to [0, 1]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run semantic PEFT FL experiment sanity check"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    # Step 1: Load config
    config = load_config(args.config)
    print_config(config)

    # Step 1: Set seed
    seed = config["experiment"].get("seed", 0)
    set_seed(seed)

    # Step 1: Create output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Seed set to: {seed}")
    print(f"Output directory: {output_dir}")

    # Step 2: Build federated dataloaders
    client_train_loaders, dataset_info = build_federated_dataloaders(config)

    print("\n========== Dataset Info ==========")
    for key, value in dataset_info.items():
        if key == "class_names":
            print(f"{key}: {value[:10]}{' ...' if len(value) > 10 else ''}")
        else:
            print(f"{key}: {value}")
    print("==================================\n")

    # Sanity checks
    num_clients = dataset_info["num_clients"]

    assert len(client_train_loaders) == num_clients, (
        f"Expected {num_clients} client loaders, "
        f"got {len(client_train_loaders)}"
    )

    assert dataset_info["server_has_data"] is False, (
        "Server should have zero direct dataset access"
    )

    assert sum(dataset_info["client_sizes"]) == dataset_info["train_size"], (
        "Client dataset sizes must sum to full training dataset size"
    )

    print("========== Client Loader Check ==========")

    for client_id, loader in enumerate(client_train_loaders):
        print(f"\nClient {client_id}")
        print(f"  Number of samples: {len(loader.dataset)}")
        print(f"  Number of batches: {len(loader)}")

        batch_images, batch_labels = next(iter(loader))
        check_batch(batch_images, batch_labels)

        print(f"  Batch image shape: {batch_images.shape}")
        print(f"  Batch label shape: {batch_labels.shape}")
        print(f"  Labels: {batch_labels.tolist()}")
        print(f"  Image min: {batch_images.min().item():.4f}")
        print(f"  Image max: {batch_images.max().item():.4f}")
        print(f"  Image dtype: {batch_images.dtype}")
        print(f"  Label dtype: {batch_labels.dtype}")

    print("\n=========================================\n")

    print("Step 2 sanity check completed successfully.")
    print("Next step: build model factory from config.")
    print("\n=========================================\n")

    models = build_all_global_models(config, dataset_info)

    for peft_method, model in models.items():
        print(f"\nBuilt model for PEFT method: {peft_method}")
        print_model_summary(model)

if __name__ == "__main__":
    main()
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_loader import build_federated_dataloaders
from src.models.global_model import (
    build_global_model,
    build_all_global_models,
    print_model_summary,
)
from src.federated.server import FLServer
from src.federated.client import FLClient


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Description:
        Load experiment configuration from a YAML file.

    INPUTS:
        config_path (str): Path to the YAML configuration file.

    OUTPUTS:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed: int) -> None:
    """
    Description:
        Set random seeds for reproducibility.

    INPUTS:
        seed (int): Random seed value.

    OUTPUTS:
        None.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Description:
        Print important configuration values.

    INPUTS:
        config (Dict[str, Any]): Full experiment configuration.

    OUTPUTS:
        None.
    """
    print("\n========== Experiment Config ==========")

    for section, values in config.items():
        print(f"\n[{section}]")

        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")

    print("=======================================\n")


def print_dataset_summary(dataset_info: Dict[str, Any]) -> None:
    """
    Description:
        Print dataset and federated split information.

    INPUTS:
        dataset_info (Dict[str, Any]): Dataset metadata returned by dataset loader.

    OUTPUTS:
        None.
    """
    print("\n========== Dataset Summary ==========")

    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    print("=====================================\n")


def test_client_batch(client_train_loaders) -> None:
    """
    Description:
        Check whether one batch can be loaded from client 0.

    INPUTS:
        client_train_loaders (List[DataLoader]): List of client train DataLoaders.

    OUTPUTS:
        None.
    """
    images, labels = next(iter(client_train_loaders[0]))

    print("\n========== Client Batch Sanity Check ==========")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First label: {labels[0].item()}")
    print("================================================\n")


def test_fedsgd_client_gradient(
    config: Dict[str, Any],
    client_train_loaders,
    dataset_info: Dict[str, Any],
    peft_method: str,
) -> None:
    """
    Description:
        Test one FedSGD gradient computation for one PEFT method.

        This checks whether:
            1. Server can build and store a global model.
            2. Client can receive a model copy.
            3. Client can run forward and backward pass.
            4. Client returns gradients only.
            5. Private images and labels are not shared.

    INPUTS:
        config (Dict[str, Any]): Full experiment configuration.
        client_train_loaders (List[DataLoader]): Client private train loaders.
        dataset_info (Dict[str, Any]): Dataset metadata.
        peft_method (str): PEFT method to test, for example "adapter" or "lora".

    OUTPUTS:
        None.
    """
    print(f"\n========== Testing FedSGD Gradient: {peft_method} ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = build_global_model(
        config=config,
        dataset_info=dataset_info,
        peft_method=peft_method,
    )

    print_model_summary(global_model)

    server = FLServer(
        global_model=global_model,
        config=config,
        device=device,
    )

    server.print_server_summary()

    client = FLClient(
        client_id=0,
        train_loader=client_train_loaders[0],
        config=config,
        device=device,
    )

    client_model = server.get_model_copy()
    client.set_model(client_model)

    client.print_client_summary()

    gradient_package = client.compute_fedsgd_gradient()

    print("\nGradient package keys:")
    print(gradient_package.keys())

    print("\nClient ID:", gradient_package["client_id"])
    print("Loss:", gradient_package["loss"])
    print("Share type:", gradient_package["share_type"])

    gradients = gradient_package["gradients"]

    print("\nNumber of shared gradient tensors:", len(gradients))

    if len(gradients) == 0:
        raise RuntimeError("No gradients were returned. Check PEFT parameter selection.")

    print("\nShared gradient tensors:")

    for name, grad in gradients.items():
        grad_norm = grad.norm().item()

        print(
            f"{name}: "
            f"shape={tuple(grad.shape)}, "
            f"norm={grad_norm:.6f}, "
            f"device={grad.device}"
        )

        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN found in gradient: {name}")

    assert "images" not in gradient_package, "Private images should not be shared."
    assert "labels" not in gradient_package, "Private labels should not be shared."

    print("\nPrivacy check passed: images and labels are not shared.")
    print(f"FedSGD gradient test passed for PEFT method: {peft_method}")
    print("============================================================\n")


def run_experiment(config_path: str) -> None:
    """
    Description:
        Main experiment runner.

        Current version tests:
            Step 1: Config loading and seed setup.
            Step 2: Federated dataset loading.
            Step 3: Global PEFT model creation.
            Step 5: Server-client setup.
            Step 6: FedSGD PEFT gradient computation.

    INPUTS:
        config_path (str): Path to experiment YAML config.

    OUTPUTS:
        None.
    """
    config = load_config(config_path)

    seed = config["experiment"].get("seed", 0)
    set_seed(seed)

    print_config_summary(config)

    client_train_loaders, dataset_info = build_federated_dataloaders(config)

    print_dataset_summary(dataset_info)

    test_client_batch(client_train_loaders)

    peft_methods = config["peft"].get("methods", ["adapter"])

    print("\n========== PEFT Methods to Test ==========")
    print(peft_methods)
    print("==========================================\n")

    for peft_method in peft_methods:
        test_fedsgd_client_gradient(
            config=config,
            client_train_loaders=client_train_loaders,
            dataset_info=dataset_info,
            peft_method=peft_method,
        )

    print("\nAll FedSGD client gradient tests passed.")


def parse_args():
    """
    Description:
        Parse command-line arguments.

    INPUTS:
        None.

    OUTPUTS:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config)

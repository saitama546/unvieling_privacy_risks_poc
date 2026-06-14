import argparse
from typing import Any, Dict
import os
import sys
import torch
import yaml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.datasets.dataset_loader import build_federated_dataloaders
from src.models.global_model import (
    build_global_model,
    print_model_summary,
)
from src.federated.server import FLServer
from src.federated.client import FLClient
from src.attacks.gradient_inversion import reconstruct_gradient_only


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Description:
        Load experiment configuration from a YAML file.

    INPUTS:
        config_path (str): Path to YAML config file.

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
        seed (int): Random seed.

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
        Print experiment config.

    INPUTS:
        config (Dict[str, Any]): Full experiment config.

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
        Print dataset metadata.

    INPUTS:
        dataset_info (Dict[str, Any]): Dataset information.

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
        Check that one client batch can be loaded.

    INPUTS:
        client_train_loaders (List[DataLoader]): Client train loaders.

    OUTPUTS:
        None.
    """
    images, labels = next(iter(client_train_loaders[0]))

    print("\n========== Client Batch Sanity Check ==========")
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"First label: {labels[0].item()}")
    print("================================================\n")


def step8_extract_attack_gradient_and_reference(
    server: FLServer,
    client: FLClient,
    victim_client_id: int,
    attack_round_id: int,
) -> Dict[str, Any]:
    """
    Description:
        Step 8: Select victim client and attack round, extract the observed
        gradient from the server, and retrieve the private reference batch
        from the client for later evaluation.

        The server-side package must contain gradients only.

    INPUTS:
        server (FLServer): Server storing received client gradients.
        client (FLClient): Victim client.
        victim_client_id (int): Selected victim client ID.
        attack_round_id (int): Selected FL round ID.

    OUTPUTS:
        Dict[str, Any]: Attack data containing observed gradients and
            private reference batch for later metrics.
    """
    print("\n========== Step 8: Attack Gradient Extraction Check ==========")

    gradient_package = server.get_single_client_gradient(
        client_id=victim_client_id,
        round_id=attack_round_id,
    )

    assert "images" not in gradient_package, "Server package leaked private images."
    assert "labels" not in gradient_package, "Server package leaked private labels."

    observed_gradients = gradient_package["gradients"]

    if len(observed_gradients) == 0:
        raise RuntimeError("Observed gradient dictionary is empty.")

    print(f"Victim client ID: {victim_client_id}")
    print(f"Attack round ID: {attack_round_id}")
    print(f"Share type: {gradient_package['share_type']}")
    print(f"Number of observed gradient tensors: {len(observed_gradients)}")

    for name, grad in observed_gradients.items():
        print(
            f"{name}: "
            f"shape={tuple(grad.shape)}, "
            f"norm={grad.norm().item():.6f}, "
            f"device={grad.device}"
        )

        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN found in observed gradient: {name}")

        if grad.norm().item() == 0.0:
            print(f"Warning: gradient norm is zero for {name}")

    private_reference_batch = client.get_last_private_batch()

    reference_images = private_reference_batch["images"]
    reference_labels = private_reference_batch["labels"]

    print("\nPrivate reference batch retained for evaluation only.")
    print(f"Reference images shape: {tuple(reference_images.shape)}")
    print(f"Reference labels shape: {tuple(reference_labels.shape)}")
    print(f"Reference labels: {reference_labels.tolist()}")

    attack_data = {
        "victim_client_id": victim_client_id,
        "attack_round_id": attack_round_id,
        "observed_gradients": observed_gradients,
        "reference_images": reference_images,
        "reference_labels": reference_labels,
        "share_type": gradient_package["share_type"],
    }

    print("Step 8 passed.")
    print("=============================================================\n")

    return attack_data


def test_fedsgd_client_gradient(
    config: Dict[str, Any],
    client_train_loaders,
    dataset_info: Dict[str, Any],
    peft_method: str,
) -> None:
    """
    Description:
        Test one FedSGD client gradient computation and run Step 8 and Step 9.

    INPUTS:
        config (Dict[str, Any]): Full experiment config.
        client_train_loaders (List[DataLoader]): Client private train loaders.
        dataset_info (Dict[str, Any]): Dataset metadata.
        peft_method (str): PEFT method, for example "adapter" or "lora".

    OUTPUTS:
        None.
    """
    print(f"\n========== Testing PEFT Method: {peft_method} ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 3: Build global PEFT model.
    global_model = build_global_model(
        config=config,
        dataset_info=dataset_info,
        peft_method=peft_method,
    )

    print_model_summary(global_model)

    # Step 5: Create server.
    server = FLServer(
        global_model=global_model,
        config=config,
        device=device,
    )

    server.print_server_summary()

    # Step 5: Create one victim client.
    client = FLClient(
        client_id=0,
        train_loader=client_train_loaders[0],
        config=config,
        device=device,
    )

    # Server sends model copy to client.
    client_model = server.get_model_copy()
    client.set_model(client_model)

    client.print_client_summary()

    # Step 6: Client computes FedSGD gradient.
    gradient_package = client.compute_fedsgd_gradient()

    print("\n========== Step 6: Client Gradient Package ==========")
    print("Gradient package keys:", gradient_package.keys())
    print("Client ID:", gradient_package["client_id"])
    print("Loss:", gradient_package["loss"])
    print("Share type:", gradient_package["share_type"])

    gradients = gradient_package["gradients"]

    print("Number of shared gradient tensors:", len(gradients))

    if len(gradients) == 0:
        raise RuntimeError("No gradients were returned.")

    for name, grad in gradients.items():
        print(
            f"{name}: "
            f"shape={tuple(grad.shape)}, "
            f"norm={grad.norm().item():.6f}, "
            f"device={grad.device}"
        )

        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN found in gradient: {name}")

    assert "images" not in gradient_package
    assert "labels" not in gradient_package

    print("Step 6 passed: client shared gradients only.")
    print("=====================================================\n")

    # Step 7: Server receives and stores client gradient.
    server.receive_client_gradient(gradient_package)
    server.print_received_gradient_summary()

    # Step 8: Extract victim gradient and private reference batch.
    attack_data = step8_extract_attack_gradient_and_reference(
        server=server,
        client=client,
        victim_client_id=0,
        attack_round_id=server.get_current_round(),
    )

    # Step 9: Gradient-only reconstruction.
    # Important: this must happen before server.run_fedsgd_server_update(),
    # because the attack model must match the model used to compute client gradients.
    reconstruction_result = reconstruct_gradient_only(
        model=server.get_model_copy(),
        observed_gradients=attack_data["observed_gradients"],
        labels=attack_data["reference_labels"],
        reference_shape=tuple(attack_data["reference_images"].shape),
        config=config,
        share_type=attack_data["share_type"],
    )

    print("\n========== Step 9 Result ==========")
    print(
        "Reconstructed image shape:",
        tuple(reconstruction_result["reconstructed_image"].shape),
    )
    print("Final gradient loss:", reconstruction_result["final_gradient_loss"])
    print("Final TV loss:", reconstruction_result["final_tv_loss"])
    print("===================================\n")

    print(f"Finished test for PEFT method: {peft_method}")


def run_experiment(config_path: str) -> None:
    """
    Description:
        Main experiment runner.

    INPUTS:
        config_path (str): Path to experiment YAML config.

    OUTPUTS:
        None.
    """
    # Step 1: Config loading and seed setup.
    config = load_config(config_path)

    seed = config["experiment"].get("seed", 0)
    set_seed(seed)

    print_config_summary(config)

    # Step 2: Federated dataset loading.
    client_train_loaders, dataset_info = build_federated_dataloaders(config)

    print_dataset_summary(dataset_info)
    test_client_batch(client_train_loaders)

    # PEFT methods from config.
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

    print("\nAll tests completed.")


def parse_args():
    """
    Description:
        Parse command-line arguments.

    INPUTS:
        None.

    OUTPUTS:
        argparse.Namespace: Parsed arguments.
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
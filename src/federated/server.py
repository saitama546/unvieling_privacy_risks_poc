import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class FLServer:
    """
    Description:
        Federated Learning server class.

        The server stores the global model, sends model copies to clients,
        receives client gradient packages, and stores gradients for later
        aggregation or attack evaluation.

        In the honest-but-curious setting, the server does not modify the model
        maliciously. It only observes gradients that clients normally send.

    INPUTS:
        global_model (nn.Module): Global PEFT model created from config.
        config (Dict[str, Any]): Full experiment configuration dictionary.
        device (Optional[torch.device]): Device where the server model is stored.
            If None, CUDA is used when available; otherwise CPU is used.

    OUTPUTS:
        FLServer: Server object containing global model and received gradients.
    """

    def __init__(
        self,
        global_model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Description:
            Initialize the FL server.

        INPUTS:
            global_model (nn.Module): Global model created from config.
            config (Dict[str, Any]): Full experiment config dictionary.
            device (Optional[torch.device]): Computation device.

        OUTPUTS:
            None.
        """
        self.config = config
        self.fl_config = config["fl"]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.global_model = global_model.to(self.device)
        self.current_round = 0

        # Stores gradients received from clients.
        # Format:
        # {
        #   round_id: {
        #       client_id: gradient_package
        #   }
        # }
        self.received_client_gradients = {}

    def get_model_copy(self) -> nn.Module:
        """
        Description:
            Return a deep copy of the server's global model.

            The client should receive a copy, not the original server model.
            This prevents client-side operations from accidentally changing
            the server model.

        INPUTS:
            None.

        OUTPUTS:
            nn.Module: Deep-copied global model.
        """
        model_copy = copy.deepcopy(self.global_model)
        model_copy = model_copy.to(self.device)

        return model_copy

    def receive_client_gradient(self, gradient_package: Dict[str, Any]) -> None:
        """
        Description:
            Receive and store one client's FedSGD gradient package.

            The gradient package should contain gradients only. It should not
            contain private images or labels.

        INPUTS:
            gradient_package (Dict[str, Any]): Dictionary returned by
                FLClient.compute_fedsgd_gradient().

                Expected keys:
                    client_id (int): Client ID.
                    loss (float): Client loss value for logging/debugging.
                    gradients (Dict[str, torch.Tensor]): Shared gradients.
                    share_type (str): Type of shared gradients.

        OUTPUTS:
            None: Stores the gradient package inside the server.
        """
        if "client_id" not in gradient_package:
            raise ValueError("gradient_package must contain key: client_id")

        if "gradients" not in gradient_package:
            raise ValueError("gradient_package must contain key: gradients")

        if "images" in gradient_package:
            raise ValueError("Private images must not be sent to the server.")

        if "labels" in gradient_package:
            raise ValueError("Private labels must not be sent to the server.")

        client_id = gradient_package["client_id"]
        round_id = self.current_round

        if round_id not in self.received_client_gradients:
            self.received_client_gradients[round_id] = {}

        self.received_client_gradients[round_id][client_id] = gradient_package

    def get_received_gradients_for_round(
        self,
        round_id: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Description:
            Return all client gradient packages received for a specific round.

        INPUTS:
            round_id (Optional[int]): FL round ID.
                If None, uses the current round.

        OUTPUTS:
            Dict[int, Dict[str, Any]]:
                Dictionary mapping client_id to gradient_package.
        """
        if round_id is None:
            round_id = self.current_round

        return self.received_client_gradients.get(round_id, {})

    def get_single_client_gradient(
        self,
        client_id: int,
        round_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Description:
            Return one client's gradient package from a specific round.

            This is useful for gradient inversion attack evaluation, where the
            attacker reconstructs from one client's gradient before aggregation.

        INPUTS:
            client_id (int): Client ID.
            round_id (Optional[int]): FL round ID.
                If None, uses the current round.

        OUTPUTS:
            Dict[str, Any]: Stored gradient package for the selected client.
        """
        if round_id is None:
            round_id = self.current_round

        round_gradients = self.get_received_gradients_for_round(round_id)

        if client_id not in round_gradients:
            raise KeyError(
                f"No gradient found for client_id={client_id} "
                f"in round_id={round_id}."
            )

        return round_gradients[client_id]

    def aggregate_gradients(
        self,
        round_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Description:
            Average gradients received from clients in one FL round.

            This implements basic FedSGD aggregation:

                average_gradient = mean(client_gradients)

            It assumes all selected clients share the same gradient keys.

        INPUTS:
            round_id (Optional[int]): FL round ID.
                If None, uses the current round.

        OUTPUTS:
            Dict[str, torch.Tensor]: Averaged gradients by parameter name.
        """
        if round_id is None:
            round_id = self.current_round

        round_gradients = self.get_received_gradients_for_round(round_id)

        if len(round_gradients) == 0:
            raise RuntimeError(f"No gradients received for round {round_id}.")

        client_packages = list(round_gradients.values())

        first_gradients = client_packages[0]["gradients"]
        gradient_names = list(first_gradients.keys())

        averaged_gradients = {}

        for name in gradient_names:
            grads_for_name = []

            for package in client_packages:
                gradients = package["gradients"]

                if name not in gradients:
                    raise KeyError(
                        f"Gradient name {name} missing from one client package."
                    )

                grads_for_name.append(gradients[name].to(self.device))

            stacked = torch.stack(grads_for_name, dim=0)
            averaged_gradients[name] = stacked.mean(dim=0)

        return averaged_gradients

    def apply_aggregated_gradients(
        self,
        averaged_gradients: Dict[str, torch.Tensor],
        lr: Optional[float] = None,
    ) -> None:
        """
        Description:
            Apply averaged FedSGD gradients to the server global model.

            This updates only parameters whose names appear in averaged_gradients.

                param = param - lr * gradient

        INPUTS:
            averaged_gradients (Dict[str, torch.Tensor]): Averaged gradients.
            lr (Optional[float]): Server learning rate.
                If None, uses config["fl"]["local_lr"].

        OUTPUTS:
            None: Updates self.global_model in place.
        """
        if lr is None:
            lr = self.fl_config.get("local_lr", 0.001)

        named_params = dict(self.global_model.named_parameters())

        with torch.no_grad():
            for name, grad in averaged_gradients.items():
                if name not in named_params:
                    raise KeyError(
                        f"Gradient name {name} not found in global model parameters."
                    )

                param = named_params[name]

                if not param.requires_grad:
                    raise RuntimeError(
                        f"Trying to update frozen parameter: {name}"
                    )

                param -= lr * grad.to(self.device)

    def run_fedsgd_server_update(
        self,
        round_id: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Description:
            Aggregate received gradients and update the global model.

        INPUTS:
            round_id (Optional[int]): FL round ID.
                If None, uses the current round.
            lr (Optional[float]): Server learning rate.

        OUTPUTS:
            Dict[str, torch.Tensor]: Averaged gradients used for update.
        """
        averaged_gradients = self.aggregate_gradients(round_id=round_id)

        self.apply_aggregated_gradients(
            averaged_gradients=averaged_gradients,
            lr=lr,
        )

        return averaged_gradients

    def increment_round(self) -> None:
        """
        Description:
            Increase the FL round counter by one.

        INPUTS:
            None.

        OUTPUTS:
            None.
        """
        self.current_round += 1

    def get_current_round(self) -> int:
        """
        Description:
            Return the current FL round number.

        INPUTS:
            None.

        OUTPUTS:
            int: Current FL round number.
        """
        return self.current_round

    def get_global_model(self) -> nn.Module:
        """
        Description:
            Return the server's original global model.

            For sending a model to clients, use get_model_copy() instead.

        INPUTS:
            None.

        OUTPUTS:
            nn.Module: Server global model.
        """
        return self.global_model

    def print_received_gradient_summary(
        self,
        round_id: Optional[int] = None,
    ) -> None:
        """
        Description:
            Print summary of gradients received in a round.

        INPUTS:
            round_id (Optional[int]): FL round ID.
                If None, uses the current round.

        OUTPUTS:
            None.
        """
        if round_id is None:
            round_id = self.current_round

        round_gradients = self.get_received_gradients_for_round(round_id)

        print(f"\n========== Received Gradients: Round {round_id} ==========")
        print(f"Number of clients received: {len(round_gradients)}")

        for client_id, package in round_gradients.items():
            print(f"\nClient {client_id}")
            print(f"  Loss: {package.get('loss')}")
            print(f"  Share type: {package.get('share_type')}")
            print(f"  Number of gradient tensors: {len(package['gradients'])}")

            for name, grad in package["gradients"].items():
                print(
                    f"    {name}: "
                    f"shape={tuple(grad.shape)}, "
                    f"norm={grad.norm().item():.6f}, "
                    f"device={grad.device}"
                )

        print("====================================================\n")

    def print_server_summary(self) -> None:
        """
        Description:
            Print a sanity-check summary of the FL server state.

        INPUTS:
            None.

        OUTPUTS:
            None.
        """
        print("\n========== FL Server Summary ==========")
        print(f"Device: {self.device}")
        print(f"Current round: {self.current_round}")
        print(f"FL protocol: {self.fl_config.get('protocol')}")
        print(f"Clients per round: {self.fl_config.get('clients_per_round')}")
        print(f"Shared signal: {self.fl_config.get('share')}")
        print(f"Global model class: {self.global_model.__class__.__name__}")
        print("Server dataset: None")
        print("=======================================\n")
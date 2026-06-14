from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.global_model import (
    get_peft_named_parameters,
    get_trainable_named_parameters,
)


class FLClient:
    """
    Description:
        Federated Learning client class that stores a private client DataLoader,
        client metadata, device information, and a local model received from the
        server.

    INPUTS:
        client_id (int): Unique integer ID for the client.
        train_loader (DataLoader): Private training DataLoader for this client.
        config (Dict[str, Any]): Full experiment configuration dictionary.
        device (Optional[torch.device]): Device where client computation runs.
            If None, CUDA is used when available; otherwise CPU is used.

    OUTPUTS:
        FLClient: Client object containing private DataLoader and local model state.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Description:
            Initialize the FL client with client ID, private train DataLoader,
            config, device, and empty local model.

        INPUTS:
            client_id (int): Unique integer ID for the client.
            train_loader (DataLoader): Private training DataLoader for this client.
            config (Dict[str, Any]): Full experiment configuration dictionary.
            device (Optional[torch.device]): Device for client computation.
                If None, automatically selects CUDA if available, else CPU.

        OUTPUTS:
            None: Initializes client attributes.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.config = config
        self.fl_config = config["fl"]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.local_model = None
        self.last_private_batch = None
        
      

    def set_model(self, model: nn.Module) -> None:
        """
        Description:
            Store a local model copy received from the server.

        INPUTS:
            model (nn.Module): Model copy sent by the FL server.

        OUTPUTS:
            None: Stores model in self.local_model and moves it to client device.
        """
        self.local_model = model.to(self.device)

    def has_model(self) -> bool:
        """
        Description:
            Check whether the client currently has a local model.

        INPUTS:
            None.

        OUTPUTS:
            bool: True if self.local_model is not None, otherwise False.
        """
        return self.local_model is not None

    def get_one_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
            Get one private batch from the client's train DataLoader.

            This batch will later be used in Step 6 to compute FedSGD gradients.

        INPUTS:
            None.

        OUTPUTS:
            Tuple[torch.Tensor, torch.Tensor]:
                images (torch.Tensor): Private image batch with shape [B, C, H, W].
                labels (torch.Tensor): Label tensor with shape [B].
        """
        images, labels = next(iter(self.train_loader))

        images = images.to(self.device)
        labels = labels.to(self.device)

        return images, labels

    def get_num_samples(self) -> int:
        """
        Description:
            Return the number of private training samples owned by this client.

        INPUTS:
            None.

        OUTPUTS:
            int: Number of samples in the client's private train dataset.
        """
        return len(self.train_loader.dataset)

    def get_num_batches(self) -> int:
        """
        Description:
            Return the number of batches in the client's private train DataLoader.

        INPUTS:
            None.

        OUTPUTS:
            int: Number of batches in the client's private train DataLoader.
        """
        return len(self.train_loader)
    
    def compute_fedsgd_gradient(self) -> Dict[str, Any]:
        """
        Description:
            Compute FedSGD gradients using one private client batch.

            The client uses its local model copy, takes one batch from its
            private DataLoader, computes classification loss, backpropagates,
            and extracts gradients according to config["fl"]["share"].

            Private images and labels are NOT shared with the server.

        INPUTS:
            None.

        OUTPUTS:
            Dict[str, Any]: Dictionary containing client ID, loss value,
                selected shared gradients, and share type.
        """
        if self.local_model is None:
            raise RuntimeError(
                "Client has no local model. Call set_model() before computing gradients."
            )

        share_type = self.fl_config.get("share", "peft_gradients")

        self.local_model.train()
        self.local_model.zero_grad(set_to_none=True)

        images, labels = self.get_one_batch()
        self.last_private_batch = {
                    "images": images.detach().clone().cpu(),
                    "labels": labels.detach().clone().cpu(),
                }

        logits = self.local_model(images)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        loss.backward()

        if share_type == "peft_gradients":
            named_params = get_peft_named_parameters(self.local_model)

        elif share_type == "trainable_gradients":
            named_params = get_trainable_named_parameters(self.local_model)

        else:
            raise ValueError(
                f"Unsupported fl.share value: {share_type}. "
                f"Supported values: peft_gradients, trainable_gradients"
            )

        gradients = {}

        for name, param in named_params:
            if param.grad is None:
                raise RuntimeError(f"Gradient for parameter {name} is None.")

            gradients[name] = param.grad.detach().clone().cpu()

        return {
            "client_id": self.client_id,
            "loss": float(loss.item()),
            "gradients": gradients,
            "share_type": share_type,
        }

    def get_last_private_batch(self) -> Dict[str, torch.Tensor]:
        """
        Description:
            Return the last private batch used by this client for gradient computation.

            This is for evaluation only. It must not be sent to the server.

        INPUTS:
            None.

        OUTPUTS:
            Dict[str, torch.Tensor]: Dictionary containing reference images and labels.
        """
        if self.last_private_batch is None:
            raise RuntimeError(
                "No private batch stored yet. "
                "Call compute_fedsgd_gradient() first."
            )

        return self.last_private_batch

    def print_client_summary(self) -> None:
        """
        Description:
            Print a sanity-check summary of the FL client state.

        INPUTS:
            None.

        OUTPUTS:
            None: Prints client information to the console.
        """
        print(f"\n========== FL Client {self.client_id} Summary ==========")
        print(f"Device: {self.device}")
        print(f"Private samples: {self.get_num_samples()}")
        print(f"Private batches: {self.get_num_batches()}")
        print(f"Has local model: {self.has_model()}")

        if self.local_model is not None:
            print(f"Local model class: {self.local_model.__class__.__name__}")

        print("=================================================\n")
        
    

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.global_model import (
    get_peft_named_parameters,
    get_trainable_named_parameters,
)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Compute total variation loss for image smoothness.

        This reduces noisy pixel artifacts in reconstructed images.

    INPUTS:
        image (torch.Tensor): Image tensor with shape [B, C, H, W].

    OUTPUTS:
        torch.Tensor: Scalar total variation loss.
    """
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))

    return tv_h + tv_w


def initialize_dummy_image(
    reference_shape: Tuple[int, int, int, int],
    device: torch.device,
    init_type: str = "random_noise",
) -> torch.Tensor:
    """
    Description:
        Initialize dummy image that will be optimized during reconstruction.

    INPUTS:
        reference_shape (Tuple[int, int, int, int]): Shape [B, C, H, W].
        device (torch.device): Device for optimization.
        init_type (str): Initialization type.

    OUTPUTS:
        torch.Tensor: Trainable dummy image tensor.
    """
    if init_type == "random_noise":
        dummy_image = torch.rand(reference_shape, device=device)

    elif init_type == "zeros":
        dummy_image = torch.zeros(reference_shape, device=device)

    else:
        raise ValueError(
            f"Unsupported dummy image init_type: {init_type}. "
            f"Supported values: random_noise, zeros"
        )

    dummy_image.requires_grad_(True)

    return dummy_image


def get_shared_named_parameters(
    model: nn.Module,
    share_type: str,
):
    """
    Description:
        Select which model parameters are used for gradient matching.

        This must match the gradient type shared by the client.

    INPUTS:
        model (nn.Module): Global/local PEFT model.
        share_type (str): Gradient sharing type from config["fl"]["share"].

    OUTPUTS:
        List[Tuple[str, nn.Parameter]]: Selected named parameters.
    """
    if share_type == "peft_gradients":
        return get_peft_named_parameters(model)

    if share_type == "trainable_gradients":
        return get_trainable_named_parameters(model)

    raise ValueError(
        f"Unsupported share_type: {share_type}. "
        f"Supported values: peft_gradients, trainable_gradients"
    )


def compute_dummy_gradients(
    model: nn.Module,
    dummy_image: torch.Tensor,
    labels: torch.Tensor,
    share_type: str,
) -> Dict[str, torch.Tensor]:
    """
    Description:
        Compute gradients produced by the dummy image.

        These dummy gradients will be compared against the observed client
        gradients received by the server.

    INPUTS:
        model (nn.Module): Model used by the client when real gradients were computed.
        dummy_image (torch.Tensor): Current dummy image with shape [B, C, H, W].
        labels (torch.Tensor): Labels used for reconstruction.
        share_type (str): Which gradients to extract.

    OUTPUTS:
        Dict[str, torch.Tensor]: Dummy gradients by parameter name.
    """
    model.zero_grad(set_to_none=True)

    logits = model(dummy_image)

    loss_fn = nn.CrossEntropyLoss()
    dummy_loss = loss_fn(logits, labels)

    named_params = get_shared_named_parameters(
        model=model,
        share_type=share_type,
    )

    params = [param for _, param in named_params]

    grads = torch.autograd.grad(
        dummy_loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )

    dummy_gradients = {}

    for (name, _), grad in zip(named_params, grads):
        dummy_gradients[name] = grad

    return dummy_gradients


def gradient_matching_loss(
    dummy_gradients: Dict[str, torch.Tensor],
    observed_gradients: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Description:
        Compute MSE loss between dummy gradients and observed client gradients.

    INPUTS:
        dummy_gradients (Dict[str, torch.Tensor]): Gradients from dummy image.
        observed_gradients (Dict[str, torch.Tensor]): Real client gradients.
        device (torch.device): Device for computation.

    OUTPUTS:
        torch.Tensor: Scalar gradient matching loss.
    """
    loss = torch.tensor(0.0, device=device)

    for name, observed_grad in observed_gradients.items():
        if name not in dummy_gradients:
            raise KeyError(f"Missing dummy gradient for parameter: {name}")

        dummy_grad = dummy_gradients[name]
        observed_grad = observed_grad.to(device)

        loss = loss + F.mse_loss(dummy_grad, observed_grad)

    return loss


def reconstruct_gradient_only(
    model: nn.Module,
    observed_gradients: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    reference_shape: Tuple[int, int, int, int],
    config: Dict[str, Any],
    share_type: str = "peft_gradients",
) -> Dict[str, Any]:
    """
    Description:
        Step 9: Gradient-only reconstruction attack.

        This reconstructs the private image by optimizing a dummy image so that
        its PEFT gradients match the observed client PEFT gradients.

        No CLIP.
        No text prompt.
        No semantic guidance.

    INPUTS:
        model (nn.Module): Model copy used for reconstruction.
        observed_gradients (Dict[str, torch.Tensor]): Client gradients observed by server.
        labels (torch.Tensor): Labels for the private batch.
            This first version assumes label-known reconstruction.
        reference_shape (Tuple[int, int, int, int]): Shape of image batch [B, C, H, W].
        config (Dict[str, Any]): Full experiment configuration.
        share_type (str): Gradient type, usually "peft_gradients".

    OUTPUTS:
        Dict[str, Any]: Reconstruction result containing:
            reconstructed_image (torch.Tensor): Final reconstructed image.
            loss_history (List[Dict[str, float]]): Loss values over iterations.
            final_gradient_loss (float): Final gradient matching loss.
            final_tv_loss (float): Final TV loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    # Freeze model parameters. We optimize only dummy image.
    for param in model.parameters():
        param.requires_grad_(param.requires_grad)

    labels = labels.to(device)

    attack_cfg = config["attack"]

    steps = attack_cfg.get("steps", 1000)
    lr = attack_cfg.get("lr", 0.1)
    lambda_grad = attack_cfg.get("lambda_grad", 1.0)
    lambda_tv = attack_cfg.get("lambda_tv", 0.001)
    init_type = attack_cfg.get("init", "random_noise")

    dummy_image = initialize_dummy_image(
        reference_shape=reference_shape,
        device=device,
        init_type=init_type,
    )

    optimizer = torch.optim.Adam([dummy_image], lr=lr)

    loss_history: List[Dict[str, float]] = []

    print("\n========== Step 9: Gradient-Only Reconstruction ==========")
    print(f"Steps: {steps}")
    print(f"LR: {lr}")
    print(f"lambda_grad: {lambda_grad}")
    print(f"lambda_tv: {lambda_tv}")
    print(f"share_type: {share_type}")
    print(f"reference_shape: {reference_shape}")
    print("==========================================================\n")

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Keep image in valid range.
        dummy_image.data.clamp_(0.0, 1.0)

        dummy_gradients = compute_dummy_gradients(
            model=model,
            dummy_image=dummy_image,
            labels=labels,
            share_type=share_type,
        )

        grad_loss = gradient_matching_loss(
            dummy_gradients=dummy_gradients,
            observed_gradients=observed_gradients,
            device=device,
        )

        tv_loss = total_variation_loss(dummy_image)

        total_loss = (lambda_grad * grad_loss) + (lambda_tv * tv_loss)

        total_loss.backward()
        optimizer.step()

        dummy_image.data.clamp_(0.0, 1.0)

        if step % 100 == 0 or step == steps - 1:
            log_item = {
                "step": step,
                "total_loss": float(total_loss.item()),
                "gradient_loss": float(grad_loss.item()),
                "tv_loss": float(tv_loss.item()),
            }

            loss_history.append(log_item)

            print(
                f"Step {step:05d} | "
                f"total={log_item['total_loss']:.6f} | "
                f"grad={log_item['gradient_loss']:.6f} | "
                f"tv={log_item['tv_loss']:.6f}"
            )

    reconstructed_image = dummy_image.detach().clone().cpu()

    result = {
        "reconstructed_image": reconstructed_image,
        "loss_history": loss_history,
        "final_gradient_loss": loss_history[-1]["gradient_loss"],
        "final_tv_loss": loss_history[-1]["tv_loss"],
    }

    print("\nStep 9 gradient-only reconstruction finished.\n")

    return result
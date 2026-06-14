from typing import Dict, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import lpips


class ReconstructionMetrics:
    """
    Description:
        Metrics helper class for evaluating reconstructed images against
        private reference images.

        Supports:
            - PSNR
            - SSIM
            - LPIPS
            - gradient loss passthrough

    INPUTS:
        device (Optional[torch.device]): Device used for LPIPS computation.
            If None, automatically selects CUDA if available, else CPU.

    OUTPUTS:
        ReconstructionMetrics: Metrics evaluator object.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Description:
            Initialize the metrics evaluator and LPIPS model.

        INPUTS:
            device (Optional[torch.device]): Device used for LPIPS computation.

        OUTPUTS:
            None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
        self.lpips_model.eval()

    def _validate_image_tensor(self, image: torch.Tensor, name: str) -> None:
        """
        Description:
            Validate image tensor shape and type.

        INPUTS:
            image (torch.Tensor): Image tensor expected in shape [B, C, H, W].
            name (str): Name of the tensor for error messages.

        OUTPUTS:
            None.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if image.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B, C, H, W], but got shape {tuple(image.shape)}."
            )

        if image.shape[0] != 1:
            raise ValueError(
                f"{name} must have batch size 1 for current metrics implementation, "
                f"but got batch size {image.shape[0]}."
            )

        if image.shape[1] not in [1, 3]:
            raise ValueError(
                f"{name} must have 1 or 3 channels, but got {image.shape[1]} channels."
            )

    def _tensor_to_numpy_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Description:
            Convert torch image tensor [1, C, H, W] into numpy image [H, W, C]
            for PSNR and SSIM computation.

        INPUTS:
            image (torch.Tensor): Image tensor with shape [1, C, H, W].

        OUTPUTS:
            np.ndarray: Numpy image with shape [H, W, C] and values in [0, 1].
        """
        image = image.detach().cpu().float().clamp(0.0, 1.0)

        # Remove batch dimension: [1, C, H, W] -> [C, H, W]
        image = image.squeeze(0)

        # [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0).numpy()

        return image

    def compute_psnr(
        self,
        reference_image: torch.Tensor,
        reconstructed_image: torch.Tensor,
    ) -> float:
        """
        Description:
            Compute PSNR between the private reference image and reconstructed image.

        INPUTS:
            reference_image (torch.Tensor): Ground-truth image with shape [1, C, H, W].
            reconstructed_image (torch.Tensor): Reconstructed image with shape [1, C, H, W].

        OUTPUTS:
            float: PSNR value. Higher is better.
        """
        self._validate_image_tensor(reference_image, "reference_image")
        self._validate_image_tensor(reconstructed_image, "reconstructed_image")

        ref_np = self._tensor_to_numpy_image(reference_image)
        recon_np = self._tensor_to_numpy_image(reconstructed_image)

        psnr_value = peak_signal_noise_ratio(ref_np, recon_np, data_range=1.0)
        return float(psnr_value)

    def compute_ssim(
        self,
        reference_image: torch.Tensor,
        reconstructed_image: torch.Tensor,
    ) -> float:
        """
        Description:
            Compute SSIM between the private reference image and reconstructed image.

        INPUTS:
            reference_image (torch.Tensor): Ground-truth image with shape [1, C, H, W].
            reconstructed_image (torch.Tensor): Reconstructed image with shape [1, C, H, W].

        OUTPUTS:
            float: SSIM value. Higher is better.
        """
        self._validate_image_tensor(reference_image, "reference_image")
        self._validate_image_tensor(reconstructed_image, "reconstructed_image")

        ref_np = self._tensor_to_numpy_image(reference_image)
        recon_np = self._tensor_to_numpy_image(reconstructed_image)

        if ref_np.shape[-1] == 1:
            # Grayscale case: remove channel dimension
            ref_np = ref_np.squeeze(-1)
            recon_np = recon_np.squeeze(-1)

            ssim_value = structural_similarity(
                ref_np,
                recon_np,
                data_range=1.0,
            )
        else:
            ssim_value = structural_similarity(
                ref_np,
                recon_np,
                channel_axis=-1,
                data_range=1.0,
            )

        return float(ssim_value)

    def compute_lpips(
        self,
        reference_image: torch.Tensor,
        reconstructed_image: torch.Tensor,
    ) -> float:
        """
        Description:
            Compute LPIPS between the private reference image and reconstructed image.

            LPIPS expects images in [-1, 1].

        INPUTS:
            reference_image (torch.Tensor): Ground-truth image with shape [1, C, H, W].
            reconstructed_image (torch.Tensor): Reconstructed image with shape [1, C, H, W].

        OUTPUTS:
            float: LPIPS value. Lower is better.
        """
        self._validate_image_tensor(reference_image, "reference_image")
        self._validate_image_tensor(reconstructed_image, "reconstructed_image")

        reference_image = reference_image.detach().to(self.device).float().clamp(0.0, 1.0)
        reconstructed_image = reconstructed_image.detach().to(self.device).float().clamp(0.0, 1.0)

        # Convert from [0, 1] -> [-1, 1]
        reference_image = (reference_image * 2.0) - 1.0
        reconstructed_image = (reconstructed_image * 2.0) - 1.0

        with torch.no_grad():
            lpips_value = self.lpips_model(reference_image, reconstructed_image)

        return float(lpips_value.item())

    def compute_all_metrics(
        self,
        reference_image: torch.Tensor,
        reconstructed_image: torch.Tensor,
        gradient_loss: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Description:
            Compute all reconstruction metrics.

        INPUTS:
            reference_image (torch.Tensor): Ground-truth private image with shape [1, C, H, W].
            reconstructed_image (torch.Tensor): Reconstructed image with shape [1, C, H, W].
            gradient_loss (Optional[float]): Final gradient matching loss from the attack.

        OUTPUTS:
            Dict[str, float]: Dictionary containing PSNR, SSIM, LPIPS,
                and optionally gradient_loss.
        """
        metrics = {
            "psnr": self.compute_psnr(reference_image, reconstructed_image),
            "ssim": self.compute_ssim(reference_image, reconstructed_image),
            "lpips": self.compute_lpips(reference_image, reconstructed_image),
        }

        if gradient_loss is not None:
            metrics["gradient_loss"] = float(gradient_loss)

        return metrics


def compute_metrics(
    reference_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
    gradient_loss: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Description:
        Convenience wrapper to compute reconstruction metrics without explicitly
        creating a class object.

    INPUTS:
        reference_image (torch.Tensor): Ground-truth private image with shape [1, C, H, W].
        reconstructed_image (torch.Tensor): Reconstructed image with shape [1, C, H, W].
        gradient_loss (Optional[float]): Final gradient matching loss from the attack.
        device (Optional[torch.device]): Device used for LPIPS computation.

    OUTPUTS:
        Dict[str, float]: Dictionary containing computed metrics.
    """
    evaluator = ReconstructionMetrics(device=device)

    return evaluator.compute_all_metrics(
        reference_image=reference_image,
        reconstructed_image=reconstructed_image,
        gradient_loss=gradient_loss,
    )
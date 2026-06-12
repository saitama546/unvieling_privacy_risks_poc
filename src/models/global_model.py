from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import timm


class BottleneckAdapter(nn.Module):
    """
    Description:
        Adapter PEFT module that projects features down to a smaller
        bottleneck dimension and then back to the original feature dimension.

    INPUTS:
        input_dim (int): Size of the input feature dimension.
        bottleneck_dim (int): Size of the adapter bottleneck dimension.

    OUTPUTS:
        BottleneckAdapter: Adapter module that maps features from
            input_dim -> bottleneck_dim -> input_dim.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int):
        """
        Description:
            Initialize the adapter down projection, activation, and up projection.

        INPUTS:
            input_dim (int): Size of the input feature dimension.
            bottleneck_dim (int): Size of the adapter bottleneck dimension.

        OUTPUTS:
            None: Initializes adapter layers.
        """
        super().__init__()

        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Apply the bottleneck adapter to input features.

        INPUTS:
            x (torch.Tensor): Input feature tensor with last dimension input_dim.

        OUTPUTS:
            torch.Tensor: Adapter output tensor with the same shape as x.
        """
        return self.up(self.activation(self.down(x)))


class LoRAFeatureAdapter(nn.Module):
    """
    Description:
        Feature-level LoRA-style PEFT module that applies a low-rank update
        to extracted backbone features.

    INPUTS:
        input_dim (int): Size of the input feature dimension.
        rank (int): Low-rank projection size.
        alpha (float): LoRA scaling factor.
        dropout (float): Dropout probability applied between LoRA projections.

    OUTPUTS:
        LoRAFeatureAdapter: LoRA-style module that returns a scaled feature update.
    """

    def __init__(self,input_dim: int,rank: int, alpha: float,dropout: float = 0.0):
        """
        Description:
            Initialize LoRA A and B projections, scaling value, and dropout.

        INPUTS:
            input_dim (int): Size of the input feature dimension.
            rank (int): Low-rank projection size.
            alpha (float): LoRA scaling factor.
            dropout (float): Dropout probability applied between LoRA projections.

        OUTPUTS:
            None: Initializes LoRA-style feature adapter layers.
        """
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be greater than 0.")

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Apply the LoRA-style feature update to input features.

        INPUTS:
            x (torch.Tensor): Input feature tensor with last dimension input_dim.

        OUTPUTS:
            torch.Tensor: Scaled low-rank feature update with the same shape as x.
        """
        return self.scale * self.lora_B(self.dropout(self.lora_A(x)))


class GlobalVisionPEFTModel(nn.Module):
    """
    Description:
        Global vision model used by the server with a pretrained backbone,
        PEFT module, and classifier head.

    INPUTS:
        backbone_name (str): Name of the timm backbone model.
        pretrained (bool): Whether to load pretrained backbone weights.
        num_classes (int): Number of output classes.
        freeze_backbone (bool): Whether to freeze backbone parameters.
        peft_method (str): PEFT method to use, either "adapter" or "lora".
        adapter_bottleneck_dim (int): Bottleneck size for adapter PEFT.
        lora_rank (int): Low-rank size for LoRA PEFT.
        lora_alpha (float): Scaling factor for LoRA PEFT.
        lora_dropout (float): Dropout probability for LoRA PEFT.

    OUTPUTS:
        GlobalVisionPEFTModel: Model that maps image batches to class logits.
    """

    def __init__(self,backbone_name: str,pretrained: bool,num_classes: int,freeze_backbone: bool,
                    peft_method: str,adapter_bottleneck_dim: int = 64,lora_rank: int = 8,
                        lora_alpha: float = 16.0,lora_dropout: float = 0.0):
        """
        Description:
            Initialize the backbone, selected PEFT module, and classifier head.

        INPUTS:
            backbone_name (str): Name of the timm backbone model.
            pretrained (bool): Whether to load pretrained backbone weights.
            num_classes (int): Number of output classes.
            freeze_backbone (bool): Whether to freeze backbone parameters.
            peft_method (str): PEFT method to use, either "adapter" or "lora".
            adapter_bottleneck_dim (int): Bottleneck size for adapter PEFT.
            lora_rank (int): Low-rank size for LoRA PEFT.
            lora_alpha (float): Scaling factor for LoRA PEFT.
            lora_dropout (float): Dropout probability for LoRA PEFT.

        OUTPUTS:
            None: Initializes model layers and optionally freezes the backbone.
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.peft_method = peft_method.lower()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if not hasattr(self.backbone, "num_features"):
            raise ValueError(
                f"Backbone {backbone_name} does not expose num_features."
            )

        self.feature_dim = self.backbone.num_features

        if self.peft_method == "adapter":
            self.peft = BottleneckAdapter(
                input_dim=self.feature_dim,
                bottleneck_dim=adapter_bottleneck_dim,
            )

        elif self.peft_method == "lora":
            self.peft = LoRAFeatureAdapter(
                input_dim=self.feature_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        else:
            raise ValueError(
                f"Unsupported PEFT method: {self.peft_method}. "
                f"Supported methods: adapter, lora"
            )

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            self.freeze_backbone_parameters()

    def freeze_backbone_parameters(self) -> None:
        """
        Description:
            Freeze pretrained backbone parameters so PEFT module and classifier
            remain trainable.

        INPUTS:
            None.

        OUTPUTS:
            None: Sets requires_grad=False for all backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Extract features from the backbone and convert output to [B, D].

        INPUTS:
            x (torch.Tensor): Input image batch with shape [B, C, H, W].

        OUTPUTS:
            torch.Tensor: Feature tensor with shape [B, D].
        """
        features = self.backbone(x)

        if features.ndim == 4:
            features = features.mean(dim=(2, 3))

        elif features.ndim == 3:
            features = features.mean(dim=1)

        elif features.ndim != 2:
            raise ValueError(
                f"Unexpected backbone feature shape: {features.shape}"
            )

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Run images through the backbone, PEFT residual update, and classifier.

        INPUTS:
            x (torch.Tensor): Input image batch with shape [B, C, H, W].

        OUTPUTS:
            torch.Tensor: Class logits with shape [B, num_classes].
        """
        features = self.extract_features(x)
        peft_update = self.peft(features)
        adapted_features = features + peft_update
        logits = self.classifier(adapted_features)
        return logits


def _resolve_num_classes(config: Dict[str, Any],dataset_info: Dict[str, Any]) -> int:
    """
    Description:
        Resolve the number of classes from config or dataset metadata.

    INPUTS:
        config (Dict[str, Any]): Experiment config containing model settings.
        dataset_info (Dict[str, Any]): Dataset metadata containing num_classes.

    OUTPUTS:
        int: Number of classes for the classifier head.
    """
    model_cfg = config["model"]
    num_classes = model_cfg.get("num_classes", "auto")

    if str(num_classes).lower() == "auto":
        return int(dataset_info["num_classes"])

    return int(num_classes)


def get_peft_methods(config: Dict[str, Any]) -> List[str]:
    """
    Description:
        Read PEFT methods from config using either the new methods list or
        old single method format.

    INPUTS:
        config (Dict[str, Any]): Experiment config containing PEFT settings.

    OUTPUTS:
        List[str]: Lowercase PEFT method names from config.
    """
    peft_cfg = config["peft"]

    if "methods" in peft_cfg:
        methods = peft_cfg["methods"]
    elif "method" in peft_cfg:
        methods = peft_cfg["method"]
    else:
        raise ValueError("Config must contain peft.method or peft.methods.")

    if isinstance(methods, str):
        methods = [methods]

    if not isinstance(methods, list):
        raise ValueError("peft.method or peft.methods must be a string or list.")

    return [str(method).lower() for method in methods]


def _get_adapter_config(peft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Description:
        Read adapter-specific PEFT settings from config.

    INPUTS:
        peft_cfg (Dict[str, Any]): PEFT config section.

    OUTPUTS:
        Dict[str, Any]: Adapter config containing bottleneck_dim.
    """
    adapter_cfg = peft_cfg.get("adapter", {})

    return {
        "bottleneck_dim": int(adapter_cfg.get("bottleneck_dim", 64)),
    }


def _get_lora_config(peft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Description:
        Read LoRA-specific PEFT settings from config.

    INPUTS:
        peft_cfg (Dict[str, Any]): PEFT config section.

    OUTPUTS:
        Dict[str, Any]: LoRA config containing rank, alpha, and dropout.
    """
    lora_cfg = peft_cfg.get("lora", {})

    return {
        "rank": int(lora_cfg.get("rank", 8)),
        "alpha": float(lora_cfg.get("alpha", 16.0)),
        "dropout": float(lora_cfg.get("dropout", 0.0)),
    }


def build_global_model(config: Dict[str, Any],dataset_info: Dict[str, Any],peft_method: Optional[str] = None) -> nn.Module:
    """
    Description:
        Build one global server model from config and dataset metadata.

    INPUTS:
        config (Dict[str, Any]): Experiment config containing model and PEFT settings.
        dataset_info (Dict[str, Any]): Dataset metadata containing class information.
        peft_method (Optional[str]): Specific PEFT method to build. If None,
            the first configured PEFT method is used.

    OUTPUTS:
        nn.Module: GlobalVisionPEFTModel configured for the selected PEFT method.
    """
    model_cfg = config["model"]
    peft_cfg = config["peft"]

    available_methods = get_peft_methods(config)

    if peft_method is None:
        peft_method = available_methods[0]
    else:
        peft_method = peft_method.lower()

    if peft_method not in available_methods:
        raise ValueError(
            f"Requested peft_method='{peft_method}' but config has {available_methods}"
        )

    if peft_method not in ["adapter", "lora"]:
        raise ValueError(
            f"Unsupported PEFT method: {peft_method}. "
            f"Supported methods: adapter, lora"
        )

    backbone_name = model_cfg["backbone"]
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    num_classes = _resolve_num_classes(config, dataset_info)

    adapter_cfg = _get_adapter_config(peft_cfg)
    lora_cfg = _get_lora_config(peft_cfg)

    model = GlobalVisionPEFTModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        peft_method=peft_method,
        adapter_bottleneck_dim=adapter_cfg["bottleneck_dim"],
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
    )

    return model


def build_all_global_models(
    config: Dict[str, Any],
    dataset_info: Dict[str, Any],
) -> Dict[str, nn.Module]:
    """
    Description:
        Build one global model for each PEFT method listed in config.

    INPUTS:
        config (Dict[str, Any]): Experiment config containing model and PEFT settings.
        dataset_info (Dict[str, Any]): Dataset metadata containing class information.

    OUTPUTS:
        Dict[str, nn.Module]: Dictionary mapping PEFT method names to models.
    """
    models = {}

    for method in get_peft_methods(config):
        models[method] = build_global_model(
            config=config,
            dataset_info=dataset_info,
            peft_method=method,
        )

    return models


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Description:
        Count total, trainable, and frozen parameters in a model.

    INPUTS:
        model (nn.Module): PyTorch model to inspect.

    OUTPUTS:
        Dict[str, int]: Parameter counts with keys total, trainable, and frozen.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }


def get_peft_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Description:
        Return trainable PEFT parameters only.

    INPUTS:
        model (nn.Module): PyTorch model to inspect.

    OUTPUTS:
        List[Tuple[str, nn.Parameter]]: PEFT parameter names and tensors.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith("peft.") and param.requires_grad
    ]


def get_classifier_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Description:
        Return trainable classifier parameters only.

    INPUTS:
        model (nn.Module): PyTorch model to inspect.

    OUTPUTS:
        List[Tuple[str, nn.Parameter]]: Classifier parameter names and tensors.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith("classifier.") and param.requires_grad
    ]


def get_trainable_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Description:
        Return all trainable model parameters.

    INPUTS:
        model (nn.Module): PyTorch model to inspect.

    OUTPUTS:
        List[Tuple[str, nn.Parameter]]: Trainable parameter names and tensors.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]


def print_model_summary(model: nn.Module) -> None:
    """
    Description:
        Print model metadata and parameter summary to the console.

    INPUTS:
        model (nn.Module): PyTorch model to summarize.

    OUTPUTS:
        None: Prints summary information and does not return a value.
    """
    param_counts = count_parameters(model)

    print("\n========== Global Model Summary ==========")
    print(f"Model class: {model.__class__.__name__}")

    if hasattr(model, "backbone_name"):
        print(f"Backbone: {model.backbone_name}")

    if hasattr(model, "peft_method"):
        print(f"PEFT method: {model.peft_method}")

    if hasattr(model, "feature_dim"):
        print(f"Feature dim: {model.feature_dim}")

    if hasattr(model, "num_classes"):
        print(f"Num classes: {model.num_classes}")

    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")

    print("\nPEFT parameter names:")
    for name, param in get_peft_named_parameters(model):
        print(f"  {name}: {tuple(param.shape)}")

    print("\nClassifier parameter names:")
    for name, param in get_classifier_named_parameters(model):
        print(f"  {name}: {tuple(param.shape)}")

    print("\nAll trainable parameter names:")
    for name, param in get_trainable_named_parameters(model):
        print(f"  {name}: {tuple(param.shape)}")

    print("==========================================\n")

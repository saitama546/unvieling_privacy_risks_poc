from typing import Any, Dict, List, Tuple
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class HuggingFaceVisionDataset(Dataset):
    """
    Description:
        Wraps a Hugging Face vision dataset so it works like a PyTorch Dataset.

    INPUTS:
        hf_dataset (datasets.Dataset): The Hugging Face dataset split.
        image_column (str): The column name that stores images.
        label_column (str): The column name that stores labels.
        transform (callable | None): Optional transform to apply to images.

    OUTPUTS:
        HuggingFaceVisionDataset: Dataset object that returns transformed image
            tensors and label tensors.
    """
    
    def __init__(self,hf_dataset,image_column: str,label_column: str,transform=None):
        """
        Description:
            Store dataset columns and optional image transform.

        INPUTS:
            hf_dataset (datasets.Dataset): Hugging Face dataset split.
            image_column (str): Name of the image column.
            label_column (str): Name of the label column.
            transform (callable | None): Optional transform applied to each image.

        OUTPUTS:
            None: Initializes the dataset wrapper attributes.
        """
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        """
        Description:
            Return the number of samples in the dataset.

        INPUTS:
            None.

        OUTPUTS:
            int: Number of samples in hf_dataset.
        """
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        """
        Description:
            Load one sample, convert image to RGB, apply transform, and return label tensor.

        INPUTS:
            index (int): Sample index to load.

        OUTPUTS:
            Tuple[torch.Tensor, torch.Tensor]: Transformed RGB image tensor and
                label tensor with dtype torch.long.
        """
        sample = self.hf_dataset[index]
        image = sample[self.image_column]
        label = sample[self.label_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def _build_transform(image_size: int):
    """
    Description:
        Build the common image transform for all vision datasets.

    INPUTS:
        image_size (int): Target image height and width.

    OUTPUTS:
        transforms.Compose: Transform that resizes images to
            (image_size, image_size) and converts them to torch.Tensor.
    """
    return transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])


def _load_huggingface_train_dataset(config: Dict[str, Any]):
    """
    Description:
        Load the training split from Hugging Face and wrap it as a PyTorch Dataset.

    INPUTS:
        config (Dict[str, Any]): Experiment config with dataset settings such as hf_path,
            train_split, image_column, label_column, and image_size.

    OUTPUTS:
        Tuple[HuggingFaceVisionDataset, List[str], int]: Wrapped training dataset,
            class names, and number of classes.
    """
    dataset_cfg = config["dataset"]
    hf_path = dataset_cfg["hf_path"]
    train_split = dataset_cfg.get("train_split", "train")
    image_column = dataset_cfg.get("image_column", "img")
    label_column = dataset_cfg.get("label_column", "label")
    image_size = dataset_cfg.get("image_size", 224)
    raw_dataset_dict = load_dataset(hf_path)
    raw_train_dataset = raw_dataset_dict[train_split]
    transform = _build_transform(image_size)
    train_dataset = HuggingFaceVisionDataset(hf_dataset=raw_train_dataset,image_column=image_column,
        label_column=label_column,
        transform=transform)
    label_feature = raw_train_dataset.features[label_column]

    if hasattr(label_feature, "names") and label_feature.names is not None:
        class_names = label_feature.names
    else:
        unique_labels = sorted(set(raw_train_dataset[label_column]))
        class_names = [str(label) for label in unique_labels]

    num_classes = len(class_names)

    return train_dataset, class_names, num_classes


def _split_dataset_iid(dataset: Dataset,num_clients: int,seed: int) -> List[Subset]:
    """
    Description:
        Split training dataset IID across clients.

    INPUTS:
        dataset (Dataset): Training dataset to split.
        num_clients (int): Number of client subsets to create.
        seed (int): Random seed for reproducible splitting.

    OUTPUTS:
        List[Subset]: Client dataset subsets, one subset per client.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")

    dataset_size = len(dataset)
    base_size = dataset_size // num_clients

    lengths = [base_size] * num_clients
    remainder = dataset_size - sum(lengths)
    for i in range(remainder):
        lengths[i] += 1
    generator = torch.Generator().manual_seed(seed)
    client_subsets = random_split(dataset,lengths,generator=generator)
    return list(client_subsets)


def _create_client_train_loaders(client_subsets: List[Subset],batch_size: int,num_workers: int,shuffle: bool) -> List[DataLoader]:
    """
    Description:
        Create one training DataLoader per client subset.

    INPUTS:
        client_subsets (List[Subset]): Client dataset subsets.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of DataLoader worker processes.
        shuffle (bool): Whether to shuffle each client subset.

    OUTPUTS:
        List[DataLoader]: Training DataLoader objects, one loader per client.
    """
    client_train_loaders = []

    for subset in client_subsets:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        client_train_loaders.append(loader)

    return client_train_loaders


def build_federated_dataloaders(config: Dict[str, Any]):
    """
    Description:
        Build federated client dataloaders from the dataset config.

    INPUTS:
        config (Dict[str, Any]): Experiment config containing dataset and experiment settings.

    OUTPUTS:
        Tuple[List[DataLoader], Dict[str, Any]]: Client training loaders and
            dataset metadata including class names, split sizes, image size,
            and server_has_data=False.
    """
    dataset_cfg = config["dataset"]
    experiment_cfg = config["experiment"]

    seed = experiment_cfg.get("seed", 0)

    source = dataset_cfg.get("source", "huggingface").lower()
    dataset_name = dataset_cfg["name"].lower()
    image_size = dataset_cfg.get("image_size", 224)
    num_clients = dataset_cfg.get("num_clients", 2)
    batch_size = dataset_cfg.get("batch_size", 1)
    num_workers = dataset_cfg.get("num_workers", 2)
    split_type = dataset_cfg.get("split", "iid").lower()
    shuffle = dataset_cfg.get("shuffle", True)

    if source == "huggingface":
        train_dataset, class_names, num_classes = _load_huggingface_train_dataset(config)
    else:
        raise ValueError(
            f"Unsupported dataset source: {source}. "
            f"Currently supported source: huggingface"
        )

    if split_type == "iid":
        client_subsets = _split_dataset_iid(
            dataset=train_dataset,
            num_clients=num_clients,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported split type: {split_type}. "
            f"Currently supported split type: iid"
        )

    client_train_loaders = _create_client_train_loaders(
        client_subsets=client_subsets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    dataset_info = {
        "name": dataset_name,
        "source": source,
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": image_size,
        "num_clients": num_clients,
        "train_size": len(train_dataset),
        "client_sizes": [len(loader.dataset) for loader in client_train_loaders],
        "server_has_data": False,
        "split": split_type,
    }

    return client_train_loaders, dataset_info

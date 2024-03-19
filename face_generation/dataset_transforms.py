from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


def get_transforms(img_size: int, mean: tuple[float], std: tuple[float]) -> "Compose":
    return Compose(
        [Resize(img_size), CenterCrop(img_size), ToTensor(), Normalize(mean, std)]
    )

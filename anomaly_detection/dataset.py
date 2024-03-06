from pathlib import Path
from typing import Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class DefectsDataset(Dataset):
    def __init__(self, img_dir, transform=None, labels=None):
        self.images = list(Path(img_dir).iterdir())
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor | Tuple[Tensor, int]:
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = self.images[idx].name
        return image if not self.labels else (image, int(self.labels[filename][0]))

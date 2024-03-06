from collections import Counter
from pathlib import Path
from shutil import rmtree
from typing import Tuple
from uuid import uuid4

import numpy as np
from PIL import Image
from tqdm import tqdm


class PixelGenerator:
    """Generate images based on statistics of train data."""

    def __init__(
        self,
        imgs_path: Path,
        img_size: Tuple[int, int] = (64, 64),
        generated_output_path: Path = Path("outputs"),
    ):
        self._img_path = imgs_path
        self._img_size = img_size
        self._generated_output_path = generated_output_path
        if self._generated_output_path.exists():
            rmtree(self._generated_output_path)
        self._generated_output_path.mkdir(parents=True)
        self._statistics = self._calculate_statistics()

    def _get_image(self, filename: Path) -> np.ndarray:
        image = Image.open(filename).convert("RGB")
        image = image.resize(self._img_size)
        return np.array(image.getdata())

    def _calculate_statistics(self):
        statistics = {}
        flat_pixels = self._get_flat_pixels()

        number_of_pixels = self._img_size[0] * self._img_size[1]

        for pixel_idx in tqdm(range(number_of_pixels)):
            statistics[pixel_idx] = self._calculate_pixel_statistics(flat_pixels, pixel_idx)

        return statistics

    def _get_flat_pixels(self):
        flat_pixels = [self._get_image(filename) for filename in self._img_path.iterdir()]
        return np.asarray(flat_pixels)

    def _calculate_pixel_statistics(self, flat_pixels, pixel_idx):
        pixel_statistics = {"R": np.zeros(256), "G": np.zeros(256), "B": np.zeros(256)}
        pixel_from_images = flat_pixels[:, pixel_idx, :]

        for color_channel in range(3):
            color_count = Counter(pixel_from_images[:, color_channel])

            for pixel_value in range(256):
                value = color_count.get(pixel_value, 0)
                pixel_statistics[self._get_color_channel(color_channel)][pixel_value] = (
                    value / sum(color_count.values()) if value != 0 else 0
                )

        return pixel_statistics

    @staticmethod
    def _get_color_channel(index):
        match index:
            case 0:
                return "R"
            case 1:
                return "G"
            case 2:
                return "B"
            case _:
                return ValueError("Index must be in range [0-2]")

    def _create_image_from_pixels(self, pixels):
        image = np.asarray(pixels).reshape((*self._img_size, 3))
        return Image.fromarray(np.uint8(image)).convert("RGB")

    def _save_image(self, image):
        image.save(self._generated_output_path / f"{uuid4()}.jpg")

    def _generate_pixels(self):
        pixels = []
        for pixel, data in self._statistics.items():
            pixel_array = []
            for v in data.values():
                index = np.random.choice(range(len(v)), p=v)
                pixel_array.append(index)
            pixels.append(np.asarray(pixel_array))
        return pixels

    def _generate(self) -> None:
        pixels = self._generate_pixels()
        image = self._create_image_from_pixels(pixels)
        self._save_image(image)

    def __iter__(self):
        return self

    def __next__(self):
        return self._generate()


if __name__ == "__main__":
    gen = PixelGenerator(Path("./avatars"))
    for i in range(10):
        next(gen)

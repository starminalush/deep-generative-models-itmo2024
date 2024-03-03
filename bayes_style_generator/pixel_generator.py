from collections import (
    Counter,
)
from pathlib import (
    Path,
)

import numpy as np
from PIL import (
    Image,
)
from tqdm import (
    tqdm,
)


class PixelGenerator:
    def __init__(self, imgs_path: Path):
        self._img_path = imgs_path
        self._img_size = self._get_img_size()
        self._statistics = self._calculate_statistics()

    def _calculate_statistics(self):
        statistics = {}
        flat_pixels = []
        for filename in self._img_path.iterdir():
            image = Image.open(filename).convert("RGB")
            flat_pixels.append(np.array(image.getdata()))
        number_of_pixels = self._img_size[0] * self._img_size[1]

        flat_pixels = np.asarray(flat_pixels)

        for pixel_idx in tqdm(range(number_of_pixels)):
            statistics[pixel_idx] = Counter()
            pixel_from_images = flat_pixels[:, pixel_idx, :]

            r_count = Counter(pixel_from_images[:, 0])
            g_count = Counter(pixel_from_images[:, 1])
            b_count = Counter(pixel_from_images[:, 2])

            statistics[pixel_idx] = {
                "R": np.zeros(256),
                "G": np.zeros(256),
                "B": np.zeros(256),
            }

            for i in range(256):
                value = r_count.get(i, 0)
                statistics[pixel_idx]["R"][i] = value / sum(r_count.values()) if value != 0 else 0
                value = g_count.get(i, 0)
                statistics[pixel_idx]["G"][i] = value / sum(g_count.values()) if value != 0 else 0
                value = b_count.get(i, 0)
                statistics[pixel_idx]["B"][i] = value / sum(b_count.values()) if value != 0 else 0
        return statistics

    def _get_img_size(self):
        img_size = None
        for filename in self._img_path.iterdir():
            image = Image.open(filename)
            if not img_size:
                img_size = image.size
            if img_size != image.size:
                raise Exception("Image size must be the same!!")
        return img_size

    def __iter__(self):
        return self

    def _generate(self):
        pixels = []
        for pixel, data in self._statistics.items():
            pixel_array = []
            for k, v in data.items():
                print(sum(v))
                index = np.random.choice(range(len(v)), p=v)
                pixel_array.append(index)
            pixels.append(np.asarray(pixel_array))
        image = np.asarray(pixels).reshape((560, 528, 3))
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        image.save("test.jpg")

    def __next__(self):
        return self._generate()


if __name__ == "__main__":
    gen = PixelGenerator(Path("./avatars"))
    next(gen)

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Tuple

import numpy as np
from styles import styles, styles_count


@dataclass
class GeneratedStyle:
    desc: str
    proba: float

    def __str__(self):
        return f"Стиль:\n{self.desc}\nВероятность:\n{self.proba}"


class BayesStyleGenerator:
    """Generate text description of style."""

    def __init__(self, styles_desc: dict[str, list[str]], styles_desc_count: dict[str, list[int]]):
        probabilities = self._calculate_proba(styles_desc, styles_desc_count)
        self._combinations, self._combinations_prod = self._generate_all_combinations(probabilities)

    @staticmethod
    def _calculate_proba(styles_desc: dict[str, list[str]], styles_desc_count: dict[str, list[int]]) -> dict[str, list]:
        probabilities = {}
        for style, counts in styles_desc_count.items():
            total_count = sum(counts)
            style_type_probabilities = [
                (
                    f"{style}: {styles_desc[style][idx]}",
                    (count + 1) / (total_count + len(counts)),
                )
                for idx, count in enumerate(counts)
            ]
            probabilities[style] = style_type_probabilities
        return probabilities

    @staticmethod
    def _generate_all_combinations(proba: dict[str, list]) -> Tuple[list, list]:
        all_combinations = list(product(*proba.values()))
        combinations_proba = [prod([value[1] for value in combination]) for combination in all_combinations]
        combinations = [", ".join([value[0] for value in combination]) for combination in all_combinations]
        return combinations, combinations_proba

    def _generate(self) -> "GeneratedStyle":
        style = np.random.choice(self._combinations, p=self._combinations_prod)
        proba = self._combinations_prod[self._combinations.index(style)]
        return GeneratedStyle(desc=style, proba=proba)

    def __iter__(self):
        return self

    def __next__(self) -> "GeneratedStyle":
        return self._generate()


if __name__ == "__main__":
    gen = BayesStyleGenerator(styles_desc=styles, styles_desc_count=styles_count)
    for i in range(10):
        print(next(gen))
        print()

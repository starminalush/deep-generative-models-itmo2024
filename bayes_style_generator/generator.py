from dataclasses import (
    dataclass,
)
from itertools import (
    product,
)
from math import (
    prod,
)

import numpy as np
from styles import (
    styles,
    styles_count,
)


@dataclass
class Generation:
    desc: str
    proba: float

    def __str__(self):
        return f"Стиль:\n{self.desc}\nВероятность:\n{self.proba}"


class BayesStyleGenerator:
    def __init__(self, styles, styles_count):
        parameters = self._calculate_proba(styles, styles_count)
        self._combinations, self._combinations_prod = self._generate_all_combinations(parameters)

    def _calculate_proba(self, styles, styles_count):
        parameters = {}
        for style, counts in styles_count.items():
            total_count = sum(counts)
            probabilities = [
                (
                    f"{style}: {styles[style][idx]}",
                    (count + 1) / (total_count + len(counts)),
                )
                for idx, count in enumerate(counts)
            ]
            parameters[style] = probabilities
        return parameters

    def _generate_all_combinations(self, proba):
        all_combinations = list(product(*proba.values()))
        combinations_proba = [prod([value[1] for value in combination]) for combination in all_combinations]
        combinations = [", ".join([value[0] for value in combination]) for combination in all_combinations]
        return combinations, combinations_proba

    def _generate(self):
        style = np.random.choice(self._combinations, p=self._combinations_prod)
        proba = self._combinations_prod[self._combinations.index(style)]
        return Generation(desc=style, proba=proba)

    def __iter__(self):
        return self

    def __next__(self):
        return self._generate()


if __name__ == "__main__":
    gen = BayesStyleGenerator(styles=styles, styles_count=styles_count)
    for i in range(10):
        print(next(gen))
        print()

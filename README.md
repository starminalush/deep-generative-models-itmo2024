Задача: генерация лиц

Датасет: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


----

## Эксперименты:

## 0. Получить бейзлайн

### Описание
Модель:

Код модели генератора и CSPUp блока представлен в файле [csp_generator.py](https://github.com/starminalush/deep-generative-models-itmo2024/blob/d260ce3ff9bd207461b5e299e8463174f48de7df/face_generation/models/csp_generator.py).

Код модели дискриминатора был взят из стандартной реализации DCGAN и переработан под размер 128x128 пикселей.

### Параметры обучения
- batch_size: 64
- num_epochs: 10 (планировалось, но обучение сразу разошлось)
- img_size: 128
- optimizer: Adam c lr = 0.001 для генератора и дискриминатора

### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/92323b86-9c04-424f-80b3-c8acdf769c5f)

Результат генерации

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/d9b02712-62b6-4381-9c1f-111d47055feb)

### Вывод:
В такой форме генератор и дискриминатор не сойдутся, нужно добавить больше стабильности

## 1. Гипотеза: замена ReLU на Tanh в последнем слое генератора и добавление BatchNorm улучшит сходимость моделей и позволит добиться лучшего качества генерации лиц.

Модель:

Код модели генератора и CSPUp блока представлен в файле [csp_generator.py](https://github.com/starminalush/deep-generative-models-itmo2024/blob/83b3eb8fd329ae049096af28f6dab1675b0d1bb5/face_generation/models/csp_generator.py).

### Параметры обучения
 - batch_size: 64
 - num_epochs: 10 
 - img_size: 128
 - optimizer: Adam c lr = 0.001 для генератора и дискриминатора

### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/148d8838-179e-4d7a-99ea-1facafdb160e)

На середине обучения лосс генератора резко скакнул вверх, но потом снова упал. Качество генераций при этом сильно ухудшилось на этом участке. С чем это связано, я пока не поняла


Результат генерации в начале обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/78381264-1f00-4b5d-87f9-f8f8a3bc6f4d)

Результат генерации, когда скакнул лосс генератора в середине обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/695a47f3-553d-45c9-9f20-ed3777d216d3)

Результат генерации в конце обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/0b79cadc-32d8-47cb-bf29-7ac0f0568a93)


### Вывод:
Произошел mode collapse, много одинаковых лиц. Но модели стали лучше сходиться. Возможно, стоит замедлить работу дискриминатора или добавить label smoothing


## 2. Гипотеза: расширение эксперимента 1 путем добавления label smoothing улучшит сходимость моделей и позволит добиться лучшего качества генерации лиц.
Модель:

см. Эксперимент 1


### Параметры обучения
 - batch_size: 64
 - num_epochs: 10 
 - img_size: 128
 - optimizer: Adam c lr = 0.001 для генератора и дискриминатора

Дополнительно добавила [label smoothing](https://github.com/starminalush/deep-generative-models-itmo2024/blob/bbc2e130fd62f8d054a44de04937ced90e2fd250/face_generation/train.py#L60-L61) в код обучения модели.

### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/34ed5650-2116-449c-8986-49939be38b43)

Результат генерации в начале обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/6acacd34-57ae-403e-b04e-5c92e5e7b3ea)

Результат генерации в конце обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/0adfe0b6-3b08-4225-9b22-224b7c501e64)

### Вывод:
label smoothing не поможет в улучшении качества генераций эксперимента 1. Возможно, стоило вместе с ним дополнительно еще отрегулировать параметры optiizer'ов

## 3. Гипотеза: расширение эксперимента 1 путем уменьшение lr в optimizer у дискриминатора улучшит сходимость моделей и позволит добиться лучшего качества генерации лиц.
Модель:

см. Эксперимент 1

### Параметры обучения
 - batch_size: 64
 - num_epochs: 10 
 - img_size: 128
 - optimizer: Adam c lr = 0.001 для генератора и 0.00001 для дискриминатора

### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/f9f8baef-a320-433f-9855-799ab87c2b90)

Там, где есть пик у генератора, не произошло ничего необычного, качество генераций не проседало, как был шум, так и есть.
Результат генерации в начале обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/e4419451-a6e3-498c-bb71-2b99a4339226)


Результат генерации в конце обучения

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/bae01545-7a0f-463c-9b69-71d83cf9c7cb)


## Вывод:
Все еще получился mode collapse.

----

## Следующие шаги:
(я в этот раз не успела, несмотря на три недели на это дз, но хочу попробовать)
 - проверить влияние размера батча на обучение
 - провести эксперимент с заменой optimizer у дискриминатора (SGD попробовать, как вариант)
 - попробовать применить gradient penalty из WGAN для стабилизации обучения
 - попробовать остальные фишки [отсюда](https://github.com/soumith/ganhacks)




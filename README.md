## Задача: генерация лиц

## Эксперименты:
## 0. Получить бейзлайн

### Описание
Датасет: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Модель:

Код модели генератора и CSPUp блока представлен в файле [csp_generator.py](https://github.com/starminalush/deep-generative-models-itmo2024/blob/d260ce3ff9bd207461b5e299e8463174f48de7df/face_generation/models/csp_generator.py).

Код модели дискриминатора был взят из стандартной реализации DCGAN и переработан под размер 128x128 пикселей.

### Параметры обучения
batch_size: 64
num_epochs: 10 (планировалось, но обучение сразу разошлось)
img_size: 128
optimizer: Adam c lr = 0.001 для генератора и дискриминатора



### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/92323b86-9c04-424f-80b3-c8acdf769c5f)

Результат генерации

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/d9b02712-62b6-4381-9c1f-111d47055feb)

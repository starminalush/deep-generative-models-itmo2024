## Задача: генерация лиц

## Эксперименты:
## 0. Получить бейзлайн

### Параметры
Датасет: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Модель:

Код модели генератора и CSPUp блока представлен в файле [csp_generator.py](https://github.com/starminalush/deep-generative-models-itmo2024/blob/d260ce3ff9bd207461b5e299e8463174f48de7df/face_generation/models/csp_generator.py).

Код модели дискриминатора был взят из стандартной реализации DCGAN и переработан под размер 128x128 пикселей.


### Результат обучения
График лоссов генератора и дискриминатора
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/a3b53f07-c555-4427-979f-7ce0162830e3)

Результат генерации
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/b5a73110-b8f7-4d43-ae02-bbafa6006f58)

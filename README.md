**Задача: стилизация лиц**

------
## Эксперименты
## 0. Отразить изображения в латентном пространстве
### Описание:
Отразила изображения в латентом пространстве, используя энкодер и оптимизацию градиента. Использовала StepLR в качестве scheduler. Оптимизировала 200 эпох.

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/d34b47b4-f6ad-4b89-94d9-ff4b87855e9e)

## 1. Перенос стиля
### Описание:
Реализовала перенос стиля с использованием энкодера через интерполяцию. Чтобы перенести стиль, не трогая лицо на изображении, на которое стиль переносят, заменяла в векторе таргета индексы 6-17 значениями вектора изображения со стилем.
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/3bdc3811-46da-4eef-98f4-f9cb3895056e)

**Вывод**: стиль переносится, все хорошо

## 2. Перенос эмоций
### Описание:
Все то же самое, что экс.1, только заменяла индексы 3-7. Для переноса эмоций специально искала фотографии с сильными эмоциями

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/bfe91d10-8024-425e-baa4-04ec215ea7de)

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/082623e5-07cc-4fd0-8f32-da4cca12a44b)

**Вывод**: кажется, нужно учитывать пол людей. При переносе эмоций с мужского лица на женское наблюдается, что лицо у женщины становится грубее. Обратная ситуация с переносом эмоций с женского лица на мужское

## 3. Face Swap
### Описание:
Добавила  arcface loss в пайплайн оптимизации. Обучала 100 эпох без scheduler с lr 0.01. 
Веса лоссов: 
rec_weight = .5
lpips_weight = 1
arcface_weight = 1

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/f88639d8-411c-437f-8aa9-c6019a1dd7af)
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/30ddc778-5693-4257-8668-a45a9c54fc99)
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/a0940dab-3204-40af-8c18-4af8ea9a5c64)
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/5184271b-c42e-48c2-979a-f14f2cb9cf80)
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/82e1ef6f-808c-4602-a41f-b49f28b6a26b)
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/e79c7f21-9987-4d66-b726-e2797286f180)

**Вывод:**: на генерацию лиц влияет вес arcface_loss, lpips_loss и внезапно, количество эпох и lr. Если ставить больше эпох и добавить scheduler, то лица становятся сильно приближены к среднему двух лиц, и как таковой пересадки не происходит.Видимо, чем лучше видно лицо, на которое переносят,тем лучше лцио переносится, так что желательно ему быть в анфас. Возможно, текущий результат можно улучшить, подкрутив еще параметры lpips_weight и arcface_weight.





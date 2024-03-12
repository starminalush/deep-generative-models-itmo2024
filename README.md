## Байесовский генератор стилей
### Краткое описание: реализовать генератор стилей и генератор аватаров на основе MLE и формулы Байеса

**1. Генерация стиля**
   
Скрипт для запуска:
```
python bayes_style_generator/text_style_generator.py
```
**Примеры вывода:**
```
Стиль:
прическа: короткая прямые, цвет волос: серебристо серый, аксесуар: нет очков, одежда: футболка с круглым вырезом, цвет одежды: розовый
Вероятность:
0.001222966329291022

Стиль:
прическа: длинные прямые, цвет волос: рыжий, аксесуар: нет очков, одежда: футболка с круглым вырезом, цвет одежды: оранжевый
Вероятность:
0.0006848611444029722

Стиль:
прическа: длинные прямые, цвет волос: рыжий, аксесуар: круглые очки, одежда: футболка с V-вырезом, цвет одежды: серый
Вероятность:
0.0004594276843703272
```

**2. Генерация аватара**
   
Скрипт для запуска:
```
python bayes_style_generator/pixel_generator.py
```
Внимание: для подсчета статистик и генерации размер изображения равен 64х64 пикселя для лучшей визуализации

**Примеры вывода:**

![6db2b945-bcab-4d5f-a4dc-205246a406be](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/e530a8b1-afca-4525-9248-f97d07756736)

![792c4f65-b995-4803-b3bd-80dea6fb8a7f](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/5d9f8f80-5715-4d75-a266-d36f59bad4d0)

![ada8899b-57e7-4ada-b989-6be7c0fc6d5d](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/da73729b-8496-469e-9cec-8fc4f48a2bab)

![cd4e2099-c6e3-4e36-a69a-73ed42176d08](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/498ea0df-2d9d-46c5-940e-69ecd20cde34)

![f8a641a8-27bf-46c2-83ba-53b78660165f](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/834a74fe-8af8-4020-95a9-fc6d89fd2c1e)



## Детекция аномалий в лунках
### Краткое описание: обучить модель [MNAD](https://github.com/cvlab-yonsei/MNAD) для детекции аномалий

## Запуск
1. Скачайте датасет и поместите его в папку anomaly_detection/data/external/
2. Запустите пайплайн командой ```dvc repro``` из папки anomaly_detection

### Гиперпараметры
- optimizer: Adam
- lr: 2e-4
- batch_size: 16
- img_size: 32
- num_epochs: 15

## Лоссы
### Train
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/01344ae5-d250-4e26-9a83-ba25350e2ff0)

### Validation
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/2d003b51-fa60-4701-a6e7-185dc2f6bc41)


## Метрики
| TPR | TNR|
|---|---|
|0.98|0.76|


## Пример реконструкций изображений
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/26e70ebf-c38d-43aa-a7a2-0b565b8f2506)

## Пример визуализации латентного пространства на 0 эпохе
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/84b1de6c-428a-40f2-8bc8-162b119f1d3d)

## Пример визуализации латентного пространства на последней эпохе
![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/b4b03d1c-16a5-4ebc-8ad2-4c78e784b4da)

## Вывод:
Попробовала новую архитектуру для решения задачи обнаружения проливов. Качество получилось немного хуже только потому, что я убрала некоторые аугментации. Можно добить TNR выше, но нужно просто добавить еще аугментаций (и покрутить параметры у модели MNAD)

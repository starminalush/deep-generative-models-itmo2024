Задача: генерация и перенос лиц

----

## Эксперименты

## 1. Обучить Stable Diffusion 1.5 на своем датасете

### Описание
Будем генерировать Фредди Меркьюри в разных окружениях. 

Для обучения используется библиотека diffusers и предобученные веса с civitai.com

### Параметры обучения
```
!python3 diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks face" \
  --class_prompt="a photo of a man" \
  --seed=13498 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=152 \
  --gradient_accumulation_steps=1 \
  --num_class_images=228 \
  --max_train_steps=1520 \
  --checkpointing_steps=1520\
  --use_8bit_adam \
  --mixed_precision="no"\
  --train_text_encoder
```
Для того, чтобы не генерился только крупный план, используется class prompt "a photo of a man" (в ноутбуке class_data_dir подгружается извне, но на самом деле все картинки в количестве 228 штук были сгенерированы по этому промпту)

###  Примеры генераций
Результат генерации для промпта: "сlose up portrait of sks face, on the street, lights, midnight, Moscow, 4K, raw, hrd, hd, high quality, realism, sharp focus"

<img width="831" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/d8f97c5a-9c9a-4db9-983b-a1accececdc2">

### Вывод
Для генерации персонажа очень важны параметры learning_rate и class_prompt. Чем ниже lr, тем чаще генерились двухголовые тела. Без class_prompt генерировался какой-то мужчина, отдаленно похожий на Меркьюри, в лучшем случае. В худшем - рандомные лица. Количество шагов чем больше, тем лучше. 

## 2. Обучить Lora адаптер

### Описание
Провести несколько экспериментов по выбору параметра **rank** при обучении Lora адаптера

Для теста возьмем параметр rank равный 4, 16, 24

### Параметры обучения
Все эксперименты проводились со следющими параметрами

```
!python3 diffusers/examples/dreambooth/train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks face" \
  --seed=13498 \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000\
  --validation_prompt="A photo of sks face on the music festival" \
  --validation_epochs=500 \
  --use_8bit_adam \
  --mixed_precision="no"\
  --rank=<4, 16, 24>
```

### Примеры генераций
Результат генерации для промпта: "close up portrait of sks face, in the bookshop, books, soft light, 4K, raw, hrd, hd, high quality, realism, sharp focus", rank 4

<img width="829" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/8594fe17-19ab-416c-a52e-a10ec59417ad">

Результат генерации для промпта: "close up portrait of sks face, in the bookshop, books, soft light, 4K, raw, hrd, hd, high quality, realism, sharp focus", rank 16

<img width="829" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/792e8905-88f3-4bb7-9db8-5db66452f5dd">

Результат генерации для промпта: "close up portrait of sks face, in the bookshop, books, soft light, 4K, raw, hrd, hd, high quality, realism, sharp focus", rank 24

<img width="829" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/3abbc111-08ca-4771-93cd-79841c5ad97d">

### Вывод
Lora учится быстрее, чем полный SD1.5. Нужно меньше итераций обучения и class_prompt не обязвтелен, и так хорошо. На качество генераций влияет параметр rank, но с какого-то момента повышать его нет смысла(rank 
16 и 24 в целом одинаково хороши) 


### 3. Сравнить лучшие checkpoint Unet и Lora адаптера
В ноутбуках уже зафиксированы промпты для генерации картинок в 5 разных окружениях и seed.

```
SEED=1337
promt_list = [
    {
     "name": "bookshop",
     "prompt":f"close up portrait of {token} face, in the bookshop, books, soft light, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
    {
     "name": "russian forest",
     "prompt":f"сlose up portrait of {token} face, in the forest, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, weapon, gun, war, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
    {
     "name": "street",
     "prompt":f"сlose up portrait of {token} face, on the street, lights, midnight, Moscow, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands, mutation, ugly, blurry",
    },
    {
     "name": "spaceship",
     "prompt":f"сlose up portrait of {token} face, in the spaceship, stars, planets, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands, mutation, ugly, blurry",
    },
     {
     "name": "music festival",
     "prompt":f"сlose up portrait of {token} face, on the music festival, scene, music, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands, mutation, ugly, blurry",
    },
]
```

Для сравнения возьмем Unet (он один и самый лучший), и Lora адаптер с rank=24


| Окружение | Пример генерации SD1.5 | Пример генерации Lora |
| -------- | ------- | ------ |
| Библиотека | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/4571fad7-841a-42be-b4e2-6b8cf6afab84"> | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/509210a9-b695-4adf-b473-40e3bffb3d86"> |
| Космос | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/04d347ab-1ef4-48bd-9cc1-3ace0fade6c3"> | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/3f76a071-6469-46ae-bd1b-b9b67f7db9ee">
| Музыкальный фестиваль | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/987fddc1-597d-444a-8cd7-743b51e8e55a"> | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/08c8cd1d-9b9f-4b05-a7a4-af98e811b1a7">|
| Лес | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/99767365-33be-40c7-9a80-f76b8498b8ae"> | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/dc33d0d5-6d4d-48c4-b91e-a73a3f1a207e"> |
| Город | <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/5d31b52e-64f2-4a80-ad2a-fdabecc2c3a4">| <img width="500" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/b407c220-4044-4d65-814d-a8c78f64608a">|


### Вывод
Несмотря на то, что lora адаптер иногда генерирует два лица, считаю, что ее генерация лучше, потому что она не генерирует летающих голов и генерации более похожи на Меркьюри. 


## 4. ControlNet

### Описание

Добавить в пайплайн ControlNet для Unet и Lora адаптер

Будем использовать sd-controlnet-canny. Для референса возьмем портрет девушки в платке Венецианова

![image](https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/9d4ec95c-c0e5-472e-a47d-c91ba115824a)

Для генерации для Unet и Lora используем промпт

```
prompt = "a photo of sks face, best quality, extremely detailed, 4k, hdr, super resolution"
```

###  Примеры генераций

**Unet**

<img width="335" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/373b178b-1253-4067-a046-4a55e340bae7">

**Lora адаптер rank 24**

<img width="335" alt="image" src="https://github.com/starminalush/deep-generative-models-itmo2024/assets/103132748/4d147423-670f-4e51-893e-d886678a6c22">

### Вывод
По результатам генерации получили +- одинаковый результат, разве что для генерации Unet выражение лица больше похоже на референс






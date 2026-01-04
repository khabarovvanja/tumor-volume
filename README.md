# Estimation of tumor volume by PET-CT in children with Hodgkin's lymphoma

## Setup
Before installing, make sure you have `Poetry` installed and `Mlflow` running at `127.0.0.1:8080`. Dependency installation:

    git clone https://github.com/khabarovvanja/tumor-volume.git
    cd tumor-volume
    make setup

## Train
Start training with default hyperparameters:

    poetry run python -m tumor_volume.commands train

You can start training with custom settings also:

    poetry run python -m tumor_volume.commands train training.epochs=100 training.train_split=0.7

## Production preparation
Опишите шаги подготовки натренированной модели к работе, что для этого нужно сделать. Сюда могут входить перевод в onnx, tensorrt, etc.
Также в этом разделе можно описать комплектацию поставки вашей модели (какие артефакты, модули нужны для запуска).

## Infer
Смысл тот же, что и у Train, но тут должно быть описано, как после тренировки запустить модель на новых данных. Также нужно описать формат таких данных, дать пример (можно в виде артефакта в вашем data storage).
Код предсказания должен зависеть от минимального количества зависимостей, поэтому его скорее всего не стоит реализовывать в одном файле с Train процедурой.

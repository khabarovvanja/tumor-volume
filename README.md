# Tumor Volume Estimation from PET-CT in Pediatric Hodgkin Lymphoma

This project focuses on automatic tumor segmentation and tumor volume estimation from PET-CT scans in pediatric patients with Hodgkin’s lymphoma.  
The pipeline includes data preprocessing, model training, experiment tracking, and inference with volumetric measurements.

---

## Setup

### Prerequisites

Before starting, ensure the following tool are installed on your system: MLflow (running locally at `http://127.0.0.1:8080`)

### Environment setup

Clone the repository and install all dependencies:

    git clone https://github.com/khabarovvanja/tumor-volume.git
    cd tumor-volume
    make setup
    export PATH="$HOME/.local/bin:$PATH"

## Train

Start training with default hyperparameters:

    poetry run python -m tumor_volume.commands train

You can start training with custom settings also:

    poetry run python -m tumor_volume.commands train training.epochs=100 training.train_split=0.7

## Inference

    poetry run python -m tumor_volume.commands infer


## Production preparation
Опишите шаги подготовки натренированной модели к работе, что для этого нужно сделать. Сюда могут входить перевод в onnx, tensorrt, etc.
Также в этом разделе можно описать комплектацию поставки вашей модели (какие артефакты, модули нужны для запуска).


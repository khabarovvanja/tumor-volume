# Tumor Volume Estimation from PET-CT in Pediatric Hodgkin Lymphoma

**Ivan Khabarov**

This project focuses on automatic tumor segmentation and tumor volume estimation
from PET-CT scans in pediatric patients with Hodgkin’s lymphoma. The pipeline
includes data preprocessing, model training, experiment tracking, and inference
with volumetric measurements.

<Details>

### Problem Statement

Currently, there is no standard method for calculating tumor burden in patients
with Hodgkin lymphoma. There is only a visual assessment of tumor metabolic
activity and disease stage based on the classic Ann Arbor classification (an
outdated method of staging lymphoma). Manual calculation of tumor volume is a
very long and laborious process, which is why it has never been taken into
account in staging or treatment protocols. An automated tumor volume assessment
system will allow the maximum tumor burden to be determined in the shortest
possible time. Tumor volume will become a more accurate factor for determining
risk groups (this method can be added to the new protocol for the treatment of
Hodgkin's lymphoma in children and adolescents).

### Input and Output Data Format

The input data will be files in DICOM format. The output data will be numpy
array files with tumor segments marked on them (mask) and the volume of the
segmented tumor (ml).

### Metrics

Since the task of determining tumor volume directly depends on the quality of
segmentation, the following metrics are appropriate for this task:

- 3D Dice similarity coefficient - for evaluating segmentation quality
- Cross Entropy - for classification.
- HD95 (95th percentile Hausdorff Distance) - one of the most important metrics
  for the quality of medical image segmentation where boundary quality is
  important.

### Validation

70% of the data will be used for training, 20% for validation, and 10% for
testing. To ensure that the experiments are reproducible, I plan to use git +
dvc + mlflow.

### Data

**My own data (in progress):**

I made an agreement with the N.N. Petrov National Medical Research Center for
Oncology in St. Petersburg to provide me with about 300 copies of PET/CT images.

**Open data:** https://autopet.grand-challenge.org/Dataset/ - The dataset
consists of patients with histologically confirmed malignant melanoma, lymphoma,
or lung cancer. There are 1,169 data samples in total.

https://www.cancerimagingarchive.net/collection/mediastinal-lymph-node-seg/?utm_source=chatgpt.com -
The dataset concerns patients with lymphadenopathy (i.e., enlarged lymph nodes)
due to a disease or illness such as cancer or infection. There are 1,026 data
instances in total.

### Modeling

**Baseline** Vanilla 3D U-Net is a standard 3D U-Net without transformers or
complex blocks. It is simple and quick to implement. It provides a starting
point for comparison.

**Main model (in progress)** 3D nnU-Net - automatically selects preprocessing,
architecture (patch size, spacing), augmentations, and hyperparameters. Often
achieves state-of-the-art results without manual fine-tuning.

### Deployment

At this point, I cannot say for sure what the final solution will look like, but
ideally, I would like to have not only data on the tumor volume, but also DICOM
images with the affected segments marked so that I can view them using 3D
Slicer.

</Details>

## Setup

### Prerequisites

Before starting, ensure the following tool are installed on your system: MLflow
(running locally at `http://127.0.0.1:8080`)

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

After training process you can run predict of the new data (converted to .npy
format):

    poetry run python -m tumor_volume.commands infer inference.input.volume=path/to/file.npy

...predicted mask and tumor volume will save to the directory
`outputs/inference/<filename>`.

# ImageCLEF2023-MEDVQA-GI-VisionQAries

Repository containing the solution to the Medical Visual Question Answering for GI Task (MEDVQA-GI) in the ImageCLEF 2023 Challenge.

## Overview

This repository contains the code and implementation details for the publication titled "**[Language-based colonoscopy image analysis with pretrained neural networks](https://ceur-ws.org/Vol-3497/paper-120.pdf)**" which was created for ImageCLEF 2023 Lab: [Medical Visual Question Answering for GI Task - MEDVQA-GI](https://www.imageclef.org/2023/medical/vqa).

### Publication Information

Title: Language-based colonoscopy image analysis with pretrained neural networks

Authors: Patrycja Cieplicka, Julia Kłos, Maciej Morawski, Jarosław Opała

## Getting Started

### Data Preparation

Please follow the instructions in [data/README.md](data/README.md) to download the required data.

### Environment Setup (Conda)

Create and activate a new Conda environment:

```bash
conda env create -f environment.yml
conda activate image-clef
```

### Running the Code

#### Task 1 - VQA

Run the following command to execute the VQA pipeline:

```bash
python3 src/pipeline_vqga.py DATA_PATH MODELS_PATH TRAIN_FLAG TEMPLATE_PATH
```

Example:

```bash
python3 src/pipeline_vqga.py \
    "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset" \
    "models/" \
    "true" \
    "src/template/vqa_05_dense_1024.yaml"
```

#### Task 3 - VLQA

Comming soon...

## MMFusion: Multi-modality Diffusion Model for Lymph Node Metastasis Diagnosis in Esophageal Cancer

## A Quick Overview 

<img width="600" height="350" src="https://github.com/wuchengyu123/MMFusion/blob/main/framework.jpg">


## Setup
### Requirements
* Linux (tested on Ubuntu 16.04, 18.04, 20.04)
* Python 3.6+
* PyTorch 1.6 or higher (tested on PyTorch 1.13.1)
* CUDA 11.3 or higher (tested on CUDA 11.6+torch-geometric 2.2.0)

### Installation
  
``conda env create -f environment.yml``

## Training and Evaluation

The training and evaluation code can be overviewed in  ``main.py``. The code of proposed model can be seen in  ``/model``.

## Dataset

Due to ethical review and privacy concerns related to the patients from whom the dataset was collected, the dataset used in the paper can not be made publicly available at this time. Currently, you can use your own multimodal dataset to run the code. The data types and requirements can be set according to ``/dataloader/Dataset.py``.

🧀We will conduct further research based on this dataset. Additionally, we will consider making part of the dataset available in the future.

## Acknowlegment

Our repo is developed based on the these projects: [CARD](https://github.com/XzwHan/CARD), [DiffMIC](https://github.com/scott-yjyang/DiffMIC)

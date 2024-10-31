<div align="center">
  <h2><a href="https://arxiv.org/abs/2405.09539">MMFusion: Multi-modality Diffusion Model for Lymph Node Metastasis Diagnosis in Esophageal Cancer</a></h2>

  Chengyu Wu<sup>\*</sup>, Chengkai Wang<sup>\*</sup>, Yaqi Wang<sup>†</sup>, Huiyu Zhou, Yatao Zhang, Qifeng Wang<sup>†</sup>, Shuai Wang<sup>†</sup>

  <p>
    <a href="https://arxiv.org/abs/2405.09539" alt="arXiv">
      <img src="https://img.shields.io/badge/arXiv-2405.09539-b31b1b.svg?style=flat" />
    </a>
  </p>
</div>
Esophageal cancer is one of the most common types of cancer  worldwide and ranks sixth in cancer-related mortality. Accurate computer assisted diagnosis of cancer progression can help physicians effectively  customize personalized treatment plans. Currently, CT-based cancer diagnosis methods have received much attention for their comprehensive ability to examine patients’ conditions. However, multi-modal based meth
ods may likely introduce information redundancy, leading to underper formance. In addition, efficient and effective interactions between multimodal representations need to be further explored, lacking insightful ex
ploration of prognostic correlation in multi-modality features. In this work, we introduce a multi-modal heterogeneous graph-based conditional feature-guided diffusion model for lymph node metastasis diagnosis based on CT images as well as clinical measurements and radiomics data. To explore the intricate relationships between multi-modal features, we construct a heterogeneous graph. Following this, a conditional feature-guided diffusion approach is applied to eliminate information redundancy. Moreover, we propose a masked relational representation learning strategy, aiming to uncover the latent prognostic correlations and priorities of primary tumor and lymph node image representations. Various experimental results validate the effectiveness of our proposed method.

## News
Our paper has been early accpeted by MICCAI 2024 under (5/6/6) !!! 🥳🥳

## A Quick Overview 

<img width="850" height="500" src="https://github.com/wuchengyu123/MMFusion/blob/main/framework.jpg">


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

Due to existing ethical review and privacy concerns related to the patients from whom the dataset was collected, the authors have no rights to make the dataset publicly available. Currently, it is recommended use your own multimodal dataset to run the code. The data types and requirements can be set according to ``/dataloader/Dataset.py``.


## Acknowlegment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [CARD](https://github.com/XzwHan/CARD): CARD: Classification and Regression Diffusion Models.
- [DiffMIC](https://github.com/scott-yjyang/DiffMIC): DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification


## Citation

If you find this repository helpful, please consider citing our paper:
```
@inproceedings{miccai24mmfusion,
  title={MMFusion: Multi-modality Diffusion Model for Lymph Node Metastasis Diagnosis in Esophageal Cancer},
  author={Wu, Chengyu and Wang, Chengkai and Zhou, Huiyu and Zhang, Yatao and Wang, Qifeng and Wang, Yaqi and Wang, Shuai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={469--479},
  year={2024},
  organization={Springer}
}
```

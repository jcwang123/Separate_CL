# Separate Contrastive Learning for Organs-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation

## Introduction

This is an official release of the paper **Separate Contrastive Learning for Organs-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation**.

It is accepted by **AAAI-2022 Oral** and has been awarded an **AAAI student scholarship**.

> [**Separate Contrastive Learning for Organs-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation**](https://arxiv.org/abs/2112.02743),   <br/>
> **Jiacheng Wang**, Xiaomeng Li, Yiming Han, Jing Qin, Liansheng Wang, Zhou Qichao<br/>
> In: Association for the Advancement of Artificial Intelligence (AAAI), 2022  <br/>
> [[arXiv](https://arxiv.org/abs/2112.02743)][[Bibetex](https://github.com/jcwang123/Separate_CL#citation)]

<div align="center" border=> <img src=framework.png width="600" > </div>

## TODO List

1. Complete the resources ...

2. Evaluate the effectiveness on more vision tasks ...


## Code List

- [x] Comparison Methods, [Here](https://github.com/jcwang123/AwesomeContrastiveLearning)
- [x] Network
- [x] Pre-processing
- [ ] Training Codes

## Usage

<!-- ### For PDDCA dataset -->

1. First, you can download the dataset at [PDDCA](https://www.imagenglab.com/newsite/pddca/). To preprocess the dataset and save as ".png", run:

    ```bash
    $ python utils/prepare_data.py
    ```

    Note that some cases lack the complete annotation, so that we can obtain 32 cases with full annotation in the end.

2. To create the region set, alternatively run:

    ```bash
    $ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method fb --min_size 400
    $ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slic --n_segments 32
    $ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slice --n_segments 32
    ```

## Citation

If you find **SepaReg** useful in your research, please consider citing:


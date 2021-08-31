# Separate Contrastive Learning for Organs-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation

Detailed codes will come soon!

## Code List

- [x] Comparison Methods, [Here](https://github.com/jcwang123/AwesomeContrastiveLearning)
- [x] Network
- [x] Pre-processing
- [ ] Training Codes

## Usage

### For PDDCA dataset

1. First, you can download the dataset at [PDDCA](https://www.imagenglab.com/newsite/pddca/). To preprocess the dataset and save as ".png", run:

```bash
$ python utils/prepare_data.py
```

2. To create the region set, run:

```bash
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method fb --min_size 400
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slic --n_segments 32
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slice --n_segments 32
```

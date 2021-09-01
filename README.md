# A paper

Prior work hasn't separated feature extraction of regions from an image, so that regions from an image will share more similar semantics than that from different images, causing unfair comparison of semantics cross images. Thus, this paper delivers a separate learning scheme that divides an image into regions and encodes their feature separately for comparing regions to more diverse images equally.

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

Note that some cases lack the complete annotation, so that we can obtain 32 cases with full annotation in the end.

2. To create the region set, run:

```bash
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method fb --min_size 400
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slic --n_segments 32
$ python utils/prepare_segs.py --dataset pddca --filter_method all --seg_method slice --n_segments 32
```

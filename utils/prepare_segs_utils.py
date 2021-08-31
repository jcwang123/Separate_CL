import numpy as np
import os, cv2
import skimage
from skimage import segmentation
from skimage.measure import label
import scipy.ndimage.morphology as snm


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh):  # change this by your need
    mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask


# remove superpixels within the empty regions
def superpix_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = lbvs.max()
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append(max_lb)
    raw_seg2d = raw_seg2d * mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0:
            continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            lb_new += 1

    return out_seg2d


def superpix_wrapper(img, seg_method, min_size, n_segments, fg_thresh=1e-4):
    if seg_method == 'fb':
        raw_seg = skimage.segmentation.felzenszwalb(img, min_size)
    elif seg_method == 'slic':
        raw_seg = skimage.segmentation.slic(img,
                                            n_segments,
                                            compactness=15,
                                            sigma=0.8,
                                            convert2lab=True)
    elif seg_method == 'slice':
        raw_seg = skimage.segmentation.slic(img,
                                            n_segments,
                                            compactness=15,
                                            sigma=0.8,
                                            convert2lab=False)
    img_0 = img[..., 0]
    _fgm = fg_mask2d(img_0, fg_thresh)
    _out_seg = superpix_masking(raw_seg, _fgm)
    return _fgm, _out_seg


def pddca_seg(filter_method, seg_method, min_size, n_segments):
    BASE_DIR = "/raid/hym/data/PDDCA/cropped_dataset"
    OUT_DIR = os.path.join("/raid/hym/data/PDDCA/cropped_superpixel",
                           seg_method)
    if seg_method == 'fb':
        OUT_DIR += '_{}'.format(str(min_size))
    elif seg_method == 'slic' or seg_method == 'slice':
        OUT_DIR += '_{}'.format(str(n_segments))
    for root, dirs, files in os.walk(BASE_DIR):
        for sub_dir in dirs:
            out_path = os.path.join(OUT_DIR, filter_method, sub_dir)
            makedir(out_path)
            print(out_path)
            img_names = os.listdir(os.path.join(root, sub_dir))
            img_names = list(filter(lambda x: x.startswith('image'),
                                    img_names))
            img_names = sorted(
                img_names,
                key=lambda x: int(x.split('-')[-1].split('.png')[0]))
            pids = [
                int(pid.split("-")[-1].split('.png')[0]) for pid in img_names
            ]
            for img_fid, pid in zip(img_names, pids):
                img_path = os.path.join(root, sub_dir, img_fid)
                img = cv2.imread(img_path)
                out_fg, out_seg = superpix_wrapper(img, seg_method, min_size,
                                                   n_segments)
                img = img[..., 0]
                seg_arr = np.int32(out_seg)
                segvs = np.unique(seg_arr)
                segvs = list(segvs)
                for i in segvs:
                    mask2d = np.zeros(out_seg.shape)
                    mask2d[seg_arr == i] = 1
                    seg2d = (img * mask2d).astype('uint8')
                    if seg2d.max() > 0:
                        xmin, xmax = np.where(seg2d > 0)[0].min(), np.where(
                            seg2d > 0)[0].max()
                        ymin, ymax = np.where(seg2d > 0)[1].min(), np.where(
                            seg2d > 0)[1].max()
                        crop_img = seg2d[xmin:xmax, ymin:ymax]
                        if (crop_img.shape[0] != 0) and (crop_img.shape[1] !=
                                                         0):
                            slice_name = "slice-" + "{:03d}".format(
                                int(pid)) + "-index-" + "{:03d}".format(
                                    int(i)) + ".png"
                            if filter_method == 'filtered':
                                ratioPix = (
                                    np.sum(crop_img > 0) /
                                    (crop_img.shape[0] * crop_img.shape[1]))
                                ratioShape = crop_img.shape[
                                    1] / crop_img.shape[0]
                                if ratioPix > 0.2 and ratioShape < 15:
                                    cv2.imwrite(
                                        os.path.join(out_path, slice_name),
                                        crop_img)
                            elif filter_method == 'all':
                                cv2.imwrite(os.path.join(out_path, slice_name),
                                            crop_img)

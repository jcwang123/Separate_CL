import os
import cv2
import numpy as np


def center_crop(x, size):
    bias = (x.shape[0] - size) // 2
    return x[bias:-bias, bias:-bias]


def PDDCA():
    import nrrd
    IMG_FOLDER = '/raid/hym/data/PDDCA/ori_data'
    OUT_FOLDER = '/raid/hym/data/PDDCA/cropped_dataset/'
    i = 0

    label_names = os.listdir(
        '/raid/hym/data/PDDCA/ori_data/0522c0001/structures/')
    label_names.sort()
    indexes = [0, 2, 5, 6, 7, 8]

    print(label_names)
    for patient in os.listdir(IMG_FOLDER):
        print('####################################')
        print(patient)

        outdir = os.path.join(OUT_FOLDER, 'Patient-{:03d}'.format(i))

        imgs, _ = nrrd.read(os.path.join(IMG_FOLDER, patient, 'img.nrrd'))
        imgs = np.clip(imgs, -1000, 1000)
        imgs = ((imgs + 1000) / 2000 * 255).astype('uint8')

        labels = [np.ones_like(imgs) * 0.5]
        tag = 0
        for name in label_names:
            if not os.path.exists(
                    os.path.join(IMG_FOLDER, patient, 'structures', name)):
                print('{} not exist'.format(name))
                tag = 1
                break
            label, _ = nrrd.read(
                os.path.join(IMG_FOLDER, patient, 'structures', name))
            labels.append(label)
        if tag:
            continue
        labels = np.array(labels)
        labels = labels[indexes]
        lbls = np.argmax(labels, 0) * 10
        os.makedirs(outdir, exist_ok=True)
        for slice_id in range(imgs.shape[-1]):
            img = center_crop(imgs[..., slice_id], 384)
            lbl = center_crop(lbls[..., slice_id], 384)

            cv2.imwrite(
                os.path.join(outdir, 'image-{:03d}.png'.format(slice_id)), img)
            cv2.imwrite(
                os.path.join(outdir, 'label-{:03d}.png'.format(slice_id)), lbl)
        i = i + 1


if __name__ == '__main__':
    PDDCA()
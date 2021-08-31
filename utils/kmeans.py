import cv2, random, argparse, time
import os, sys

sys.path.append('/raid/hym/code/Superpixel_CL/')
from net.resnet import resnet50
import numpy as np
import glob
import albumentations as A
from tqdm import tqdm
import torch

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    choices=['breast', 'npc', 'pddca'],
                    default='npc')
parser.add_argument('--train_obj',
                    type=str,
                    choices=['seg', 'img'],
                    default='seg')
parser.add_argument('--filter_method',
                    type=str,
                    choices=['all', 'filtered'],
                    default='all')
parser.add_argument('--seg_method',
                    type=str,
                    choices=['fb', 'slic', 'slice'],
                    default='fb')
parser.add_argument('--pre_dir', type=str)
parser.add_argument('--min_size', type=int)
parser.add_argument('--n_segments', type=int)
parser.add_argument('--k', help="number of clusters", type=int)
parser.add_argument('--gpu', type=str, default='4')
cfg = parser.parse_args()


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    if cfg.train_obj == 'img':
        root_path = os.path.join('/raid/hym/data', cfg.dataset,
                                 'cropped_dataset')
    elif cfg.train_obj == 'seg':
        if cfg.seg_method == 'fb':
            root_path = os.path.join(
                '/raid/hym/data', cfg.dataset, 'cropped_superpixel',
                '{}_{}'.format(cfg.seg_method,
                               str(cfg.min_size)), cfg.filter_method)
        elif cfg.seg_method == 'slic' or cfg.seg_method == 'slice':
            root_path = os.path.join(
                '/raid/hym/data', cfg.dataset, 'cropped_superpixel',
                '{}_{}'.format(cfg.seg_method,
                               str(cfg.n_segments)), cfg.filter_method)
            cfg.pre_file = os.path.join(
                '/raid/hym/logs/superpixel_cl/', cfg.dataset, 'pretrain/seg',
                '{}_{}'.format(cfg.seg_method, str(cfg.n_segments)), 'base',
                cfg.pre_dir, 'ckpt/pretrain_weight_ep_100.t7')

    if cfg.dataset == 'pddca':
        root_path = root_path.replace('pddca', 'PDDCA')
        cfg.dataset = 'PDDCA'
        patient_ids = [i for i in range(20)]
    else:
        patient_ids = [i for i in range(70)]
    print(root_path)

    seq_names = os.listdir(root_path)

    images = []
    for p_id in patient_ids:
        if cfg.train_obj == 'img':
            images += glob.glob(
                os.path.join(root_path, '{}/image*'.format(seq_names[p_id])))
        elif cfg.train_obj == 'seg':
            images += glob.glob(
                os.path.join(root_path, '{}/slice*'.format(seq_names[p_id])))
    images.sort()
    transf = A.Compose([A.Resize(128, 128)])

    resnet = resnet50().cuda()
    sys.path.append('/raid/hym/code/byol-pytorch-master/byol_pytorch')
    from byol_pytorch import BYOL
    learner = torch.load(cfg.pre_file).cuda()

    embeddings = []
    for path_img in tqdm(images):
        img = cv2.imread(path_img).copy()
        img = torch.from_numpy(img)
        tsf = transf(image=img.numpy().astype('uint8'))
        img = tsf['image']
        img = torch.from_numpy(img) / 255.
        img = img.permute(2, 0, 1)
        img = img.unsqueeze_(0).cuda()

        with torch.no_grad():
            emb = learner(img, return_embedding=True)[0].cpu().numpy()
            embeddings.append(emb)
    embeddings = np.array(embeddings)

    start = time.time()
    print("Compute clustering ... ", end="", flush=True)
    kmeans = KMeans(n_clusters=cfg.k, n_jobs=-1, random_state=0)
    idx = kmeans.fit_predict(embeddings)
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)

    start = time.time()
    print("Generate output file ... ", end="", flush=True)
    if cfg.train_obj == 'seg':
        if cfg.seg_method == 'fb':
            out_base = os.path.join(
                '/raid/hym/data/{}/kmeans_superpixel'.format(cfg.dataset),
                '{}_{}'.format(cfg.dataset, cfg.seg_method,
                               str(cfg.min_size)), cfg.filter_method)
        elif cfg.seg_method == 'slic' or cfg.seg_method == 'slice':
            out_base = os.path.join(
                '/raid/hym/data/{}/kmeans_superpixel'.format(cfg.dataset),
                '{}_{}'.format(cfg.seg_method,
                               str(cfg.n_segments)), cfg.filter_method)

    os.makedirs(out_base, exist_ok=True)

    for i in range(len(idx)):
        out_dir = os.path.join(out_base, '{:02d}'.format(idx[i]))
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(images[i], 0)
        slice_name = '{:02d}'.format(len(os.listdir(out_dir)) + 1) + '.png'
        cv2.imwrite(os.path.join(out_dir, slice_name), img)
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)
    return


if __name__ == "__main__":
    main()

import argparse, sys

sys.path.append('/raid/hym/code/Superpixel_CL')
from utils.prepare_segs_utils import *

parser = argparse.ArgumentParser(description='prepare superpixels and filter')
parser.add_argument('--dataset',
                    type=str,
                    choices=['breast', 'npc', 'pddca'],
                    default='breast')
parser.add_argument('--filter_method',
                    type=str,
                    default='all',
                    choices=['all', 'filtered'])
parser.add_argument('--seg_method',
                    type=str,
                    default='slic',
                    choices=['fb', 'slic', 'slice'])
parser.add_argument('--min_size', type=int)
parser.add_argument('--n_segments', type=int)
cfg = parser.parse_args()


def main():
    if cfg.dataset == 'breast':
        Breast_seg(cfg.filter_method, cfg.seg_method, cfg.min_size,
                   cfg.n_segments)
    elif cfg.dataset == 'npc':
        NPC_seg(cfg.filter_method, cfg.seg_method, cfg.min_size,
                cfg.n_segments)
    elif cfg.dataset == 'pddca':
        pddca_seg(cfg.filter_method, cfg.seg_method, cfg.min_size,
                  cfg.n_segments)


if __name__ == '__main__':
    main()

import numpy as np
from skimage import measure
import torch


def get_largest_connected_component_mask(img_arr: np.array,
                                         connectivity: Optional[int] = None
                                         ) -> torch.Tensor:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (batch_size, spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    largest_cc = np.zeros(shape=img_arr.shape, dtype=img_arr.dtype)
    for i, item in enumerate(img_arr):
        item = measure.label(item, connectivity=connectivity)
        if item.max() != 0:
            largest_cc[i,
                       ...] = item == (np.argmax(np.bincount(item.flat)[1:]) +
                                       1)
    return torch.as_tensor(largest_cc, device=img.device)
import torch
import torch.nn.functional as F

def Transformer_2D(src, flow):
    b = flow.shape[0]
    h = flow.shape[2]
    w = flow.shape[3]

    size = (h,w)

    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = grid.to(torch.float32)
    grid = grid.repeat(b,1,1,1)
    new_locs = grid+flow
    shape = flow.shape[2:]
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 1)
    new_locs = new_locs[..., [1 , 0]]
    warped = F.grid_sample(src,new_locs,padding_mode="border")

    return warped

def load_model(model, pre_dir):
    state_dict = torch.load(pre_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pre_dir)
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                  key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            print('Drop parameter {}.'.format(key))
            
    for key in model_state_dict:
        if key not in state_dict:
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]  
    model.load_state_dict(state_dict, strict=False)
    return model
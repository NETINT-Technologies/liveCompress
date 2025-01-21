import torch
from PIL import Image

"""A helper function to load a checkpoint from a specified path onto a predefined model.
Args:
    --checkpoint_path: The path to the checkpoint file.
    --net: The model to load the checkpoint onto.
    --device: The device to load the checkpoint onto (e.g. 'cuda' or 'cpu').
"""
def load_checkpoints(checkpoint_path, net, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #remove the 'module.' prefix from the keys of the state dictionary, as the model expects this
    checkpoint['state_dict'] = {key.replace('module.', '', 1): checkpoint['state_dict'][key] for key in list(checkpoint['state_dict'].keys())}
    net.load_state_dict(checkpoint['state_dict'])
import torch
from PIL import Image
import numpy as np
import os
import glob

from zoedepth_misc import colorize

_DEVICE = "cuda"

def load_zoe():
    #local_repo = "/Depth/ZoeDepth"
    #zoe = torch.hub.load(local_repo, "ZoeD_N", source="local", pretrained=True)
    #zoe = zoe.to(_DEVICE)
    repo = "isl-org/ZoeDepth"
    zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    zoe = zoe.to(_DEVICE)
    return zoe

def get_depthdir(image_path, dest_dir, sub):
    if sub:
        #if the dataset has sub folders setup accordingly
        sub_dir = '/'.join(image_path.split('/')[-3:-1])
        depth_dir = os.path.join(dest_dir, sub_dir)
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
    else:
        depth_dir = dest_dir
    return depth_dir



def extract_depth(image_list, dest_dir, sub=False, colorimage=False):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # load model
    zoe = load_zoe()

    for image_path in image_list:
        # set up path
        img_id = image_path.split('/')[-1].split('.')[0]
        depth_dir = get_depthdir(image_path, dest_dir, sub)
        depth_map_path = os.path.join(depth_dir, img_id)
        if os.path.exists(depth_map_path):
            print(f'Depth image {img_id} exists')
            continue

        # render and save depth map
        img = Image.open(image_path)
        depth = zoe.infer_pil(img)
        np.save(depth_map_path, depth)

        # for visualization
        if colorimage:
            colored_depth = colorize(depth)
            depth_img = Image.fromarray(colored_depth)
            depth_img.save(os.path.join(depth_dir, img_id+'_zoe.png'))


if __name__ == "__main__":
    db_img_list = glob.glob("./test_images/*.png")
    db_depth_dir = "./test_images"
    print(f"Processing {len(db_img_list)}")
    extract_depth(db_img_list, db_depth_dir, sub=False, colorimage=True)
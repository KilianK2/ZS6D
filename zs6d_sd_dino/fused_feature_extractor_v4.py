import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from extractor_sd import load_model, process_features_and_mask, get_mask
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from extractor_dino import ViTExtractor
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def compute_pair_feature(model, aug, save_path, files, category, mask=False, dist='cos', real_size=960):



    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 244
    model_dict = {'small': 'dinov2_vits14',
                  'base': 'dinov2_vitb14',
                  'large': 'dinov2_vitl14',
                  'giant': 'dinov2_vitg14'}

    model_type = 'dino_vits8'
    layer =  9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet =  'key'
    stride =  4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    input_text = "a photo of " + category[-1][0] if TEXT_INPUT else None

    current_save_results = 0

    N = len(files) // 2
    pbar = tqdm(total=N)
    result = []
    if 'Anno' in files[0]:
        Anno = True
    else:
        Anno = False
    for pair_idx in range(N):

        # Load image 1
        img1 = Image.open(files[2 * pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load image 2
        img2 = Image.open(files[2 * pair_idx + 1]).convert('RGB')
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        with torch.no_grad():

            else:
                if not ONLY_DINO:
                    features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False,
                                                          raw=True)
                    features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text, mask=False,
                                                          raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            if dist == 'l1' or dist == 'l2':
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if FUSE_DINO and not ONLY_DINO:
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)



            if DRAW_DENSE:
                if not Anno:
                    mask1 = get_mask(model, aug, img1, category[0])
                    mask2 = get_mask(model, aug, img2, category[-1])
                if Anno:
                    mask1 = torch.Tensor(
                        resize(img1, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1) > 0).to(device)
                    mask2 = torch.Tensor(
                        resize(img2, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1) > 0).to(device)
                    print(mask1.shape, mask2.shape, mask1.sum(), mask2.sum())
                if ONLY_DINO or not FUSE_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)

                img1_desc_reshaped = img1_desc.permute(0, 1, 3, 2).reshape(-1, img1_desc.shape[-1], num_patches,
                                                                           num_patches)
                img2_desc_reshaped = img2_desc.permute(0, 1, 3, 2).reshape(-1, img2_desc.shape[-1], num_patches,
                                                                           num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs(mask2, mask1, img2, img1, img2_desc_reshaped,
                                                                      img1_desc_reshaped, mask=mask)

                if not os.path.exists(f'{save_path}/{category[0]}'):
                    os.makedirs(f'{save_path}/{category[0]}')
                fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.axis('off')
                ax2.axis('off')
                ax1.imshow(src_color_map)
                ax2.imshow(trg_dense_output)
                fig_colormap.savefig(f'{save_path}/{category[0]}/{pair_idx}_colormap.png')
                plt.close(fig_colormap)

            if DRAW_SWAP:
                if not DRAW_DENSE:
                    mask1 = get_mask(model, aug, img1, category[0])
                    mask2 = get_mask(model, aug, img2, category[-1])

                if (ONLY_DINO or not FUSE_DINO) and not DRAW_DENSE:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)

                img1_desc_reshaped = img1_desc.permute(0, 1, 3, 2).reshape(-1, img1_desc.shape[-1], num_patches,
                                                                           num_patches)
                img2_desc_reshaped = img2_desc.permute(0, 1, 3, 2).reshape(-1, img2_desc.shape[-1], num_patches,
                                                                           num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs_replace(mask2, mask1, img2, img1,
                                                                              img2_desc_reshaped, img1_desc_reshaped,
                                                                              mask=mask, resolution=156)
                if not os.path.exists(f'{save_path}/{category[0]}'):
                    os.makedirs(f'{save_path}/{category[0]}')
                fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.axis('off')
                ax2.axis('off')
                ax1.imshow(src_color_map)
                ax2.imshow(trg_dense_output)
                fig_colormap.savefig(f'{save_path}/{category[0]}/{pair_idx}_swap.png')
                plt.close(fig_colormap)
            if not DRAW_SWAP and not DRAW_DENSE:
                result.append([img1_desc.cpu(), img2_desc.cpu()])
            else:
                result.append([img1_desc.cpu(), img2_desc.cpu(), mask1.cpu(), mask2.cpu()])

    pbar.update(1)
    return result
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

"""
I am using: 
CO_PCA = True -> Dimensionality Reduction for better comparison between two images
FUSE_DINO = True -> Using both SD and DINO feature extraction
ONLY_DINO = False -> Using both SD and DINO feature extraction
"""



# MASK:
# - Enable masking in feature extraction
# - Focuses processing on specific areas of the image
# - Useful for tasks requiring region-specific analysis
MASK = True

# VER:
# - Indicates the version of the model or method
# - Helps in tracking different iterations and ensuring compatibility
# - Important for maintaining consistent results across different environments
VER = "v1-5"

# PCA:
# - Determines if Principal Component Analysis is applied to individual features
# - Disabled here to possibly retain original feature dimensions
# - Can affect performance and accuracy depending on the task
PCA = False

# CO_PCA:
# - Enables Coupled PCA for processing pairs of images
# - Aligns feature spaces to enhance comparability
# - Useful for tasks requiring correlated or comparative analysis
CO_PCA = True

# PCA_DIMS:
# - Specifies dimensions for PCA reduction stages
# - Array format allows specifying different dimensions at multiple stages
# - Critical for controlling feature complexity and extraction speed
PCA_DIMS = [256, 256, 256]

# SIZE:
# - Sets the image size for processing
# - Critical for ensuring features are extracted at a consistent scale
# - Affects the granularity of extracted features
SIZE = 960

# EDGE_PAD:
# - Controls whether edges of images are padded during processing
# - Padding can help avoid artifacts at image boundaries during transformations
# - Set to False to maintain original image boundaries
EDGE_PAD = False

# FUSE_DINO:
# - Controls the fusion of features from the DINO model with other model outputs
# - Disabled here, indicating no feature fusion
# - Can be enabled to enhance feature robustness by combining model strengths
FUSE_DINO = 1

# ONLY_DINO:
# - Restricts feature extraction to only use the DINO model
# - Disabled to allow use of additional models or methods
# - When enabled, ensures feature uniformity and model-specific processing
ONLY_DINO = 0

# DINOV2:
# - Indicates use of the DINOV2 model variant
# - Affects feature extraction methods and underlying architecture
# - Important for leveraging advanced self-supervised learning capabilities
DINOV2 = True

# MODEL_SIZE:
# - Specifies the size of the model to be used ('small', 'base', 'large', 'giant')
# - Larger sizes typically provide more detailed features at increased computational cost
# - Choice impacts the complexity and detail of feature extraction
MODEL_SIZE = 'base'

# DRAW_DENSE:
# - Enables generation of dense feature visualization outputs
# - Useful for examining how features populate the image space
# - Set to True to activate this visualization feature
DRAW_DENSE = 1

# DRAW_SWAP:
# - Enables swapping of features or image parts between different images
# - Useful for comparative and creative visualizations like style transfer
# - Set to True to activate feature swapping
DRAW_SWAP = 1

# TEXT_INPUT:
# - Controls whether text inputs are incorporated into feature processing
# - Disabled to focus purely on visual features
# - Enabling can enhance processing by providing context or additional data modalities
TEXT_INPUT = False

# SEED:
# - Sets the seed value for random number generation
# - Ensures consistent randomness across different runs for reproducibility
# - Critical in experiments where outcome consistency is necessary
SEED = 42

# TIMESTEP:
# - Specifies a particular timestep or epoch in a time-based or iterative process
# - Used for scheduling or synchronizing specific tasks
# - Important for managing progress in tasks that evolve over time
TIMESTEP = 100

# DIST:
# - Chooses the distance metric based on feature fusion settings
# - 'l2' for Euclidean distance, 'cos' for cosine similarity
# - Metric selection impacts how features are compared or clustered
DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'

# Logic to maintain feature processing consistency
# - Forces feature fusion if only DINO features are used
if ONLY_DINO:
    FUSE_DINO = True


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP)




def sd_dino_compute_pair_feature(model, aug, save_path, files, category, mask=False, dist='cos', real_size=960):
    # Convert category to a list if it's not already one
    if type(category) == str:
        category = [category]

    # Determine the image size based on whether DINOV2 is enabled
    img_size = 840 if DINOV2 else 244

    # Define model dictionary for different model scales (small, base, large, giant)
    model_dict = {'small': 'dinov2_vits14',
                  'base': 'dinov2_vitb14',
                  'large': 'dinov2_vitl14',
                  'giant': 'dinov2_vitg14'}

    # Determine the model type and layer based on the model size and whether DINOV2 is enabled
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39

    # Feature extraction details based on model type
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]

    # Initialize the Vision Transformer (ViT) Extractor with selected model
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    # Generate input text if TEXT_INPUT flag is enabled
    input_text = "a photo of " + category[-1][0] if TEXT_INPUT else None

    current_save_results = 0

    # Prepare to process pairs of images
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
            # Process each image through the model and the feature extractor if not using CO_PCA
            if not CO_PCA:
                if not ONLY_DINO:
                    # Process features with the main model

                    # TODO: img1_desc has features extracted form the SD model (when ONLY_DINO is false)
                    # img1_desc variable stores the features extracted by the SD model
                    img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False,
                                                          pca=PCA).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3,
                                                                                                               2)
                    img2_desc = process_features_and_mask(model, aug, img2_input, category[-1], input_text=input_text,
                                                          mask=mask, pca=PCA).reshape(1, 1, -1,
                                                                                      num_patches ** 2).permute(0, 1, 3,
                                                                                                                2)

                # Fuse features from DINO if enabled
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)

                    # TODO: img1_desc_dino has features extracted from DINO model (when FUSE_DINO is true)
                    # When FUSE_DINO is true, the img1_desc_dino variable holds the features extracted from the DINO model for image 1
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            else:
                # If CO_PCA is enabled, process raw features for PCA
                if not ONLY_DINO:
                    features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False,
                                                          raw=True)
                    features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text, mask=False,
                                                          raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

                # Again, fuse features from DINO if enabled
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            # Normalize features if using L1 or L2 distance
            if dist == 'l1' or dist == 'l2':
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            # Combine features from both models if FUSE_DINO and not ONLY_DINO are true
            if FUSE_DINO and not ONLY_DINO:
                # TODO: Fusion of SD and DINO Features for image1 (FUSE_DINO = True and ONLY_DINO = False)
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)

            # Use only DINO features if ONLY_DINO is true
            if ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino

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
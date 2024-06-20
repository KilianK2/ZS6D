import torch
from PIL import Image
import numpy as np
from extractor_sd import load_model, process_features_and_mask, get_mask
from extractor_dino import ViTExtractor
from utils.utils_correspondence import resize, co_pca, find_nearest_patchs, find_nearest_patchs_replace
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class FusedFeatureExtractor:
    def __init__(self, model_type='dino_vits8', dinov2=True, stride=4, pca_dims=[256, 256, 256], image_size=960,
                 device='cuda', seed=42):
        self.model_type = model_type
        self.dinov2 = dinov2
        self.stride = stride
        self.pca_dims = pca_dims
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.model, self.aug = load_model(diffusion_ver="v1-5", image_size=image_size, num_timesteps=100)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = True

        model_dict = {'small': 'dinov2_vits14', 'base': 'dinov2_vitb14', 'large': 'dinov2_vitl14',
                      'giant': 'dinov2_vitg14'}
        self.model_type = model_dict['base'] if self.dinov2 else 'dino_vits8'
        self.layer = 11 if self.dinov2 else 9
        if 'l' in self.model_type:
            self.layer = 23
        elif 'g' in self.model_type:
            self.layer = 39
        self.facet = 'token' if self.dinov2 else 'key'

        self.extractor = ViTExtractor(self.model_type, self.stride, device=self.device)
        self.patch_size = self.extractor.model.patch_embed.patch_size[
            0] if self.dinov2 else self.extractor.model.patch_embed.patch_size
        self.num_patches = int(self.patch_size / self.stride * (self.image_size // self.patch_size - 1) + 1)

    def preprocess(self, img):
        return resize(img, self.image_size, resize=True, to_pil=True, edge=False)

    def extract_features(self, img1, img2, input_text=None, mask=False, co_pca=True, pca=False, fuse_dino=True,
                         only_dino=False):
        img1_input = self.preprocess(img1)
        img2_input = self.preprocess(img2)

        results = {}
        with torch.no_grad():
            if not co_pca:
                if not only_dino:
                    img1_desc = process_features_and_mask(self.model, self.aug, img1_input, input_text=input_text,
                                                          mask=mask, pca=pca).reshape(1, 1, -1,
                                                                                      self.num_patches ** 2).permute(0,
                                                                                                                     1,
                                                                                                                     3,
                                                                                                                     2)
                    img2_desc = process_features_and_mask(self.model, self.aug, img2_input, input_text=input_text,
                                                          mask=mask, pca=pca).reshape(1, 1, -1,
                                                                                      self.num_patches ** 2).permute(0,
                                                                                                                     1,
                                                                                                                     3,
                                                                                                                     2)
                    results['img1_desc'] = img1_desc
                    results['img2_desc'] = img2_desc

                if fuse_dino:
                    img1_batch = self.extractor.preprocess_pil(img1)
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), self.layer,
                                                                        self.facet)
                    img2_batch = self.extractor.preprocess_pil(img2)
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), self.layer,
                                                                        self.facet)
                    results['img1_desc_dino'] = img1_desc_dino
                    results['img2_desc_dino'] = img2_desc_dino

            else:
                if not only_dino:
                    features1 = process_features_and_mask(self.model, self.aug, img1_input, input_text=input_text,
                                                          mask=mask, raw=True)
                    features2 = process_features_and_mask(self.model, self.aug, img2_input, input_text=input_text,
                                                          mask=mask, raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, self.pca_dims)
                    img1_desc = processed_features1.reshape(1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)
                    img2_desc = processed_features2.reshape(1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)
                    results['img1_desc'] = img1_desc
                    results['img2_desc'] = img2_desc

                if fuse_dino:
                    img1_batch = self.extractor.preprocess_pil(img1)
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), self.layer,
                                                                        self.facet)
                    img2_batch = self.extractor.preprocess_pil(img2)
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), self.layer,
                                                                        self.facet)
                    results['img1_desc_dino'] = img1_desc_dino
                    results['img2_desc_dino'] = img2_desc_dino

        return results

# Example usage:
# model = FusedFeatureExtractor()
# img1 = Image.open("path_to_image1.jpg")
# img2 = Image.open("path_to_image2.jpg")
# results = model.extract_features(img1, img2)

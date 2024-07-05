import os
import json
import torch
from tqdm import tqdm
import numpy as np
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import logging
from src.pose_extractor import PoseViTExtractor
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


class Fused_ZS6D:

    def __init__(self, templates_gt_path, norm_factors_path, model_type='dino_vits8', stride=4, subset_templates=8,
                 max_crop_size=80):
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size

        try:
            with open(os.path.join(templates_gt_path), 'r') as f:
                self.templates_gt = json.load(f)

            with open(os.path.join(norm_factors_path), 'r') as f:
                self.norm_factors = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load templates or norm_factors: {e}")
            raise

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.extractor = PoseViTExtractor(model_type=self.model_type, stride=self.stride, device=self.device)
        """TODO: Utilizing FusedFeatureExtractor"""
        # self.extractor = FusedPoseViTExtractor(model_type=self.model_type, stride=self.stride, device=self.device)

        self.templates_desc = {}
        templates_gt_subset = {}
        try:
            for obj_id, template_labels in tqdm(self.templates_gt.items()):
                self.templates_desc[obj_id] = torch.cat(
                    [torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                     for i, template_label in enumerate(template_labels)
                     if i % subset_templates == 0], dim=0)

                templates_gt_subset[obj_id] = [template_label for i, template_label in
                                               enumerate(template_labels) if i % subset_templates == 0]
        except Exception as e:
            self.logger.error(f"Error processing template descriptors: {e}")
            raise

        self.templates_gt = templates_gt_subset

        self.logger.info("Preparing templates and loading of extractor is done!")

    def get_pose(self, model, aug, img, obj_id, mask, cam_K, bbox=None):
        try:
            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=224)

            """Fused Descriptors"""
            desc = self.get_fused_features(model, aug, img)
            print(f"Shape of fused desc_sd_dino before squeezing: {desc.shape}")
            desc_sd_dino = desc.squeeze(0).squeeze(0).detach().cpu()
            print(f"Shape of fused desc_sd_dino input before find_template_cpu: {desc_sd_dino.shape}")
            print(f"Shape of templates_desc[{obj_id}] before find_template_cpu: {self.templates_desc[obj_id].shape}")


            matched_templates = utils.find_template_cpu(desc_sd_dino, self.templates_desc[obj_id], num_results=1)

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            with torch.no_grad():
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0]
                else:
                    crop_size = self.max_crop_size

                resize_factor = float(crop_size) / img_crop.size[0]

                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans(img_crop,
                                                                                                          template,
                                                                                                          num_pairs=20,
                                                                                                          load_size=crop_size)

                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(
                    f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)
                img_uv = cv2.resize(img_uv, (crop_size, crop_size))

                R_est, t_est = utils.get_pose_from_correspondences(points1, points2,
                                                                   y_offset, x_offset,
                                                                   img_uv, cam_K,
                                                                   self.norm_factors[str(obj_id)],
                                                                   scale_factor=1.0,
                                                                   resize_factor=resize_factor)

                return R_est, t_est
        except Exception as e:
            self.logger.error(f"Error in get_pose: {e}")
            raise


    def get_fused_features(self, model, aug, img):
        PCA = True
        EDGE_PAD = False
        real_size = 960
        img_size = 840 # if DINOV2 else 244
        """model_dict = {'small': 'dinov2_vits14',
                      'base': 'dinov2_vitb14',
                      'large': 'dinov2_vitl14',
                      'giant': 'dinov2_vitg14'}
        """
        model_type = 'dinov2_vitb14' # if DINOV2 else 'dino_vits8'

        layer = 11 # if DINOV2 else 9
        facet = 'token' # if DINOV2 else 'key'
        stride = 14 # if DINOV2 else 4
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        extractor_sd_dino = ViTExtractor(model_type, stride, device=device)
        patch_size = extractor_sd_dino.model.patch_embed.patch_size[0] # if DINOV2 else extractor.model.patch_embed.patch_size
        num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

        input_text = None


        img1_input = resize(img, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img, img_size, resize=True, to_pil=True, edge=EDGE_PAD)


        with torch.no_grad():
            img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False,
                                                  pca=PCA).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            print(f"Shape of img1_desc (SD) features: {img1_desc.shape}")
            img1_batch = extractor_sd_dino.preprocess_pil(img1)
            print(f"Shape of img1_batch: {img1_batch.shape}")
            img1_desc_dino = extractor_sd_dino.extract_descriptors(img1_batch.to(device), layer, facet)
            print(f"Shape of img1_desc_dino: {img1_desc_dino.shape}")



            img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
            print(f"Shape of img1_desc (SD) normalized: {img1_desc.shape}")

            img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
            print(f"Shape of img1_desc_dino normalized: {img1_desc_dino.shape}")

            img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)

            print(f"Shape of img1_desc (Fused) after fusion: {img1_desc.shape}")



        return img1_desc.cpu()
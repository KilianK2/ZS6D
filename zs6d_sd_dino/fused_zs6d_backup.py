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
from extractor_sd import process_features_and_mask
from utils.utils_correspondence import resize
import torch.nn.functional as F


class Fused_ZS6D:

    def __init__(self, templates_gt_path, norm_factors_path, model_type, stride, subset_templates=10,
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


    def get_pose(self, model, aug, stride, img, obj_id, mask, cam_K, bbox=None):
        try:
            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            """TODO: Preprocessing of Image"""
            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=224)

            """Setup SD-DINO"""
            patch_size = self.extractor.model.patch_embed.patch_size[0]
            num_patches = int(patch_size / stride * (224 // patch_size - 1) + 1)
            real_size = 960
            img_resized = resize(img, real_size, resize=True, to_pil=True, edge=False)

            """TODO: Integration of SD DINO"""
            with torch.no_grad():
                """SD Features"""
                features_sd = process_features_and_mask(model, aug, img_resized, input_text=None, mask=False,
                                                        raw=True)  # sd
                dim = [256, 256, 256]
                processed_features_sd = co_pca_single(features_sd, dim)
                desc_sd = processed_features_sd.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                # normalize the features
                desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                """DINO Features"""
                desc_dino = self.extractor.extract_descriptors(img_prep.to(self.device), layer=11, facet='key', bin=False,
                                                          include_cls=True)
                # normalize the features
                desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)

                """Fused SD-DINO Features"""
                desc_sd_dino = torch.cat((desc_sd, desc_dino), dim=-1)  # Fusion of SD DINO
                desc = desc_sd_dino.squeeze(0).squeeze(0).detach().cpu()

            matched_templates = utils.find_template_cpu(desc, self.templates_desc[obj_id], num_results=1)

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


def co_pca_single(features_sd, dim):
    processed_features = {}

    s5_size = features_sd['s5'].shape[-1]
    s4_size = features_sd['s4'].shape[-1]
    s3_size = features_sd['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features_sd['s5'].reshape(features_sd['s5'].shape[0], features_sd['s5'].shape[1], -1)
    s4_1 = features_sd['s4'].reshape(features_sd['s4'].shape[0], features_sd['s4'].shape[1], -1)
    s3_1 = features_sd['s3'].reshape(features_sd['s3'].shape[0], features_sd['s3'].shape[1], -1)

    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensor in zip(['s5', 's4', 's3'], [s5_1, s4_1, s3_1]):
        target_dim = target_dims[name]

        features = tensor.permute(0, 2, 1)  # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)

        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim])  # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1)  # Bx(d)x(t_x+t_y)


        processed_features[name] = features
    # reshape the features
    processed_features['s5'] = processed_features['s5'].reshape(processed_features['s5'].shape[0], -1, s5_size,
                                                                  s5_size)
    processed_features['s4'] = processed_features['s4'].reshape(processed_features['s4'].shape[0], -1, s4_size,
                                                                  s4_size)
    processed_features['s3'] = processed_features['s3'].reshape(processed_features['s3'].shape[0], -1, s3_size,
                                                                  s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features['s5'] = F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]),
                                             mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features['s5'] = torch.cat([processed_features['s4'], processed_features['s5']], dim=1)

    # Set s3 as the new s4
    processed_features['s4'] = processed_features['s3']

    # Remove s3 from the features dictionary
    processed_features.pop('s3')

    # Current order are layer 8, 5, 2
    features_gether_s4_s5 = torch.cat([processed_features['s4'], F.interpolate(processed_features['s5'], size=(
    processed_features['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features_gether_s4_s5
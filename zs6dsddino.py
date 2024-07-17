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
#from src.pose_extractor import PoseViTExtractor
from zs6d_sd_dino.pose_extractor_sd_dino import PoseViTExtractorSdDino
from external.sd_dino.extractor_sd import process_features_and_mask
from external.sd_dino.utils.utils_correspondence import resize



class ZS6DSdDino:

    def __init__(self, model_sd, aug_sd, image_size_dino, image_size_sd, layer, facet, templates_gt_path, norm_factors_path, model_type='dinov2_vitb14', stride=14, subset_templates=1,
                 max_crop_size=182):
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size

        self.model_sd = model_sd
        self.aug_sd = aug_sd
        self.image_size_dino = image_size_dino
        self.image_size_sd = image_size_sd
        self.layer = layer
        self.facet = facet
        #self.patch_size = self.model.patch_embed.patch_size[0]
        #self.num_patches = int(self.patch_size / self.stride * (image_size_dino // self.patch_size))

        try:
            with open(os.path.join(templates_gt_path), 'r') as f:
                self.templates_gt = json.load(f)

            with open(os.path.join(norm_factors_path), 'r') as f:
                self.norm_factors = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load templates or norm_factors: {e}")
            raise

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.extractor = PoseViTExtractorSdDino(model_type=self.model_type, stride=self.stride, device=self.device)



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

    def get_pose(self, num_patches, img, obj_id, mask, cam_K, bbox=None):
        try:
            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=self.image_size_dino)



            with torch.no_grad():
                """SD-DINO"""
                # check if this works (but img_crop is correct)
                img_base = img_crop.convert('RGB')

                # Resizing
                img_sd = resize(img_base, self.image_size_sd, resize=True, to_pil=True, edge=False)


                # Stable Diffusion
                desc_sd = process_features_and_mask(self.model_sd, self.aug_sd, img_sd, input_text=None, mask=False, pca=True).reshape(
                    1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                print(f"Shape of SD features: {desc_sd.shape}")

                # DinoV2
                desc_dino = self.extractor.extract_descriptors(img_prep.to(self.device), self.layer, self.facet)
                print(f"Shape of DINO features: {desc_dino.shape}")


                # normalization
                desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
                desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                # fusion
                desc_sd_dino_raw = torch.cat((desc_sd, desc_dino), dim=-1)
                print(f"Shape of SD-DINO features: {desc_sd_dino_raw.shape}")

                desc_sd_dino = desc_sd_dino_raw.squeeze(0).squeeze(0).detach().cpu()

            matched_templates = utils.find_template_cpu(desc_sd_dino, self.templates_desc[obj_id], num_results=1) #found template

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            with torch.no_grad():
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0] #crop_size = 184
                else:
                    crop_size = self.max_crop_size

                resize_factor = float(crop_size) / img_crop.size[0] #rezise_factor = 1.0

                #img_crop = Image.fromarray(img_prep.squeeze().cpu().numpy())

                """TODO: Check if input for kmeans is correct. What about crop_size? load_size = crop_size"""
                input_image = img_base
                input_pil = img_prep
                template_image = template
                template_pil, _, _ = self.extractor.preprocess(template_image, load_size=self.image_size_dino)

                # v7 and v6 are valid but wrong results
                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans_sd_dino_v7(self.image_size_sd, self.model_sd, self.aug_sd, num_patches, input_image, input_pil,
                                                                                                          template_image, template_pil,
                                                                                                          num_pairs=20,
                                                                                                          load_size=self.image_size_dino)


                #points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans(img_crop,
                #                                                                                          template,
                #                                                                                          num_pairs=20,
                #                                                                                          load_size=crop_size)

                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(
                    f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)

                """TODO: Check if rezising is correct"""
                img_uv = cv2.resize(img_uv, (self.image_size_dino, self.image_size_dino))

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


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
from src.pose_extractor_sd_dino import PoseViTExtractorSdDino
from external.sd_dino.extractor_sd import process_features_and_mask
from external.sd_dino.utils.utils_correspondence import resize
import matplotlib.pyplot as plt


class ZS6DSdDino:

    def __init__(self, model_sd, aug_sd, image_size_dino, image_size_sd, layer, facet, templates_gt_path, norm_factors_path, model_type='dinov2_vitb14', stride=14, subset_templates=15,
                 max_crop_size=840):
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
            def show_debug_image(img, name):
                if isinstance(img, np.ndarray):
                    height, width = img.shape[:2]
                elif isinstance(img, Image.Image):
                    width, height = img.size
                else:
                    raise ValueError("Unsupported image type")

                title = f"{name} - Size: {width}x{height}"

                plt.imshow(img)
                plt.title(title)
                plt.axis('off')
                plt.show()

            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=self.image_size_dino)

            show_debug_image(img_crop, "Cropped Image of full scene")




            with torch.no_grad():
                """SD-DINO"""
                # check if this works (but img_crop is correct)
                img_base = img_crop.convert('RGB')

                # Resizing
                img_sd = resize(img_base, self.image_size_sd, resize=True, to_pil=True, edge=False)

                show_debug_image(np.array(img_sd), "Resized SD Image of full scene")

                # Stable Diffusion
                desc_sd = process_features_and_mask(self.model_sd, self.aug_sd, img_sd, input_text=None, mask=False, pca=True).reshape(
                    1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                print(f"Shape of SD features: {desc_sd.shape}")

                # DINOv2
                desc_dino = self.extractor.extract_descriptors(img_prep.to(self.device), self.layer, self.facet)
                print(f"Shape of DINO features: {desc_dino.shape}")


                # Normalization
                desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
                desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                # Fusion
                desc_sd_dino_raw = torch.cat((desc_sd, desc_dino), dim=-1)
                print(f"Shape of SD-DINO features: {desc_sd_dino_raw.shape}")

                desc_sd_dino = desc_sd_dino_raw.squeeze(0).squeeze(0).detach().cpu()

            matched_templates = utils.find_template_cpu(desc_sd_dino, self.templates_desc[obj_id], num_results=1) #found template

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            show_debug_image(np.array(template), "Matched Template")


            with torch.no_grad():
                """
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0] #crop_size = 37
                else:
                    crop_size = self.max_crop_size

                resize_factor = float(crop_size) / img_crop.size[0] #rezise_factor = 1.0
                """

                #img_crop = Image.fromarray(img_prep.squeeze().cpu().numpy())

                """ Find Correspondences """
                input_image = img_base # size 37x37
                input_pil = img_prep
                template_image = template # size 840x840
                template_pil, _, _ = self.extractor.preprocess(template_image, load_size=self.image_size_dino)



                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans_sd_dino( input_image, input_pil, template_image, template_pil, num_patches, self.model_sd, self.aug_sd, self.image_size_sd,
                                                                                                          num_pairs=20,
                                                                                                          load_size=self.image_size_dino)

                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(
                    f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)

                show_debug_image(img_uv, "Original UV Image")

                """TODO: Check if rezising is correct
                # resizing of points1 and points2

                # resizing of img_uv
                crop_size = img_crop.size[0]

                # Calculate the scaling factor
                scale_factor = crop_size / self.image_size_dino

                # Scale the points
                points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
                points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

                img_uv = cv2.resize(img_uv, (crop_size, crop_size))
                """

                crop_size = img_crop.size[0]

                def scale_points(points, original_size, new_size, num_patches, stride):
                    scale_factor = new_size / original_size
                    patch_size = original_size // num_patches
                    scaled_points = []
                    for y, x in points:
                        # Convert patch coordinates to pixel coordinates
                        y_pixel = y * stride + patch_size // 2
                        x_pixel = x * stride + patch_size // 2

                        # Scale the pixel coordinates
                        y_scaled = y_pixel * scale_factor
                        x_scaled = x_pixel * scale_factor

                        scaled_points.append((y_scaled, x_scaled))
                    return scaled_points

                # Calculate the scaling factor
                scale_factor = crop_size / self.image_size_dino

                # Scale points1 and points2
                points1 = scale_points(points1, self.image_size_dino, crop_size, num_patches, self.stride)
                points2 = scale_points(points2, self.image_size_dino, crop_size, num_patches, self.stride)

                # Resize img_uv to match the crop size
                img_uv = cv2.resize(img_uv, (crop_size, crop_size))

                """
                # Adjust camera matrix for the new image size
                cam_K_scaled = cam_K.copy()
                cam_K_scaled[0, 0] *= scale_factor
                cam_K_scaled[1, 1] *= scale_factor
                cam_K_scaled[0, 2] *= scale_factor
                cam_K_scaled[1, 2] *= scale_factor
                """

                # Scale offsets
                y_offset = y_offset * scale_factor
                x_offset = x_offset * scale_factor

                # set resizing_factor
                resize_factor = 1

                show_debug_image(img_uv, "Resized UV Image")

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


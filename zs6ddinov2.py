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


class ZS6DDinoV2:

    def __init__(self, image_size_dino, image_size_sd, layer, facet, templates_gt_path, norm_factors_path, model_type='dinov2_vitb14', stride=14, subset_templates=15,
                 max_crop_size=840):
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size


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
                #img_sd = resize(img_base, self.image_size_sd, resize=True, to_pil=True, edge=False)
                #show_debug_image(np.array(img_sd), "Resized SD Image of full scene")

                # Stable Diffusion
                #desc_sd = process_features_and_mask(self.model_sd, self.aug_sd, img_sd, input_text=None, mask=False, pca=True).reshape(
                #    1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                #print(f"Shape of SD features: {desc_sd.shape}")

                # DINOv2
                desc_dino = self.extractor.extract_descriptors(img_prep.to(self.device), self.layer, self.facet)
                print(f"Shape of DINO features: {desc_dino.shape}")

                # Normalization
                desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
                #desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                # Fusion
                #desc_sd_dino_raw = torch.cat((desc_sd, desc_dino), dim=-1)
                #print(f"Shape of SD-DINO features: {desc_sd_dino_raw.shape}")

                desc_dinov2 = desc_dino.squeeze(0).squeeze(0).detach().cpu()
                print(f"Shape of Template: {self.templates_desc[obj_id].shape}")

            matched_templates = utils.find_template_cpu(desc_dinov2, self.templates_desc[obj_id], num_results=1) #found template

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            show_debug_image(np.array(template), "Matched Template")

            mask_template = self.generate_template_mask(template)
            #self.visualize_template_and_mask(template, mask_template)

            with torch.no_grad():

                """ Find Correspondences """
                cropped_image = img_base # size 37x37
                cropped_pil = img_prep
                template_image = template # size 840x840
                template_pil, _, _ = self.extractor.preprocess(template_image, load_size=self.image_size_dino)

                crop_size = img_crop.size[0]

                # Calculate the scaling factor
                scale_factor = crop_size / self.image_size_dino

                #points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans_sd_dino_v5(cropped_image, cropped_pil, template_image, template_pil, num_patches, self.model_sd, self.aug_sd, self.image_size_sd, scale_factor,
                #                                                                                          num_pairs=20)

                #self.display_image_variables(cropped_image,cropped_pil,template_image,template_pil)



                # working find_correspondences_sd_dino_own7
                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_kmeans_dinoV2_v13(mask_crop, mask_template, cropped_image, cropped_pil, template_image, template_pil, "", "", self.image_size_sd, scale_factor, num_patches)



                #self.display_points_on_images(cropped_image, template_image, points1, points2)




                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(
                    f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)

                show_debug_image(img_uv, "Original UV Image")

                # resizing of img_uv
                img_uv = cv2.resize(img_uv, (crop_size, crop_size))

                #self.visualize_uv_points(img_uv, points2)

                show_debug_image(img_uv, "Resized UV Image")

                resize_factor = float(crop_size) / img_crop.size[0]

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

    def display_image_variables(self, cropped_image, cropped_pil, template_image, template_pil):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Image Variables and Sizes', fontsize=16)

        images = [cropped_image, cropped_pil, template_image, template_pil]
        titles = ['cropped_image', 'cropped_pil', 'template_image', 'template_pil']

        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // 2
            col = i % 2

            if isinstance(img, Image.Image):
                axs[row, col].imshow(img)
                size = img.size
            elif isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)  # Remove batch dimension if present
                if img.dim() == 3:
                    img = img.permute(1, 2, 0).cpu().numpy()
                elif img.dim() == 2:
                    img = img.cpu().numpy()
                axs[row, col].imshow(img)
                size = f"{img.shape[1]}x{img.shape[0]}"
            elif isinstance(img, np.ndarray):
                axs[row, col].imshow(img)
                size = f"{img.shape[1]}x{img.shape[0]}"
            else:
                axs[row, col].text(0.5, 0.5, f"Unsupported type: {type(img)}", ha='center', va='center')
                size = "Unknown"

            axs[row, col].set_title(f"{title}\nSize: {size}")
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.show()


    def display_points_on_images(self, cropped_image, template_image, points1, points2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Display cropped image with points1
        if isinstance(cropped_image, Image.Image):
            ax1.imshow(cropped_image)
        elif isinstance(cropped_image, np.ndarray):
            ax1.imshow(cropped_image)
        elif isinstance(cropped_image, torch.Tensor):
            if cropped_image.dim() == 4:
                cropped_image = cropped_image.squeeze(0)
            if cropped_image.dim() == 3:
                cropped_image = cropped_image.permute(1, 2, 0).cpu().numpy()
            elif cropped_image.dim() == 2:
                cropped_image = cropped_image.cpu().numpy()
            ax1.imshow(cropped_image)

        ax1.scatter(*zip(*points1), c='r', s=40)
        ax1.set_title("Cropped Image with Points 1")

        # Display template image with points2
        if isinstance(template_image, Image.Image):
            ax2.imshow(template_image)
        elif isinstance(template_image, np.ndarray):
            ax2.imshow(template_image)
        elif isinstance(template_image, torch.Tensor):
            if template_image.dim() == 4:
                template_image = template_image.squeeze(0)
            if template_image.dim() == 3:
                template_image = template_image.permute(1, 2, 0).cpu().numpy()
            elif template_image.dim() == 2:
                template_image = template_image.cpu().numpy()
            ax2.imshow(template_image)

        ax2.scatter(*zip(*points2), c='b', s=40)
        ax2.set_title("Template Image with Points 2")

        plt.tight_layout()
        plt.show()

    def visualize_uv_points(self, img_uv, points2, title="UV Map with Correspondence Points"):
        """
        Visualize the points2 on the img_uv (UV map).

        Args:
        img_uv (np.ndarray): The UV map image
        points2 (list of tuples): Correspondence points to be plotted on the UV map
        title (str): Title for the plot
        """
        # Create a figure
        plt.figure(figsize=(12, 12))

        # Display the UV map
        plt.imshow(img_uv)

        # Plot the points
        for point in points2:
            plt.plot(point[1], point[0], 'ro', markersize=5)  # Red dots for points

        # Set title and remove axis ticks
        plt.title(title)
        plt.axis('off')

        # Add text to show image size
        plt.text(5, img_uv.shape[0] + 15, f"UV Map Size: {img_uv.shape[1]}x{img_uv.shape[0]}",
                 fontsize=10, color='black')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def generate_template_mask(self, template_image):
        """
        Generate a binary mask from the template image.

        Args:
        template_image (PIL.Image or np.ndarray): The template image

        Returns:
        np.ndarray: Binary mask of the template image
        """
        # Convert to numpy array if it's a PIL Image
        if isinstance(template_image, Image.Image):
            template_array = np.array(template_image)
        else:
            template_array = template_image

        # Convert to grayscale if it's a color image
        if len(template_array.shape) == 3:
            gray = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = template_array

        # Threshold the image to create a binary mask
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Optional: Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask = mask // 255

        return mask

    def visualize_template_and_mask(self, template_image, mask):
        """
        Visualize the template image and its corresponding mask side by side.

        Args:
        template_image (PIL.Image or np.ndarray): The template image
        mask (np.ndarray): The binary mask of the template image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Display template image
        ax1.imshow(template_image)
        template_size = template_image.shape[:2] if isinstance(template_image, np.ndarray) else template_image.size[
                                                                                                ::-1]
        ax1.set_title(f"Template Image - Size: {template_size[0]}x{template_size[1]}")
        ax1.axis('off')

        # Display mask
        ax2.imshow(mask, cmap='gray')
        mask_size = mask.shape
        ax2.set_title(f"Template Mask - Size: {mask_size[0]}x{mask_size[1]}")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()



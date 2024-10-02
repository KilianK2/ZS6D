import cv2
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
#import src.extractor as extractor
from PIL import Image
from typing import Union, List, Tuple
from src.correspondences import chunk_cosine_sim
from sklearn.cluster import KMeans
import numpy as np
import time
#import zs6d_sd_dino.sd_dino.extractor_dino as extractor
from external.kmeans_pytorch.kmeans_pytorch import kmeans
from external.sd_dino.extractor_sd import process_features_and_mask, get_mask
from external.sd_dino.utils.utils_correspondence import resize, find_nearest_patchs
import torch.nn.functional as F
#from ZS6D.src import extractor
#import src.extractor as extractor
from external.sd_dino.extractor_dino import ViTExtractor
from src.pose_extractor import PoseViTExtractor
import torch
import math
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns

class PoseViTExtractorSdDino(PoseViTExtractor):

    def __init__(self, model_type, stride, device, model: nn.Module = None):
        self.model_type = model_type
        self.stride = stride
        self.device = device
        self.model = model
        super().__init__(model_type=self.model_type, stride=self.stride, model=self.model, device=self.device)


    def find_correspondences_nearest_neighbor_sd_dino(self, image_size_sd, model_sd, aug_sd, num_patches, input_image,
                                                      input_pil, template_image, template_pil, num_pairs: int = 10,
                                                      load_size: int = 840,
                                                      layer: int = 11, facet: str = 'token', bin: bool = False,
                                                      thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        start_time_corr = time.time()
        start_time_desc = time.time()

        """Input Image"""
        desc_dino_input = self.extract_descriptors(input_pil.to(self.device), layer, facet)
        image_sd_input = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd_input = process_features_and_mask(model_sd, aug_sd, image_sd_input, mask=False, pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # normalization
        descriptors1_dino = desc_dino_input / desc_dino_input.norm(dim=-1, keepdim=True)
        descriptors1_sd = desc_sd_input / desc_sd_input.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((descriptors1_sd, descriptors1_dino), dim=-1)
        num_patches1, load_size1 = num_patches, load_size

        """Template Image"""
        desc_dino_template = self.extract_descriptors(template_pil.to(self.device), layer, facet, bin)
        image_sd_template = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd_template = process_features_and_mask(model_sd, aug_sd, image_sd_template, mask=False, pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # normalization
        descriptors2_dino = desc_dino_template / desc_dino_template.norm(dim=-1, keepdim=True)
        descriptors2_sd = desc_sd_template / desc_sd_template.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((descriptors2_sd, descriptors2_dino), dim=-1)
        num_patches2, load_size2 = num_patches, load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_nn = time.time()
        # find nearest neighbors
        distances, indices = torch.topk(similarities[0, 0], k=num_pairs, largest=True)
        end_time_nn = time.time()
        elapsed_nn = end_time_nn - start_time_nn

        # get coordinates to show
        img1_indices_to_show = torch.arange(num_patches1 * num_patches1, device=self.device)[indices]
        img2_indices_to_show = indices
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_corr = time.time()
        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    def find_correspondences_fastkmeans_sd_dino_v6(self, image_size_sd, model_sd, aug_sd, num_patches, input_image,
                                                   input_pil, template_image, template_pil, num_pairs: int = 10,
                                                   load_size: int = 840,
                                                   layer: int = 11, facet: str = 'token', bin: bool = False,
                                                   thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()

        """Input Image"""
        desc_dino_input = self.extract_descriptors(input_pil.to(self.device), layer, facet, bin)

        image_sd_input = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd_input = process_features_and_mask(model_sd, aug_sd, image_sd_input, mask=False, pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # normalization
        descriptors1_dino = desc_dino_input / desc_dino_input.norm(dim=-1, keepdim=True)
        descriptors1_sd = desc_sd_input / desc_sd_input.norm(dim=-1, keepdim=True)

        descriptors1 = torch.cat((descriptors1_sd, descriptors1_dino), dim=-1)
        num_patches1, load_size1 = num_patches, load_size

        """Template Image"""
        desc_dino_template = self.extract_descriptors(template_pil.to(self.device), layer, facet, bin)

        image_sd_template = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd_template = process_features_and_mask(model_sd, aug_sd, image_sd_template, mask=False, pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # normalization
        descriptors2_dino = desc_dino_template / desc_dino_template.norm(dim=-1, keepdim=True)
        descriptors2_sd = desc_sd_template / desc_sd_template.norm(dim=-1, keepdim=True)

        descriptors2 = torch.cat((descriptors2_sd, descriptors2_dino), dim=-1)
        num_patches2, load_size2 = num_patches, load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # Convert PIL images to NumPy arrays in BGR color order
        # input_np = cv2.cvtColor(np.array(input_pil), cv2.COLOR_RGB2BGR)
        # template_np = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2BGR)

        # Generate segmentation masks for both images
        mask1 = get_mask(model_sd, aug_sd, input_image)
        mask2 = get_mask(model_sd, aug_sd, template_image)

        # Resize masks to match the spatial dimensions of the feature maps
        resized_mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(0).float(), size=(num_patches1, num_patches1),
                                      mode='nearest').squeeze().bool()
        resized_mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0).float(), size=(num_patches2, num_patches2),
                                      mode='nearest').squeeze().bool()

        # Flatten masks
        flat_mask1 = resized_mask1.flatten()
        flat_mask2 = resized_mask2.flatten()

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        # image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        image_idxs = torch.arange(num_patches1 * num_patches1, device=self.device)

        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by segmentation mask.
        bbs_mask = torch.bitwise_and(bbs_mask, flat_mask1)

        # Maybe adjustments here
        bbs_mask = torch.bitwise_and(bbs_mask, flat_mask2[nn_1])

        # applying k-means to extract k high quality well distributed correspondence pairs
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length

        start_time_kmeans = time.time()
        cluster_ids_x, cluster_centers = kmeans(X=normalized,
                                                num_clusters=n_clusters,
                                                distance='cosine',
                                                tqdm_flag=False,
                                                iter_limit=200,
                                                device=self.device)

        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean descriptor similarity
        bb_sims = torch.sum(bb_descs1 * bb_descs2, dim=-1)
        ranks = bb_sims

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1 * num_patches1, device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb

        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    # working but wrong scaling
    def find_correspondences_fastkmeans_sd_dino(self, input_image, input_pil, template_image, template_pil, num_patches, model_sd, aug_sd, image_size_sd, num_pairs: int = 10, load_size: int = 840,
                                        layer: int = 11, facet: str = 'token', bin: bool = True) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()


        """ Descriptor 1: Input """
        # Stable Diffusion
        img_sd1 = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False) # size 960 x 960

        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                            pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # DINOv2
        desc_dino1 = self.extract_descriptors(input_pil.to(self.device), layer, facet)

        # Normalization
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)

        # Fusion
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)

        num_patches1, load_size1 = self.num_patches, self.load_size


        """ Descriptor 2: Template """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False) # size 960 x 960

        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)

        # Normalization
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)

        # Fusion
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)

        num_patches2, load_size2 = self.num_patches, self.load_size


        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # same
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length

        start_time_kmeans = time.time()
        cluster_ids_x, cluster_centers = kmeans(X=normalized,
                                                num_clusters=n_clusters,
                                                distance='cosine',
                                                tqdm_flag=False,
                                                iter_limit=200,
                                                device=self.device)

        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        # Select the best correspondences based on descriptor similarity
        best_corr_indices = []
        for k in range(n_clusters):
            cluster_indices = np.where(kmeans_labels == k)[0]
            if len(cluster_indices) > 0:
                cluster_sims = sim_1[bbs_mask][cluster_indices]
                best_corr_index = cluster_indices[torch.argmax(cluster_sims)]
                best_corr_indices.append(best_corr_index)

        # ab hier stimmt bis schluss
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[best_corr_indices]
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb

        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    # working but wrong scaling
    def find_correspondences_fastkmeans_sd_dino_v2(self, input_image, input_pil, template_image, template_pil, num_patches,
                                                model_sd, aug_sd, image_size_sd, num_pairs: int = 10,
                                                load_size: int = 840,
                                                layer: int = 11, facet: str = 'token', bin: bool = True) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()

        """ Descriptor 1: Input """
        # Stable Diffusion
        img_sd1 = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino1 = self.extract_descriptors(input_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
        num_patches1, load_size1 = self.num_patches, self.load_size

        """ Descriptor 2: Template """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)
        num_patches2, load_size2 = self.num_patches, self.load_size

        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # ab hier bis next gleich

        # Compute similarities between descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # Find mutual nearest neighbors (best buddies)
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)
        sim_2, nn_2 = torch.max(similarities, dim=-2)
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]


        bbs_mask = nn_2[nn_1] == image_idxs

        # Select best buddy descriptors
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]  # Descriptors from image 1 that are best buddies
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]  # Corresponding descriptors from image 2

        # Concatenate best buddy descriptors for clustering
        all_keys_together = torch.cat((bb_descs1, bb_descs2), dim=1)
        n_clusters = min(num_pairs, len(all_keys_together))
        length = torch.sqrt((all_keys_together ** 2).sum(dim=1, keepdim=True))
        normalized = all_keys_together / length

        # Perform k-means clustering on the concatenated descriptors
        start_time_kmeans = time.time()
        cluster_ids_x, cluster_centers = kmeans(X=normalized,
                                                num_clusters=n_clusters,
                                                distance='cosine',
                                                tqdm_flag=False,
                                                iter_limit=200,
                                                device=self.device)
        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        # Anders bis next
        # Select the best correspondence from each cluster
        bb_topk_sims = torch.full((n_clusters,), float('-inf'), device=self.device)
        bb_indices_to_show = torch.full((n_clusters,), -1, dtype=torch.long, device=self.device)
        ranks = sim_1[bbs_mask]

        # ab hier bis schluss gleich
        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # Convert selected correspondences to image coordinates
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[bb_indices_to_show]
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    def find_correspondences_fastkmeans_sd_dino_v3(self, input_image, input_pil, template_image, template_pil,
                                                   num_patches,
                                                   model_sd, aug_sd, image_size_sd, num_pairs: int = 10,
                                                   load_size: int = 840,
                                                   layer: int = 11, facet: str = 'token', bin: bool = True) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()

        """ Descriptor 1: Input """
        # Stable Diffusion
        img_sd1 = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino1 = self.extract_descriptors(input_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
        num_patches1, load_size1 = self.num_patches, self.load_size

        """ Descriptor 2: Template """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)
        num_patches2, load_size2 = self.num_patches, self.load_size

        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # Perform K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_pairs)

        # Flatten the descriptors for k-means
        desc1_flat = descriptors1.view(-1, descriptors1.shape[-1]).cpu().numpy()
        desc2_flat = descriptors2.view(-1, descriptors2.shape[-1]).cpu().numpy()

        # Fit k-means on the concatenated descriptors
        kmeans.fit(torch.cat((desc1_flat, desc2_flat), axis=0))

        # Get cluster centers and labels
        cluster_centers = kmeans.cluster_centers_
        labels1 = kmeans.predict(desc1_flat)
        labels2 = kmeans.predict(desc2_flat)

        # Find the nearest neighbors in the template descriptors for each input descriptor
        nn_1 = []
        for i in range(len(labels1)):
            same_cluster_indices = (labels2 == labels1[i]).nonzero()[0]
            if len(same_cluster_indices) > 0:
                distances = torch.norm(torch.tensor(desc2_flat[same_cluster_indices]) - torch.tensor(desc1_flat[i]),
                                       dim=1)
                nn_1.append(same_cluster_indices[distances.argmin()].item())
            else:
                nn_1.append(-1)

        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = torch.tensor(nn_1, device=self.device)
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_bb = time.time()
        end_time_corr = time.time()


        return points1, points2, input_pil, template_pil

    def find_correspondences_fastkmeans_sd_dino_v4(self, input_image, input_pil, template_image, template_pil,
                                                   num_patches,
                                                   model_sd, aug_sd, image_size_sd, num_pairs: int = 10,
                                                   load_size: int = 840,
                                                   layer: int = 11, facet: str = 'token', bin: bool = True) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()

        """ Descriptor 1: Input """
        # Stable Diffusion
        img_sd1 = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino1 = self.extract_descriptors(input_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
        num_patches1, load_size1 = self.num_patches, self.load_size

        """ Descriptor 2: Template """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)
        num_patches2, load_size2 = self.num_patches, self.load_size

        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # Compute cosine similarity matrix
        sim = torch.einsum('bcn,bcm->bnm', descriptors1.squeeze(), descriptors2.squeeze())

        # Get top-k nearest neighbors for each descriptor in image 1
        k = min(10, sim.shape[1])  # Choose k based on the number of patches in image 2
        topk_values, topk_indices = torch.topk(sim, k=k, dim=1)

        # Prepare data for k-means clustering
        n_clusters = num_pairs
        data = []
        for i in range(topk_indices.shape[0]):
            for j in range(k):
                data.append([i, topk_indices[i, j].item(), topk_values[i, j].item()])
        data = np.array(data)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data[:, :2])

        # Select the point with the highest similarity score from each cluster
        selected_points = []
        for cluster in range(n_clusters):
            cluster_points = data[cluster_labels == cluster]
            best_point = cluster_points[np.argmax(cluster_points[:, 2])]
            selected_points.append(best_point[:2].astype(int))

        # Convert selected points to image coordinates
        points1, points2 = [], []
        for p1, p2 in selected_points:
            y1, x1 = divmod(p1, num_patches1[1])
            y2, x2 = divmod(p2, num_patches2[1])

            x1_show = int(x1 * self.stride[1] + self.stride[1] + self.p // 2)
            y1_show = int(y1 * self.stride[0] + self.stride[0] + self.p // 2)
            x2_show = int(x2 * self.stride[1] + self.stride[1] + self.p // 2)
            y2_show = int(y2 * self.stride[0] + self.stride[0] + self.p // 2)

            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_corr = time.time()
        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    def find_correspondences_sd_dino_own8(self, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        # Reshape descriptors for correspondence finding
        src_features = template_image_desc.squeeze().permute(1, 0)  # (num_patches^2, feature_dim)
        tgt_features = cropped_image_desc.squeeze().permute(1, 0)  # (num_patches^2, feature_dim)

        # Find the best correspondences
        best_matches = []
        for patch_idx in tqdm(range(num_patches * num_patches)):
            distances = torch.linalg.norm(tgt_features - src_features[patch_idx], dim=1)
            tgt_patch_idx = torch.argmin(distances)
            best_matches.append((patch_idx, tgt_patch_idx.item(), distances[tgt_patch_idx].item()))

        # Sort matches by distance (lowest to highest)
        best_matches.sort(key=lambda x: x[2])

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        for src_idx, tgt_idx, _ in best_matches[:num_pairs]:
            src_y = src_idx // num_patches
            src_x = src_idx % num_patches
            tgt_y = tgt_idx // num_patches
            tgt_x = tgt_idx % num_patches

            # Convert patch coordinates to pixel coordinates
            x1_show = tgt_x * self.stride[1] + self.p // 2
            y1_show = tgt_y * self.stride[0] + self.p // 2
            x2_show = src_x * self.stride[1] + self.p // 2
            y2_show = src_y * self.stride[0] + self.p // 2

            # Scale the points
            x1_scaled = int(x1_show * scale_factor)
            y1_scaled = int(y1_show * scale_factor)
            x2_scaled = int(x2_show * scale_factor)
            y2_scaled = int(y2_show * scale_factor)

            points1.append((x1_scaled, y1_scaled))
            points2.append((x2_scaled, y2_scaled))

        return points1, points2, cropped_image, template_image

    def find_correspondences_sd_dino_own7(self, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        # Reshape descriptors for correspondence finding
        src_features = template_image_desc.squeeze().permute(1, 0)  # (num_patches^2, feature_dim)
        tgt_features = cropped_image_desc.squeeze().permute(1, 0)  # (num_patches^2, feature_dim)

        # Find the best correspondences
        distances = torch.cdist(src_features, tgt_features)
        src_to_tgt_indices = torch.argmin(distances, dim=1)
        best_distances = torch.gather(distances, 1, src_to_tgt_indices.unsqueeze(1)).squeeze()

        # Sort matches by distance (lowest to highest)
        sorted_indices = torch.argsort(best_distances)
        best_matches = [(idx.item(), src_to_tgt_indices[idx].item(), best_distances[idx].item()) for idx in
                        sorted_indices[:num_pairs]]

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        for src_idx, tgt_idx, _ in best_matches:
            src_y = src_idx // num_patches
            src_x = src_idx % num_patches
            tgt_y = tgt_idx // num_patches
            tgt_x = tgt_idx % num_patches

            # Convert patch coordinates to pixel coordinates
            x1_show = tgt_x * self.stride[1] + self.p // 2
            y1_show = tgt_y * self.stride[0] + self.p // 2
            x2_show = src_x * self.stride[1] + self.p // 2
            y2_show = src_y * self.stride[0] + self.p // 2

            # Scale the points
            x1_scaled = int(x1_show * scale_factor)
            y1_scaled = int(y1_show * scale_factor)
            x2_scaled = int(x2_show * scale_factor)
            y2_scaled = int(y2_show * scale_factor)

            points1.append((x1_scaled, y1_scaled))
            points2.append((x2_scaled, y2_scaled))

        return points1, points2, cropped_image, template_image

    def find_correspondences_sd_dino6b(self, mask_cropped_image, cropped_image, cropped_pil, template_image,
                                                 template_pil,
                                                 model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                                 num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        # points1 for cropped image -> target_image here
        # points2 for template image -> src_image here

        mask_cropped_image = torch.from_numpy(mask_cropped_image).float()
        if mask_cropped_image.dim() == 3 and mask_cropped_image.size(0) == 1:
            mask_cropped_image = mask_cropped_image.squeeze(0)

        start_time_corr = time.time()
        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        cropped_image_feature_reshaped = cropped_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        template_image_feature_reshaped = template_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()

        patch_size = 60  # the resolution of the output image, set to 256 could be faster

        resized_cropped_image_mask = F.interpolate(mask_cropped_image.unsqueeze(0).unsqueeze(0),
                                                   size=(patch_size, patch_size),
                                                   mode='nearest').squeeze().cuda()
        template_image_feature_upsampled = F.interpolate(template_image_feature_reshaped, size=(patch_size, patch_size),
                                                         mode='bilinear').squeeze()
        cropped_image_feature_upsampled = F.interpolate(cropped_image_feature_reshaped, size=(patch_size, patch_size),
                                                        mode='bilinear').squeeze()

        # mask only the cropped image feature
        cropped_image_feature_upsampled = cropped_image_feature_upsampled * resized_cropped_image_mask.repeat(
            cropped_image_feature_upsampled.shape[0], 1, 1)
        # Set the masked area to a very small number
        cropped_image_feature_upsampled[cropped_image_feature_upsampled == 0] = -100000

        # Calculate the cosine similarity between src_feature and tgt_feature
        template_image_features_2d = template_image_feature_upsampled.reshape(template_image_feature_upsampled.shape[0],
                                                                              -1).permute(1, 0)
        cropped_image_features_2d = cropped_image_feature_upsampled.reshape(cropped_image_feature_upsampled.shape[0],
                                                                            -1).permute(1, 0)

        # Find the best correspondences
        best_matches = []
        for patch_idx in tqdm(range(patch_size * patch_size)):
            # Only consider patches within the mask of the cropped image
            if resized_cropped_image_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                distances = torch.linalg.norm(template_image_features_2d - cropped_image_features_2d[patch_idx], dim=1)
                tgt_patch_idx = torch.argmin(distances)
                best_matches.append((patch_idx, tgt_patch_idx.item(), distances[tgt_patch_idx].item()))

        # Sort matches by distance (lowest to highest)
        best_matches.sort(key=lambda x: x[2])

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        for src_idx, tgt_idx, _ in best_matches[:num_pairs]:
            src_y = src_idx // patch_size
            src_x = src_idx % patch_size
            tgt_y = tgt_idx // patch_size
            tgt_x = tgt_idx % patch_size

            # Convert patch coordinates to pixel coordinates
            x1_show = (int(src_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(src_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(tgt_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(tgt_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2

            # Scale the points
            x1_scaled = int(x1_show * scale_factor)
            y1_scaled = int(y1_show * scale_factor)
            x2_scaled = int(x2_show * scale_factor)
            y2_scaled = int(y2_show * scale_factor)

            points1.append((x1_scaled, y1_scaled))
            points2.append((x2_scaled, y2_scaled))

        return points1, points2, cropped_image, template_image


    def find_correspondences_sd_dino6a(self, cropped_image, cropped_pil, template_image, template_pil,
                                     model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                     num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        # points1 for cropped image -> target_image here
        # points2 for template image -> src_image here

        start_time_corr = time.time()
        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        cropped_image_feature_reshaped = cropped_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        template_image_feature_reshaped = template_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()

        print(f"Shape of cropped_image_feature_reshaped: {cropped_image_feature_reshaped.shape}")
        print(f"Shape of template_image_feature_reshaped: {template_image_feature_reshaped.shape}")

        patch_size = 60  # the resolution of the output image, set to 256 could be faster

        template_image_feature_upsampled = F.interpolate(template_image_feature_reshaped, size=(patch_size, patch_size),
                                                         mode='bilinear').squeeze()
        cropped_image_feature_upsampled = F.interpolate(cropped_image_feature_reshaped, size=(patch_size, patch_size),
                                                        mode='bilinear').squeeze()

        print(f"Shape of template_image_feature_upsampled: {template_image_feature_upsampled.shape}")
        print(f"Shape of cropped_image_feature_upsampled: {cropped_image_feature_upsampled.shape}")

        # Calculate the cosine similarity between src_feature and tgt_feature
        template_image_features_2d = template_image_feature_upsampled.reshape(template_image_feature_upsampled.shape[0],
                                                                              -1).permute(1, 0)
        cropped_image_features_2d = cropped_image_feature_upsampled.reshape(cropped_image_feature_upsampled.shape[0],
                                                                            -1).permute(1, 0)

        print(f"Shape of template_image_features_2d: {template_image_features_2d.shape}")
        print(f"Shape of cropped_image_features_2d: {cropped_image_features_2d.shape}")

        # Find the best correspondences
        best_matches = []
        for patch_idx in tqdm(range(patch_size * patch_size)):
            distances = torch.linalg.norm(cropped_image_features_2d - template_image_features_2d[patch_idx], dim=1)
            tgt_patch_idx = torch.argmin(distances)
            best_matches.append((patch_idx, tgt_patch_idx.item(), distances[tgt_patch_idx].item()))

        # Sort matches by distance (lowest to highest)
        best_matches.sort(key=lambda x: x[2])

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        for src_idx, tgt_idx, _ in best_matches[:num_pairs]:
            src_y = src_idx // patch_size
            src_x = src_idx % patch_size
            tgt_y = tgt_idx // patch_size
            tgt_x = tgt_idx % patch_size

            # Convert patch coordinates to pixel coordinates
            x1_show = (int(tgt_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(tgt_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(src_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(src_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2

            # Scale the points
            x1_scaled = int(x1_show * scale_factor)
            y1_scaled = int(y1_show * scale_factor)
            x2_scaled = int(x2_show * scale_factor)
            y2_scaled = int(y2_show * scale_factor)

            points1.append((x1_scaled, y1_scaled))
            points2.append((x2_scaled, y2_scaled))

        return points1, points2, cropped_image, template_image

    def find_correspondences_sd_dino_own6c(self, mask_cropped_image, mask_template, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        # points1 for cropped image -> target_image here
        # points2 for template image -> src_image here

        print(f"START: MASK_CROPPED_IMAGE {mask_cropped_image.shape}")
        print(f"START: MASK_TEMPLATE_IMAGE {mask_template.shape}")

        def visualize_resized_masks(resized_template_image_mask, resized_cropped_image_mask):
            """
            Visualize the resized masks for the template image and cropped image side by side.

            Args:
            resized_template_image_mask (torch.Tensor): The resized mask for the template image
            resized_cropped_image_mask (torch.Tensor): The resized mask for the cropped image
            """
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Display template image mask
            ax1.imshow(resized_template_image_mask.cpu().numpy(), cmap='gray')
            ax1.set_title(
                f"Template Image Mask - Size: {resized_template_image_mask.shape[0]}x{resized_template_image_mask.shape[1]}")
            ax1.axis('off')
            print(f"SIZE TEMPLATE IMAGE MASK: {resized_template_image_mask.shape[0]}x{resized_template_image_mask.shape[1]}")


            # Display cropped image mask
            ax2.imshow(resized_cropped_image_mask.cpu().numpy(), cmap='gray')
            ax2.set_title(
                f"Cropped Image Mask - Size: {resized_cropped_image_mask.shape[0]}x{resized_cropped_image_mask.shape[1]}")
            ax2.axis('off')
            print(f"SIZE CROPPED IMAGE MASK: {resized_cropped_image_mask.shape[0]}x{resized_cropped_image_mask.shape[1]}")
            plt.tight_layout()
            plt.show()

        mask_cropped_image = torch.from_numpy(mask_cropped_image).float() // 255
        mask_template = torch.from_numpy(mask_template).float()





        start_time_corr = time.time()
        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        src_feature_reshaped = cropped_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        tgt_feature_reshaped = template_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()

        patch_size = 60

        mask_cropped_image = mask_cropped_image // 255

        resized_src_mask = F.interpolate(mask_cropped_image.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        resized_tgt_mask = F.interpolate(mask_template.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()
        tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()
        # mask the feature
        src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0], 1, 1)
        tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0], 1, 1)
        # Set the masked area to a very small number
        src_feature_upsampled[src_feature_upsampled == 0] = -100000
        tgt_feature_upsampled[tgt_feature_upsampled == 0] = -100000
        # Calculate the cosine similarity between src_feature and tgt_feature
        src_features_2d = src_feature_upsampled.reshape(src_feature_upsampled.shape[0], -1).permute(1, 0)
        tgt_features_2d = tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0], -1).permute(1, 0)

        mapping = torch.zeros(patch_size, patch_size, 2).cuda()

        print(f"Min value in resized_src_mask: {resized_src_mask.min()}")
        print(f"Max value in resized_src_mask: {resized_src_mask.max()}")
        print(f"Unique values in resized_src_mask: {torch.unique(resized_src_mask)}")

        print(f"Min value in resized_tgt_mask: {resized_tgt_mask.min()}")
        print(f"Max value in resized_tgt_mask: {resized_tgt_mask.max()}")
        print(f"Unique values in resized_tgt_mask: {torch.unique(resized_tgt_mask)}")

        best_matches = []
        for patch_idx in tqdm(range(patch_size * patch_size)):
            if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
                tgt_patch_idx = torch.argmin(distances)

                src_patch_row = patch_idx // patch_size
                src_patch_col = patch_idx % patch_size
                tgt_patch_row = tgt_patch_idx // patch_size
                tgt_patch_col = tgt_patch_idx % patch_size

                mapping[src_patch_row, src_patch_col] = torch.tensor([tgt_patch_row, tgt_patch_col])

                best_matches.append((
                    (src_patch_row, src_patch_col),
                    (tgt_patch_row, tgt_patch_col),
                    distances[tgt_patch_idx].item()
                ))

        best_matches.sort(key=lambda x: x[2])

        points1 = []  # For template image
        points2 = []  # For cropped image

        for (src_row, src_col), (tgt_row, tgt_col), _ in best_matches[:num_pairs]:
            # Convert patch coordinates to pixel coordinates
            # Assuming self.stride and self.p are defined in your class
            x1 = tgt_col * self.stride[1] + self.stride[1] // 2
            y1 = tgt_row * self.stride[0] + self.stride[0] // 2
            x2 = src_col * self.stride[1] + self.stride[1] // 2
            y2 = src_row * self.stride[0] + self.stride[0] // 2

            # Scale the points
            x1_scaled = int(x1 * scale_factor)
            y1_scaled = int(y1 * scale_factor)
            x2_scaled = int(x2 * scale_factor)
            y2_scaled = int(y2 * scale_factor)

            points1.append((x1_scaled, y1_scaled))  # Template image points
            points2.append((x2_scaled, y2_scaled))  # Cropped image points

        print("Template image points (points1):")
        print(points1)
        print("Cropped image points (points2):")
        print(points2)

        return points1, points2, cropped_pil, template_pil


    def find_correspondences_sd_dino_own6(self, mask_cropped_image, mask_template, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        # points1 for cropped image -> target_image here
        # points2 for template image -> src_image here

        print(f"START: MASK_CROPPED_IMAGE {mask_cropped_image.shape}")
        print(f"START: MASK_TEMPLATE_IMAGE {mask_template.shape}")

        def visualize_resized_masks(resized_template_image_mask, resized_cropped_image_mask):
            """
            Visualize the resized masks for the template image and cropped image side by side.

            Args:
            resized_template_image_mask (torch.Tensor): The resized mask for the template image
            resized_cropped_image_mask (torch.Tensor): The resized mask for the cropped image
            """
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Display template image mask
            ax1.imshow(resized_template_image_mask.cpu().numpy(), cmap='gray')
            ax1.set_title(
                f"Template Image Mask - Size: {resized_template_image_mask.shape[0]}x{resized_template_image_mask.shape[1]}")
            ax1.axis('off')
            print(f"SIZE TEMPLATE IMAGE MASK: {resized_template_image_mask.shape[0]}x{resized_template_image_mask.shape[1]}")


            # Display cropped image mask
            ax2.imshow(resized_cropped_image_mask.cpu().numpy(), cmap='gray')
            ax2.set_title(
                f"Cropped Image Mask - Size: {resized_cropped_image_mask.shape[0]}x{resized_cropped_image_mask.shape[1]}")
            ax2.axis('off')
            print(f"SIZE CROPPED IMAGE MASK: {resized_cropped_image_mask.shape[0]}x{resized_cropped_image_mask.shape[1]}")
            plt.tight_layout()
            plt.show()

        mask_cropped_image = torch.from_numpy(mask_cropped_image).float()
        mask_template = torch.from_numpy(mask_template).float()

        start_time_corr = time.time()
        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

        cropped_image_feature_reshaped = cropped_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        template_image_feature_reshaped = template_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()

        print(f"Shape of cropped_image_feature_reshaped: {cropped_image_feature_reshaped.shape}")
        print(f"Shape of template_image_feature_reshaped: {template_image_feature_reshaped.shape}")

        patch_size = 256 # the resolution of the output image, set to 256 could be faster

        resized_cropped_image_mask = F.interpolate(mask_cropped_image.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        resized_template_image_mask = F.interpolate(mask_template.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        template_image_feature_upsampled = F.interpolate(template_image_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()
        cropped_image_feature_upsampled = F.interpolate(cropped_image_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()

        visualize_resized_masks(resized_template_image_mask, resized_cropped_image_mask)

        print(f"Shape of resized_cropped_image_mask: {resized_cropped_image_mask.shape}")
        print(f"Shape of resized_template_image_mask: {resized_template_image_mask.shape}")

        print(f"Shape of template_image_feature_upsampled: {template_image_feature_upsampled.shape}")
        print(f"Shape of cropped_image_feature_upsampled: {cropped_image_feature_upsampled.shape}")


        # mask the feature
        template_image_feature_upsampled = template_image_feature_upsampled * resized_template_image_mask.repeat(template_image_feature_upsampled.shape[0], 1, 1)
        cropped_image_feature_upsampled = cropped_image_feature_upsampled * resized_cropped_image_mask.repeat(template_image_feature_upsampled.shape[0], 1, 1)
        # Set the masked area to a very small number
        template_image_feature_upsampled[template_image_feature_upsampled == 0] = -100000
        cropped_image_feature_upsampled[cropped_image_feature_upsampled == 0] = -100000

        print(f"Shape of template_image_feature_upsampled: {template_image_feature_upsampled.shape}")
        print(f"Shape of cropped_image_feature_upsampled: {cropped_image_feature_upsampled.shape}")

        # Calculate the cosine similarity between src_feature and tgt_feature
        template_image_features_2d = template_image_feature_upsampled.reshape(template_image_feature_upsampled.shape[0], -1).permute(1, 0)
        cropped_image_features_2d = cropped_image_feature_upsampled.reshape(cropped_image_feature_upsampled.shape[0], -1).permute(1, 0)

        print(f"Shape of template_image_features_2d: {template_image_features_2d.shape}")
        print(f"Shape of cropped_image_features_2d: {cropped_image_features_2d.shape}")


        # Find the best correspondences
        best_matches = []
        for patch_idx in tqdm(range(patch_size * patch_size)):
            # use mask to search for features inside the mask area -> ==1 if part of mask
            if resized_template_image_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                distances = torch.linalg.norm(cropped_image_features_2d - template_image_features_2d[patch_idx], dim=1)
                tgt_patch_idx = torch.argmin(distances)
                best_matches.append((patch_idx, tgt_patch_idx.item(), distances[tgt_patch_idx].item()))

        # Sort matches by distance (lowest to highest)
        best_matches.sort(key=lambda x: x[2])

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        for src_idx, tgt_idx, _ in best_matches[:num_pairs]:
            src_y = src_idx // patch_size
            src_x = src_idx % patch_size
            tgt_y = tgt_idx // patch_size
            tgt_x = tgt_idx % patch_size

            # Convert patch coordinates to pixel coordinates
            x1_show = (int(tgt_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(tgt_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(src_x) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(src_y) - 1) * self.stride[0] + self.stride[0] + self.p // 2

            # Scale the points
            x1_scaled = int(x1_show * scale_factor)
            y1_scaled = int(y1_show * scale_factor)
            x2_scaled = int(x2_show * scale_factor)
            y2_scaled = int(y2_show * scale_factor)

            points1.append((x1_scaled, y1_scaled))
            points2.append((x2_scaled, y2_scaled))

        print("points1: ")
        print(points1)
        print("points2: ")
        print(points2)

        return points1, points2, cropped_pil, template_pil


    def find_correspondences_sd_dino_own5(self, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()
        with torch.no_grad():
            # Extract features for cropped image
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            # Extract features for template image
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

            # Get masks
            mask_cropped = get_mask(model_sd, aug_sd, cropped_image)
            mask_template = get_mask(model_sd, aug_sd, template_image)


        src_feature_reshaped = cropped_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        tgt_feature_reshaped = template_image_desc.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
        #src_img = Image.open(trg_img_path)
        #tgt_img = Image.open(src_img_path)

        patch_size = 256  # the resolution of the output image, set to 256 could be faster

        src_img = resize(cropped_image, patch_size, resize=True, to_pil=False, edge=False)
        tgt_img = resize(template_image, patch_size, resize=True, to_pil=False, edge=False)
        resized_src_mask = F.interpolate(mask_cropped.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        resized_tgt_mask = F.interpolate(mask_template.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                         mode='nearest').squeeze().cuda()
        src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()
        tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(patch_size, patch_size),
                                              mode='bilinear').squeeze()
        # mask the feature
        src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0], 1, 1)
        tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0], 1, 1)
        # Set the masked area to a very small number
        src_feature_upsampled[src_feature_upsampled == 0] = -100000
        tgt_feature_upsampled[tgt_feature_upsampled == 0] = -100000
        # Calculate the cosine similarity between src_feature and tgt_feature
        src_features_2d = src_feature_upsampled.reshape(src_feature_upsampled.shape[0], -1).permute(1, 0)
        tgt_features_2d = tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0], -1).permute(1, 0)
        swapped_image = src_img
        mapping = torch.zeros(patch_size, patch_size, 2).cuda()
        for patch_idx in tqdm(range(patch_size * patch_size)):
            # If the patch is in the resized_src_mask_out_layers, find the corresponding patch in the target_output
            if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                # Find the corresponding patch with the highest cosine similarity
                distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
                tgt_patch_idx = torch.argmin(distances)

                tgt_patch_row = tgt_patch_idx // patch_size
                tgt_patch_col = tgt_patch_idx % patch_size

        # Initialize lists to store correspondences
        points1 = []  # For cropped image
        points2 = []  # For template image

        # Find the best correspondences
        best_matches = []
        for patch_idx in tqdm(range(patch_size * patch_size)):
            if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
                tgt_patch_idx = torch.argmin(distances)
                best_matches.append((patch_idx, tgt_patch_idx.item(), distances[tgt_patch_idx].item()))

        # Sort matches by distance (lower is better) and select top num_pairs
        best_matches.sort(key=lambda x: x[2])
        selected_matches = best_matches[:num_pairs]

        # Convert patch indices to image coordinates
        for src_idx, tgt_idx, _ in selected_matches:
            # Source (cropped) image coordinates
            src_y, src_x = src_idx // patch_size, src_idx % patch_size
            # Target (template) image coordinates
            tgt_y, tgt_x = tgt_idx // patch_size, tgt_idx % patch_size

            # Convert to original image coordinates
            x1_show = int(src_x * self.stride[1] + self.stride[1] + self.p // 2)
            y1_show = int(src_y * self.stride[0] + self.stride[0] + self.p // 2)
            x2_show = int(tgt_x * self.stride[1] + self.stride[1] + self.p // 2)
            y2_show = int(tgt_y * self.stride[0] + self.stride[0] + self.p // 2)

            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

            # Scale the points
            points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
            points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

        # Reshape descriptors and masks
        cropped_desc = cropped_image_desc.squeeze().permute(1, 0).reshape(-1, num_patches, num_patches)
        template_desc = template_image_desc.squeeze().permute(1, 0).reshape(-1, num_patches, num_patches)

        # Resize masks to match descriptor size
        mask_cropped = F.interpolate(mask_cropped.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches),
                                     mode='nearest').squeeze()
        mask_template = F.interpolate(mask_template.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches),
                                      mode='nearest').squeeze()

        # Apply masks
        cropped_desc = cropped_desc * mask_cropped
        template_desc = template_desc * mask_template

        # Reshape to 2D for similarity computation
        cropped_desc_2d = cropped_desc.reshape(cropped_desc.shape[0], -1).t()
        template_desc_2d = template_desc.reshape(template_desc.shape[0], -1).t()

        # Normalize descriptors
        cropped_desc_2d = F.normalize(cropped_desc_2d, p=2, dim=1)
        template_desc_2d = F.normalize(template_desc_2d, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(cropped_desc_2d, template_desc_2d.t())

        # Find best matches in both directions
        best_template_match = similarity.max(dim=1)[1]
        best_cropped_match = similarity.max(dim=0)[1]

        # Find mutual best matches
        mutual_best = (best_template_match[best_cropped_match] == torch.arange(similarity.shape[1], device=self.device))
        mutual_best_indices = torch.where(mutual_best)[0]

        # Sort mutual best matches by similarity score
        mutual_best_scores = similarity[best_cropped_match[mutual_best_indices], mutual_best_indices]
        sorted_indices = torch.argsort(mutual_best_scores, descending=True)

        # Select top matches
        top_indices = sorted_indices[:min(num_pairs, len(sorted_indices))]

        # Convert to (y, x) coordinates
        points1 = []  # Cropped image points
        points2 = []  # Template image points
        for idx in top_indices:
            cropped_idx = best_cropped_match[mutual_best_indices[idx]]
            template_idx = mutual_best_indices[idx]

            y1, x1 = divmod(int(cropped_idx), num_patches)
            y2, x2 = divmod(int(template_idx), num_patches)

            # Convert to original image coordinates
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2

            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        # Scale the points
        points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
        points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]


        end_time_corr = time.time()
        print(f"Correspondence finding time: {end_time_corr - start_time_corr:.2f} seconds")

        return points1, points2, cropped_image, template_image

    def find_correspondences_sd_dino_own4(self, mask, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        result = []

        start_time_corr = time.time()
        with torch.no_grad():
            """ Descriptor 1: Cropped_Image """
            # Stable Diffusion
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            """ Descriptor 2: Template_Image """
            # Stable Diffusion
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

            mask_cropped_image = get_mask(model_sd, aug_sd, cropped_image)
            mask_template_image = get_mask(model_sd, aug_sd, template_image)

            # cropped_image_desc = cropped_image_desc.permute(0, 1, 3, 2).reshape(-1, cropped_image_desc.shape[-1], num_patches,
            #                                                           num_patches)
            # template_image_desc = template_image_desc.permute(0, 1, 3, 2).reshape(-1, template_image_desc.shape[-1], num_patches,
            #                                                           num_patches)

            result.append([cropped_image_desc, template_image_desc, mask_cropped_image, mask_template_image])

        for (feature2, feature1, mask2, mask1) in result:
            src_feature_reshaped = feature1.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
            tgt_feature_reshaped = feature2.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
            src_img = Image.open(template_image)
            tgt_img = Image.open(cropped_image)

            patch_size = 256  # the resolution of the output image, set to 256 could be faster

            src_img = resize(src_img, patch_size, resize=True, to_pil=False, edge=False)
            tgt_img = resize(tgt_img, patch_size, resize=True, to_pil=False, edge=False)
            resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                             mode='nearest').squeeze().cuda()
            resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
                                             mode='nearest').squeeze().cuda()
            src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size),
                                                  mode='bilinear').squeeze()
            tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(patch_size, patch_size),
                                                  mode='bilinear').squeeze()
            # mask the feature
            src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0], 1,
                                                                                    1)
            tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0], 1,
                                                                                    1)
            # Set the masked area to a very small number
            src_feature_upsampled[src_feature_upsampled == 0] = -100000
            tgt_feature_upsampled[tgt_feature_upsampled == 0] = -100000
            # Calculate the cosine similarity between src_feature and tgt_feature
            src_features_2d = src_feature_upsampled.reshape(src_feature_upsampled.shape[0], -1).permute(1, 0)
            tgt_features_2d = tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0], -1).permute(1, 0)
            swapped_image = src_img
            mapping = torch.zeros(patch_size, patch_size, 2).cuda()
            for patch_idx in tqdm(range(patch_size * patch_size)):
                # If the patch is in the resized_src_mask_out_layers, find the corresponding patch in the target_output a
                if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                    # Find the corresponding patch with the highest cosine similarity
                    distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
                    tgt_patch_idx = torch.argmin(distances)

                    tgt_patch_row = tgt_patch_idx // patch_size
                    tgt_patch_col = tgt_patch_idx % patch_size

                    # Swap the patches in output
                    #swapped_image[patch_idx // patch_size, patch_idx % patch_size, :] = tgt_img[tgt_patch_row,
                    #                                                                    tgt_patch_col, :]
                    #mapping[patch_idx // patch_size, patch_idx % patch_size] = torch.tensor(
                    #    [tgt_patch_row, tgt_patch_col])
            # save the swapped image

            points1, points2 = [], []
            for match in best_matches:

                x1_show = int(x1 * self.stride[1] + self.stride[1] + self.p // 2)
                y1_show = int(y1 * self.stride[0] + self.stride[0] + self.p // 2)
                x2_show = int(x2 * self.stride[1] + self.stride[1] + self.p // 2)
                y2_show = int(y2 * self.stride[0] + self.stride[0] + self.p // 2)

                points1.append((y1_show, x1_show))
                points2.append((y2_show, x2_show))

            # Scale the points
            points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
            points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]


        return points1, points2, cropped_image, template_image

    def find_correspondences_sd_dino_own3(self, mask, cropped_image, cropped_pil, template_image, template_pil,
                                          model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                          num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        def show_debug_images(trg_dense_output, src_color_map):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Display trg_dense_output
            ax1.imshow(trg_dense_output)
            ax1.set_title(f"Target Dense Output - Size: {trg_dense_output.shape[1]}x{trg_dense_output.shape[0]}")
            ax1.axis('off')

            # Display src_color_map
            ax2.imshow(src_color_map)
            ax2.set_title(f"Source Color Map - Size: {src_color_map.shape[1]}x{src_color_map.shape[0]}")
            ax2.axis('off')

            plt.tight_layout()
            plt.show()

        def perform_clustering(features, n_clusters=10):
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
            # Convert the features to float32
            features = features.cpu().detach().numpy().astype('float32')
            # Initialize a k-means clustering index with the desired number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            # Train the k-means index with the features
            kmeans.fit(features)
            # Assign the features to their nearest cluster
            labels = kmeans.predict(features)

            return labels

        result = []

        start_time_corr = time.time()
        with torch.no_grad():

            """ Descriptor 1: Cropped_Image """
            # Stable Diffusion
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            """ Descriptor 2: Template_Image """
            # Stable Diffusion
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

            mask_cropped_image = get_mask(model_sd, aug_sd, cropped_image)
            mask_template_image = get_mask(model_sd, aug_sd, template_image)

            #cropped_image_desc = cropped_image_desc.permute(0, 1, 3, 2).reshape(-1, cropped_image_desc.shape[-1], num_patches,
            #                                                           num_patches)
            #template_image_desc = template_image_desc.permute(0, 1, 3, 2).reshape(-1, template_image_desc.shape[-1], num_patches,
            #                                                           num_patches)

            result.append([cropped_image_desc, template_image_desc, mask_cropped_image, mask_template_image])

        points1 = []
        points2 = []
        n_clusters = 6

        for (feature1, feature2, mask1, mask2) in result:
            num_patches = int(math.sqrt(feature1.shape[-2]))
            feature1 = feature1.squeeze()
            feature2 = feature2.squeeze()

            src_feature_reshaped = feature1.permute(1, 0).reshape(-1, num_patches, num_patches).cuda()
            tgt_feature_reshaped = feature2.permute(1, 0).reshape(-1, num_patches, num_patches).cuda()
            resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches),
                                             mode='nearest').squeeze().cuda()
            resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches),
                                             mode='nearest').squeeze().cuda()
            src_feature_upsampled = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0], 1, 1)
            tgt_feature_upsampled = tgt_feature_reshaped * resized_tgt_mask.repeat(src_feature_reshaped.shape[0], 1, 1)

            feature1 = src_feature_upsampled.unsqueeze(0)
            feature2 = tgt_feature_upsampled.unsqueeze(0)

            w1, h1 = feature1.shape[2], feature1.shape[3]
            w2, h2 = feature2.shape[2], feature2.shape[3]

            features1_2d = feature1.reshape(feature1.shape[1], -1).permute(1, 0)
            features2_2d = feature2.reshape(feature2.shape[1], -1).permute(1, 0)

            labels_img1 = perform_clustering(features1_2d, n_clusters)
            labels_img2 = perform_clustering(features2_2d, n_clusters)

            cluster_means_img1 = [features1_2d.cpu().detach().numpy()[labels_img1 == i].mean(axis=0) for i in
                                  range(n_clusters)]
            cluster_means_img2 = [features2_2d.cpu().detach().numpy()[labels_img2 == i].mean(axis=0) for i in
                                  range(n_clusters)]

            distances = np.linalg.norm(
                np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img2, axis=0), axis=-1)
            row_ind, col_ind = linear_sum_assignment(distances)

            for i, j in zip(row_ind, col_ind):
                # Find centroid of cluster i in image 1
                y1, x1 = np.mean(np.where(labels_img1.reshape(w1, h1) == i), axis=1)
                # Find centroid of matched cluster j in image 2
                y2, x2 = np.mean(np.where(labels_img2.reshape(w2, h2) == j), axis=1)

                points1.append((int(y1), int(x1)))
                points2.append((int(y2), int(x2)))

            # Scale the points
            points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
            points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

        return points1, points2, cropped_image, template_image


    def find_correspondences_sd_dino_own2(self, mask, cropped_image, cropped_pil, template_image, template_pil,
                                         model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                         num_pairs: int = 20, layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        def show_debug_images(trg_dense_output, src_color_map):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Display trg_dense_output
            ax1.imshow(trg_dense_output)
            ax1.set_title(f"Target Dense Output - Size: {trg_dense_output.shape[1]}x{trg_dense_output.shape[0]}")
            ax1.axis('off')

            # Display src_color_map
            ax2.imshow(src_color_map)
            ax2.set_title(f"Source Color Map - Size: {src_color_map.shape[1]}x{src_color_map.shape[0]}")
            ax2.axis('off')

            plt.tight_layout()
            plt.show()

        start_time_corr = time.time()
        with torch.no_grad():

            """ Descriptor 1: Cropped_Image """
            # Stable Diffusion
            img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            cropped_image_desc = torch.cat((desc_sd1, desc_dino1), dim=-1)

            """ Descriptor 2: Template_Image """
            # Stable Diffusion
            img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            # DINOv2
            desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
            # Normalization and Fusion
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            template_image_desc = torch.cat((desc_sd2, desc_dino2), dim=-1)

            mask1 = get_mask(model_sd, aug_sd, cropped_image)
            mask2 = get_mask(model_sd, aug_sd, template_image)

            cropped_image_desc = cropped_image_desc.permute(0, 1, 3, 2).reshape(-1, cropped_image_desc.shape[-1], num_patches,
                                                                       num_patches)
            template_image_desc = template_image_desc.permute(0, 1, 3, 2).reshape(-1, template_image_desc.shape[-1], num_patches,
                                                                       num_patches)

            # Find nearest patches
            trg_dense_output, src_color_map = find_nearest_patchs(mask2, mask1, template_image, cropped_image,
                                                                       template_image_desc, cropped_image_desc,
                                                                       mask=True, resolution=156)

            show_debug_images(trg_dense_output, src_color_map)

            # Use the results of find_nearest_patchs to determine correspondences
            diff = np.abs(trg_dense_output - src_color_map)
            total_diff = np.sum(diff, axis=-1)
            flat_diff = total_diff.flatten()

            # Get indices of the best matches (lowest differences)
            best_matches = np.argsort(flat_diff)[:num_pairs]

            points1, points2 = [], []
            for match in best_matches:
                y1, x1 = np.unravel_index(match, total_diff.shape)

                # Find corresponding point in template image
                y2, x2 = np.unravel_index(np.argmin(np.sum((src_color_map - trg_dense_output[y1, x1]) ** 2, axis=-1)),
                                          src_color_map.shape[:2])

                x1_show = int(x1 * self.stride[1] + self.stride[1] + self.p // 2)
                y1_show = int(y1 * self.stride[0] + self.stride[0] + self.p // 2)
                x2_show = int(x2 * self.stride[1] + self.stride[1] + self.p // 2)
                y2_show = int(y2 * self.stride[0] + self.stride[0] + self.p // 2)

                points1.append((y1_show, x1_show))
                points2.append((y2_show, x2_show))

            # Scale the points
            points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
            points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

        end_time_corr = time.time()
        elapsed_corr = end_time_corr - start_time_corr


        return points1, points2, cropped_image, template_image

    def find_correspondences__sd_dino_own(self, input_image, input_pil, template_image, template_pil,
                                                   num_patches,
                                                   model_sd, aug_sd, image_size_sd, scale_factor, num_pairs: int = 10,
                                                   layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        """
        Integrating Cluster and match function from sd dino
        """

        def find_nearest_patchs(mask1, mask2, image1, image2, features1, features2, mask=False, resolution=None,
                                edit_image=None):
            def polar_color_map(image_shape):
                h, w = image_shape[:2]
                x = np.linspace(-1, 1, w)
                y = np.linspace(-1, 1, h)
                xx, yy = np.meshgrid(x, y)

                # Find the center of the mask
                mask = mask2.cpu()
                mask_center = np.array(np.where(mask > 0))
                mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
                mask_center_y, mask_center_x = mask_center

                # Calculate distance and angle based on mask_center
                xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
                max_radius = np.sqrt(h ** 2 + w ** 2) / 2
                radius = np.sqrt(xx_shifted ** 2 + yy_shifted ** 2) * max_radius
                angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

                angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
                radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
                radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

                return angle, radius

            if resolution is not None:  # resize the feature map to the resolution
                features1 = F.interpolate(features1, size=resolution, mode='bilinear')
                features2 = F.interpolate(features2, size=resolution, mode='bilinear')

            # resize the image to the shape of the feature map
            resized_image1 = resize(image1, features1.shape[2], resize=True, to_pil=False)
            resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)

            if mask:  # mask the features
                resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=features1.shape[2:],
                                              mode='nearest')
                resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:],
                                              mode='nearest')
                features1 = features1 * resized_mask1.repeat(1, features1.shape[1], 1, 1)
                features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
                # set where mask==0 a very large number
                features1[(features1.sum(1) == 0).repeat(1, features1.shape[1], 1, 1)] = 100000
                features2[(features2.sum(1) == 0).repeat(1, features2.shape[1], 1, 1)] = 100000

            features1_2d = features1.reshape(features1.shape[1], -1).permute(1, 0).cpu().detach().numpy()
            features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0).cpu().detach().numpy()

            features1_2d = torch.tensor(features1_2d).to("cuda")
            features2_2d = torch.tensor(features2_2d).to("cuda")
            resized_image1 = torch.tensor(resized_image1).to("cuda").float()
            resized_image2 = torch.tensor(resized_image2).to("cuda").float()

            mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2],
                                  mode='nearest').squeeze(0).squeeze(0)
            mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2],
                                  mode='nearest').squeeze(0).squeeze(0)

            # Mask the images
            resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
            resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
            # Normalize the images to the range [0, 1]
            resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
            resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

            angle, radius = polar_color_map(resized_image2.shape)

            angle_mask = angle * mask2.cpu().numpy()
            radius_mask = radius * mask2.cpu().numpy()

            hsv_mask = np.zeros(resized_image2.shape, dtype=np.float32)
            hsv_mask[:, :, 0] = angle_mask
            hsv_mask[:, :, 1] = radius_mask
            hsv_mask[:, :, 2] = 1

            rainbow_mask2 = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

            if edit_image is not None:
                rainbow_mask2 = cv2.imread(edit_image, cv2.IMREAD_COLOR)
                rainbow_mask2 = cv2.cvtColor(rainbow_mask2, cv2.COLOR_BGR2RGB) / 255
                rainbow_mask2 = cv2.resize(rainbow_mask2, (resized_image2.shape[1], resized_image2.shape[0]))

            # Apply the rainbow mask to image2
            rainbow_image2 = rainbow_mask2 * mask2.cpu().numpy()[:, :, None]

            # Create a white background image
            background_color = np.array([1, 1, 1], dtype=np.float32)
            background_image = np.ones(resized_image2.shape, dtype=np.float32) * background_color

            # Apply the rainbow mask to image2 only in the regions where mask2 is 1
            rainbow_image2 = np.where(mask2.cpu().numpy()[:, :, None] == 1, rainbow_mask2, background_image)

            nearest_patches = []

            distances = torch.cdist(features1_2d, features2_2d)
            nearest_patch_indices = torch.argmin(distances, dim=1)
            nearest_patches = torch.index_select(torch.tensor(rainbow_mask2).cuda().reshape(-1, 3), 0,
                                                 nearest_patch_indices)

            nearest_patches_image = nearest_patches.reshape(resized_image1.shape)
            rainbow_image2 = torch.tensor(rainbow_image2).to("cuda")

            # TODO: upsample the nearest_patches_image to the resolution of the original image
            # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
            # rainbow_image2 = F.interpolate(rainbow_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

            nearest_patches_image = (nearest_patches_image).cpu().numpy()
            resized_image2 = (rainbow_image2).cpu().numpy()

            return nearest_patches_image, resized_image2

        # Load image 1
        img1 = cropped_image
        img1_sd = resize(img1, image_size_sd, resize=True, to_pil=True, edge=False)
        img1_dino = resize(img1, image_size_dino, resize=True, to_pil=True)

        # Load image 2
        img2 = template_image
        img2_sd = resize(img2, image_size_sd, resize=True, to_pil=True, edge=False)
        img2 = resize(img2, image_size_dino, resize=True, to_pil=True, edge=False)

        with torch.no_grad():

            img1_desc_sd = process_features_and_mask(model_sd, aug_sd, img1_sd, input_text=None, mask=False,
                                                  pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3,
                                                                                                       2)
            img2_desc_sd = process_features_and_mask(model_sd, aug_sd, img2_sd, input_text=None,
                                                  mask=False, pca=True).reshape(1, 1, -1,
                                                                              num_patches ** 2).permute(0, 1, 3,
                                                                                                        2)

            img1_batch = self.preprocess_pil(img1)
            img1_desc_dino = self.extract_descriptors(img1_batch.to(self.device), layer, facet)
            img2_batch = self.preprocess_pil(img2)
            img2_desc_dino = self.extract_descriptors(img2_batch.to(self.device), layer, facet)


            img1_desc_sd = img1_desc_sd / img1_desc_sd.norm(dim=-1, keepdim=True)
            img2_desc_sd = img2_desc_sd / img2_desc_sd.norm(dim=-1, keepdim=True)

            img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
            img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)


            # cat two features together
            img1_desc = torch.cat((img1_desc_sd, img1_desc_dino), dim=-1)
            img2_desc = torch.cat((img2_desc_sd, img2_desc_dino), dim=-1)


            mask1 = get_mask(model_sd, aug_sd, img1)
            mask2 = get_mask(model_sd, aug_sd, img2)


            img1_desc_reshaped = img1_desc.permute(0, 1, 3, 2).reshape(-1, img1_desc.shape[-1], num_patches,
                                                                       num_patches)
            img2_desc_reshaped = img2_desc.permute(0, 1, 3, 2).reshape(-1, img2_desc.shape[-1], num_patches,
                                                                       num_patches)
            trg_dense_output, src_color_map = find_nearest_patchs(mask2, mask1, img2, img1, img2_desc_reshaped,
                                                                  img1_desc_reshaped, mask=True)



        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        # Scale the points
        points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
        points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]
        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    def find_correspondences_fastkmeans_sd_dino_v5(self, mask_crop, mask_template, input_image, input_pil, template_image, template_pil,
                                                   model_sd, aug_sd, image_size_sd, scale_factor, num_patches, num_pairs: int = 20,
                                                   layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()

        """ Descriptor 1: Cropped_Image """
        # Stable Diffusion
        img_sd1 = resize(input_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        print(f"Size of input_image after resizing: {img_sd1.size}")
        print(f"Size of input_pil: {input_pil.shape}")

        # DINOv2
        desc_dino1 = self.extract_descriptors(input_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
        num_patches1 = self.num_patches

        """ Descriptor 2: Template_Image """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

        print(f"Size of template_image after resizing: {img_sd2.size}")
        print(f"Size of template_pil: {template_pil.shape}")

        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)
        num_patches2 = self.num_patches

        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # Compute similarities between descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine


        start_time_bb = time.time()
        # Find mutual nearest neighbors (best buddies)
        sim_1, nn_1 = torch.max(similarities, dim=-1)
        sim_2, nn_2 = torch.max(similarities, dim=-2)
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = (nn_2[nn_1] == torch.arange(num_patches1[0] * num_patches1[1], device=self.device))


        # Select best buddy descriptors
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]

        # Concatenate best buddy descriptors for clustering
        all_keys_together = torch.cat((bb_descs1, bb_descs2), dim=1)
        n_clusters = min(num_pairs, len(all_keys_together))
        normalized = all_keys_together / torch.sqrt((all_keys_together ** 2).sum(dim=1, keepdim=True))

        # Perform k-means clustering on the concatenated descriptors
        start_time_kmeans = time.time()
        cluster_ids_x, _ = kmeans(X=normalized, num_clusters=n_clusters, distance='cosine',
                                  tqdm_flag=False, iter_limit=200, device=self.device)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        # Select the best correspondence from each cluster
        unique_labels, label_counts = torch.unique(cluster_ids_x, return_counts=True)
        indices_to_show = torch.cat([torch.where(cluster_ids_x == label)[0][:count]
                                     for label, count in zip(unique_labels, label_counts)])

        # Convert selected correspondences to image coordinates
        img1_indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[indices_to_show]
        img2_indices_to_show = nn_1[img1_indices_to_show]
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        # Scale the points

        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        elapsed_corr = end_time_corr - start_time_corr

        return points1, points2, input_pil, template_pil

    def find_correspondences_fastkmeans_sd_dino_v10(self, mask_crop, mask_template, cropped_image, cropped_pil,
                                                    template_image, template_pil,
                                                    model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                                    num_pairs: int = 20,
                                                    layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        """ Descriptor 1: Cropped_Image """
        # Stable Diffusion
        img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)

        """ Descriptor 2: Template_Image """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)

        # Determine patch size based on num_patches
        patch_size = num_patches  # Adjust this as needed; ensure it's correctly defined

        # Normalize the masks (assuming masks are initially in the range [0, 255])
        mask_crop = mask_crop / 255.0 if mask_crop.max() > 1 else mask_crop
        mask_template = mask_template / 255.0 if mask_template.max() > 1 else mask_template

        # Convert masks to torch tensors and ensure they are on the correct device
        mask_crop = torch.from_numpy(mask_crop).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_template = torch.from_numpy(mask_template).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Resize masks to match the patch size
        resized_src_mask = F.interpolate(mask_crop, size=(patch_size, patch_size), mode='nearest').squeeze().cuda()
        resized_tgt_mask = F.interpolate(mask_template, size=(patch_size, patch_size), mode='nearest').squeeze().cuda()

        # Resize the descriptors to match the patch size
        src_feature_upsampled = F.interpolate(descriptors1.reshape(1, -1, num_patches, num_patches),
                                              size=(patch_size, patch_size), mode='bilinear').squeeze()
        tgt_feature_upsampled = F.interpolate(descriptors2.reshape(1, -1, num_patches, num_patches),
                                              size=(patch_size, patch_size), mode='bilinear').squeeze()

        # Apply the resized masks to the upsampled features
        src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0], 1, 1)
        tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0], 1, 1)

        # Compute similarities between descriptors
        similarities = chunk_cosine_sim(src_feature_upsampled.unsqueeze(0), tgt_feature_upsampled.unsqueeze(0))

        # Weigh the similarities based on the masks
        similarities = similarities * (resized_src_mask * resized_tgt_mask).unsqueeze(0)

        # Find mutual nearest neighbors (best buddies)
        sim_1, nn_1 = torch.max(similarities, dim=-1)
        sim_2, nn_2 = torch.max(similarities, dim=-2)
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = (nn_2[nn_1] == torch.arange(patch_size * patch_size, device=self.device))

        # Apply masks to best buddies
        bbs_mask = bbs_mask & (resized_src_mask.view(-1) > 0) & (resized_tgt_mask.view(-1) > 0)

        # Select best buddy descriptors
        bb_descs1 = src_feature_upsampled.view(src_feature_upsampled.shape[0], -1)[:, bbs_mask]
        bb_descs2 = tgt_feature_upsampled.view(tgt_feature_upsampled.shape[0], -1)[:, nn_1[bbs_mask]]

        # Concatenate best buddy descriptors for clustering
        all_keys_together = torch.cat((bb_descs1, bb_descs2), dim=0).permute(1, 0)
        n_clusters = min(num_pairs, len(all_keys_together))
        normalized = all_keys_together / torch.sqrt((all_keys_together ** 2).sum(dim=1, keepdim=True))

        # Perform k-means clustering on the concatenated descriptors
        cluster_ids_x, _ = kmeans(X=normalized, num_clusters=n_clusters, distance='cosine',
                                  tqdm_flag=False, iter_limit=200, device=self.device)

        # Select the best correspondence from each cluster
        unique_labels, label_counts = torch.unique(cluster_ids_x, return_counts=True)
        indices_to_show = torch.cat([torch.where(cluster_ids_x == label)[0][:count]
                                     for label, count in zip(unique_labels, label_counts)])

        # Convert selected correspondences to image coordinates
        img1_indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[indices_to_show]
        img2_indices_to_show = nn_1[img1_indices_to_show]
        img1_y_to_show = (img1_indices_to_show / patch_size).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % patch_size).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / patch_size).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % patch_size).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
        points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

        return points1, points2, cropped_pil, template_pil


    def find_correspondences_fastkmeans_sd_dino_v11(self, mask_crop, mask_template, cropped_image, cropped_pil,
                                                    template_image, template_pil,
                                                    model_sd, aug_sd, image_size_sd, scale_factor, num_patches,
                                                    num_pairs: int = 20,
                                                    layer: int = 11, facet: str = 'token') -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        """ Descriptor 1: Cropped_Image """
        # Stable Diffusion
        img_sd1 = resize(cropped_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd1 = process_features_and_mask(model_sd, aug_sd, img_sd1, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino1 = self.extract_descriptors(cropped_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
        desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
        descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)

        """ Descriptor 2: Template_Image """
        # Stable Diffusion
        img_sd2 = resize(template_image, image_size_sd, resize=True, to_pil=True, edge=False)
        desc_sd2 = process_features_and_mask(model_sd, aug_sd, img_sd2, input_text=None, mask=False,
                                             pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        # DINOv2
        desc_dino2 = self.extract_descriptors(template_pil.to(self.device), layer, facet)
        # Normalization and Fusion
        desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)
        desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
        descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)

        crop_desc_reshaped = descriptors1.permute(0, 1, 3, 2).reshape(-1, descriptors1.shape[-1], num_patches, num_patches)
        template_desc_reshaped = descriptors2.permute(0, 1, 3, 2).reshape(-1, descriptors2.shape[-1], num_patches, num_patches)

        #print(f"Shape of crop_desc_reshaped: {crop_desc_reshaped.shape}")
        #print(f"Shape of template_desc_reshaped: {template_desc_reshaped.shape}")

        # Reshape descriptors for patch-wise comparison
        crop_features_2d = crop_desc_reshaped.reshape(crop_desc_reshaped.shape[1], -1).permute(1, 0)
        template_features_2d = template_desc_reshaped.reshape(template_desc_reshaped.shape[1], -1).permute(1, 0)

        #print(f"Shape of crop_features_2d: {crop_features_2d.shape}")
        #print(f"Shape of template_features_2d: {template_features_2d.shape}")

        # Find nearest patches
        distances = torch.cdist(crop_features_2d, template_features_2d)
        nearest_patch_indices = torch.argmin(distances, dim=1)

        # Convert linear indices to 2D coordinates
        img1_indices = torch.arange(crop_features_2d.shape[0])
        img1_y_to_show = (img1_indices / num_patches).cpu().numpy()
        img1_x_to_show = (img1_indices % num_patches).cpu().numpy()
        img2_y_to_show = (nearest_patch_indices / num_patches).cpu().numpy()
        img2_x_to_show = (nearest_patch_indices % num_patches).cpu().numpy()

        # Convert patch coordinates to pixel coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        # Apply scale factor
        points1 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points1]
        points2 = [(int(y * scale_factor), int(x * scale_factor)) for y, x in points2]

        print(points1)
        print(points2)

        return points1, points2, cropped_pil, template_pil






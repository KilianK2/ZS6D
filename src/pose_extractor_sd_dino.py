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
from external.sd_dino.utils.utils_correspondence import resize
import torch.nn.functional as F
#from ZS6D.src import extractor
#import src.extractor as extractor
from external.sd_dino.extractor_dino import ViTExtractor
from src.pose_extractor import PoseViTExtractor
import torch


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
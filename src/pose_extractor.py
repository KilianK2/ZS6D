import cv2
import torch
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
from zs6d_sd_dino.sd_dino.extractor_sd import process_features_and_mask, get_mask
from zs6d_sd_dino.sd_dino.utils.utils_correspondence import resize
import torch.nn.functional as F
#from ZS6D.src import extractor
#import src.extractor as extractor
from zs6d_sd_dino.sd_dino.extractor_dino import ViTExtractor

class PoseViTExtractor(ViTExtractor):

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        self.model_type = model_type
        self.stride = stride
        self.model = model
        self.device = device
        super().__init__(model_type = self.model_type, stride = self.stride, model=self.model, device=self.device)

        self.prep = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=self.mean, std=self.std)
                         ])
        
    


    def preprocess(self, img: Image.Image, 
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        
        scale_factor = 1
        
        if load_size is not None:
            width, height = img.size # img has to be quadratic

            img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)

            scale_factor = img.size[0] / width

        
        prep_img = self.prep(img)[None, ...]

        return prep_img, img, scale_factor
    
    # Overwrite functionality of _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]
    # to extract multiple facets and layers in one turn
    
    def _extract_multi_features(self, batch: torch.Tensor, layers: List[int] = [9,11], facet: str = 'key') -> List[torch.Tensor]:
        B, C, H, W = batch.shape
        self._feats = []
        # for (layer,fac) in zip(layers,facet):
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats
    
    def extract_multi_descriptors(self, batch: torch.Tensor, layers: List[int] = [9,11], facet: str = 'key',
                        bin: List[bool] = [True, False], include_cls: List[bool] = [False, False]) -> torch.Tensor:

        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                        choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_multi_features(batch, layers, facet)
        descs = []
        for i, x in enumerate(self._feats):
            if facet[i] == 'token':
                x.unsqueeze_(dim=1) #Bx1xtxd
            if not include_cls[i]:
                x = x[:, :, 1:, :]  # remove cls token
            else:
                assert not bin[i], "bin = True and include_cls = True are not supported together, set one of them False."
            if not bin:
                desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
            else:
                desc = self._log_bin(x)
            descs.append(desc)
        return descs
    
    
    def find_correspondences_fastkmeans(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        start_time_corr = time.time()
        
        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine
        
        
        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        # bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        # bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        # all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length
        
        start_time_kmeans = time.time()
        #'euclidean'
        # cluster_ids_x, cluster_centers = kmeans(X = normalized, num_clusters=n_clusters, distance='cosine', device=self.device)
        cluster_ids_x, cluster_centers = kmeans(X = normalized, 
                                        num_clusters=n_clusters, 
                                        distance='cosine',
                                        tqdm_flag = False,
                                        iter_limit=200, 
                                        device=self.device)
        
        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans
        
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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
        
        #print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil

    def find_correspondences_fastkmeans_sd_dino(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 840,
                                        layer: int = 11, facet: str = 'token', bin: bool = True,
                                        thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        # bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        # bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        # all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length

        start_time_kmeans = time.time()
        # 'euclidean'
        # cluster_ids_x, cluster_centers = kmeans(X = normalized, num_clusters=n_clusters, distance='cosine', device=self.device)
        cluster_ids_x, cluster_centers = kmeans(X=normalized,
                                                num_clusters=n_clusters,
                                                distance='cosine',
                                                tqdm_flag=False,
                                                iter_limit=200,
                                                device=self.device)

        kmeans_labels = cluster_ids_x.detach().cpu().numpy()

        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        # print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil

    def find_correspondences_fastkmeans_sd_dino_v2(self, model_sd, aug_sd, image_size_sd, image_size_dino, num_patches,
                                                pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 840,
                                                layer: int = 9, facet: str = 'key', bin: bool = True) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)

        """SD DINO 1 """
        img_base = pil_img1.convert('RGB')

        img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)
        img_dino = resize(img_base, image_size_dino, resize=True, to_pil=True, edge=False)

        desc_sd = process_features_and_mask(model_sd, aug_sd, img_sd, input_text=None, mask=False,
                                            pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of SD features: {desc_sd.shape}")

        # DinoV2
        img_dino_batch = self.preprocess_pil(img_dino)
        desc_dino = self.extract_descriptors(img_dino_batch.to(self.device), layer, facet)
        print(f"Shape of DINO features: {desc_dino.shape}")

        # normalization
        desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
        desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

        # fusion
        descriptors1 = torch.cat((desc_sd, desc_dino), dim=-1)

        num_patches1, load_size1 = self.num_patches, self.load_size

        """SD DINO 2 """
        img_base = pil_img2.convert('RGB')

        img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)
        img_dino = resize(img_base, image_size_dino, resize=True, to_pil=True, edge=False)

        desc_sd = process_features_and_mask(model_sd, aug_sd, img_sd, input_text=None, mask=False,
                                            pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of SD features: {desc_sd.shape}")

        # DinoV2
        img_dino_batch = self.preprocess_pil(img_dino)
        desc_dino = self.extract_descriptors(img_dino_batch.to(self.device), layer, facet)
        print(f"Shape of DINO features: {desc_dino.shape}")

        # normalization
        desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
        desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

        # fusion
        descriptors2 = torch.cat((desc_sd, desc_dino), dim=-1)

        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

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

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            kmeans_labels]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        return points1, points2, image1_pil, image2_pil

    def find_correspondences_fastkmeans_sd_dino_v3(self, model_sd, aug_sd, image_size_sd, image_size_dino, num_patches,
                                                   pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 840,
                                                   layer: int = 9, facet: str = 'key', bin: bool = True,
                                                   thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)

        """SD DINO 1 """
        img_base = pil_img1.convert('RGB')

        img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)
        img_dino = resize(img_base, image_size_dino, resize=True, to_pil=True, edge=False)

        desc_sd = process_features_and_mask(model_sd, aug_sd, img_sd, input_text=None, mask=False,
                                            pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of SD features: {desc_sd.shape}")

        # DinoV2
        img_dino_batch = self.preprocess_pil(img_dino)
        desc_dino = self.extract_descriptors(img_dino_batch.to(self.device), layer, facet)
        print(f"Shape of DINO features: {desc_dino.shape}")

        # normalization
        desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
        desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

        # fusion
        descriptors1 = torch.cat((desc_sd, desc_dino), dim=-1)

        num_patches1, load_size1 = self.num_patches, self.load_size

        """SD DINO 2 """
        img_base = pil_img2.convert('RGB')

        img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)
        img_dino = resize(img_base, image_size_dino, resize=True, to_pil=True, edge=False)

        desc_sd = process_features_and_mask(model_sd, aug_sd, img_sd, input_text=None, mask=False,
                                            pca=True).reshape(
            1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of SD features: {desc_sd.shape}")

        # DinoV2
        img_dino_batch = self.preprocess_pil(img_dino)
        desc_dino = self.extract_descriptors(img_dino_batch.to(self.device), layer, facet)
        print(f"Shape of DINO features: {desc_dino.shape}")

        # normalization
        desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
        desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

        # fusion
        descriptors2 = torch.cat((desc_sd, desc_dino), dim=-1)

        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

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

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        return points1, points2, image1_pil, image2_pil

    # correspondences without best buddy and without saliency maps
    def find_correspondences_fastkmeans_sd_dino_v4(self, model_sd, aug_sd, num_patches, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 840,
                                                layer: int = 11, facet: str = 'token', bin: bool = False) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        num_patches = int(840 // 14)

        # Fusing Stable Diffusion and DINOv2 descriptors
        with torch.no_grad():
            # Stable Diffusion
            img_sd1 = resize(image1_pil, self.image_size_sd, resize=True, to_pil=True, edge=False)
            img_sd2 = resize(image2_pil, self.image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(self.model_sd, self.aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(
                1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_sd2 = process_features_and_mask(self.model_sd, self.aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(
                1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

            # DINOv2
            desc_dino1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
            desc_dino2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)

            # Normalization
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)

            # Fusion
            descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
            descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)

        num_patches1, load_size1 = self.num_patches, self.load_size
        num_patches2, load_size2 = self.num_patches, self.load_size

        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

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

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        return points1, points2, image1_pil, image2_pil

    # fast kmeans with adjusted descriptors and using saliency maps from DINO
    def find_correspondences_fastkmeans_sd_dino_v5(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 840,
                                                layer: int = 11, facet: str = 'token', bin: bool = False,
                                                thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        num_patches = int(load_size // self.stride)

        # Fusing Stable Diffusion and DINOv2 descriptors
        with torch.no_grad():
            # Stable Diffusion
            img_sd1 = resize(image1_pil, self.image_size_sd, resize=True, to_pil=True, edge=False)
            img_sd2 = resize(image2_pil, self.image_size_sd, resize=True, to_pil=True, edge=False)
            desc_sd1 = process_features_and_mask(self.model_sd, self.aug_sd, img_sd1, input_text=None, mask=False,
                                                 pca=True).reshape(
                1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
            desc_sd2 = process_features_and_mask(self.model_sd, self.aug_sd, img_sd2, input_text=None, mask=False,
                                                 pca=True).reshape(
                1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

            # DINOv2
            desc_dino1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
            desc_dino2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)

            # Normalization
            desc_dino1 = desc_dino1 / desc_dino1.norm(dim=-1, keepdim=True)
            desc_dino2 = desc_dino2 / desc_dino2.norm(dim=-1, keepdim=True)
            desc_sd1 = desc_sd1 / desc_sd1.norm(dim=-1, keepdim=True)
            desc_sd2 = desc_sd2 / desc_sd2.norm(dim=-1, keepdim=True)

            # Fusion
            descriptors1 = torch.cat((desc_sd1, desc_dino1), dim=-1)
            descriptors2 = torch.cat((desc_sd2, desc_dino2), dim=-1)

        num_patches1, load_size1 = self.num_patches, self.load_size
        num_patches2, load_size2 = self.num_patches, self.load_size

        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

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

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        return points1, points2, image1_pil, image2_pil


    # SD DINO Features with segmentation mask for both inputs, mask is matched to spatial dimensions, best buddies mask only in the region of segmentation mask
    def find_correspondences_fastkmeans_sd_dino_v6(self, image_size_sd, model_sd, aug_sd, num_patches, input_image, input_pil, template_image, template_pil, num_pairs: int = 10, load_size: int = 840,
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
        #input_np = cv2.cvtColor(np.array(input_pil), cv2.COLOR_RGB2BGR)
        #template_np = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2BGR)

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

    # SD Dino without mask
    def find_correspondences_fastkmeans_sd_dino_v7(self, image_size_sd, model_sd, aug_sd, num_patches, input_image, input_pil, template_image, template_pil, num_pairs: int = 10, load_size: int = 840,
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

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1 * num_patches1, device=self.device)

        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

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

    def find_correspondences_fastkmeans_sd_dino_v8(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224,
                                                layer: int = 9, facet: str = 'key', bin: bool = True,
                                                thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1_dino = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        descriptors1_sd = process_features_and_mask(self.model_sd, self.aug_sd, pil_img1, mask=False, pca=True).reshape(
            1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)
        descriptors1 = torch.cat((descriptors1_sd, descriptors1_dino), dim=-1)
        num_patches1, load_size1 = self.num_patches, self.load_size

        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2_dino = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        descriptors2_sd = process_features_and_mask(self.model_sd, self.aug_sd, pil_img2, mask=False, pca=True).reshape(
            1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)
        descriptors2 = torch.cat((descriptors2_sd, descriptors2_dino), dim=-1)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        # Generate segmentation masks for both images
        mask1 = get_mask(self.model_sd, self.aug_sd, pil_img1)
        mask2 = get_mask(self.model_sd, self.aug_sd, pil_img2)

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
        image_idxs = torch.arange(num_patches1 * num_patches1, device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by segmentation mask.
        bbs_mask = torch.bitwise_and(bbs_mask, flat_mask1)
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

        return points1, points2, image1_pil, image2_pil

    def find_correspondences(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        start_time_corr = time.time()
        
        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine
        
        
        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        # apply k-means on a concatenation of a pairs descriptors.
        all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
        normalized = all_keys_together / length
        
        start_time_kmeans = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans
        
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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
        
        print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil
    
    
    def find_correspondences_old(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size

        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        similarities = chunk_cosine_sim(descriptors1, descriptors2)

        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        # apply k-means on a concatenation of a pairs descriptors.
        all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
        normalized = all_keys_together / length
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
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

        return points1, points2, image1_pil, image2_pil





    

    


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


class PoseViTExtractorSdDino(PoseViTExtractor):

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        self.model_type = model_type
        self.stride = stride
        self.model = model
        self.device = device
        super().__init__(model_type = self.model_type, stride = self.stride, model=self.model, device=self.device)

    def find_correspondences_nearest_neighbors(self, model):
        pass
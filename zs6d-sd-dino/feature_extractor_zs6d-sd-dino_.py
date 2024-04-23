import torch
from sd-dino.extractor_sd import load_model as load_sd_model, process_features_and_mask
from extractor_dino import ViTExtractor


class FusedFeatureExtractor:
    def __init__(self, sd_model_version, sd_image_size, num_timesteps, dino_model_type, stride, device='cuda'):
        self.device = device
        self.sd_model, self.sd_aug = load_sd_model(diffusion_ver=sd_model_version, image_size=sd_image_size,
                                                   num_timesteps=num_timesteps)
        self.dino_extractor = ViTExtractor(dino_model_type, stride, device=self.device)
        self.sd_model.to(self.device)

    def process_image(self, img_path):
        """Process an image and return the PIL image and torch tensor."""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.sd_aug(img).unsqueeze(0).to(self.device)
        return img, img_tensor

    def extract_features(self, img_tensor, layer, facet, use_dino=True):
        """Extract features using SD and optionally fuse with DINO."""
        # Extract features from SD model
        sd_features = process_features_and_mask(self.sd_model, self.sd_aug, img_tensor, mask=False, raw=True)

        if use_dino:
            # Extract features from DINO model
            dino_features = self.dino_extractor.extract_descriptors(img_tensor, layer, facet)
            # Normalize and fuse features
            sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
            dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)
            # Concatenate features along the feature dimension
            fused_features = torch.cat((sd_features, dino_features), dim=-1)
            return fused_features

        return sd_features

    def visualize_features(self, features):
        """A simple visualization of features."""
        plt.figure(figsize=(10, 5))
        plt.imshow(features.cpu().detach().numpy()[0, 0], aspect='auto')
        plt.colorbar()
        plt.title('Feature Visualization')
        plt.show()


# Example usage:
# Create an instance of the fused feature extractor
extractor = FusedFeatureExtractor(sd_model_version='v1-5', sd_image_size=224, num_timesteps=100,
                                  dino_model_type='dinov2_vitb16', stride=16)

# Process an image and extract fused features
img, img_tensor = extractor.process_image('path_to_image.jpg')
fused_features = extractor.extract_features(img_tensor, layer=11, facet='token', use_dino=True)

# Visualize the fused features
extractor.visualize_features(fused_features)

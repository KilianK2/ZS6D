import torch
from zs6d_sd_dino.sd_dino.extractor_sd import process_features_and_mask
from zs6d_sd_dino.sd_dino.extractor_dino import ViTExtractor



def get_fused_features(extractor_sd_dino, model, aug, img_prep_dino, img_prep_sd):
    PCA = True
    img_size = 224  # # used to be 840 # if DINOV2 else 244
    model_type = 'dinov2_vitb14'
    layer = 11
    facet = 'token'
    stride = 14
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor_sd_dino = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor_sd_dino.model.patch_embed.patch_size[0]
    num_patches = int(patch_size / stride * (img_size // patch_size))

    input_text = None

    with torch.no_grad():
        img1_desc = process_features_and_mask(model, aug, img_prep_sd, input_text=input_text, mask=False,
                                              pca=PCA).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of img1_desc (SD) features: {img1_desc.shape}")



        img1_desc_dino = extractor_sd_dino.extract_descriptors(img_prep_dino.to(device), layer, facet)
        print(f"Shape of img1_desc_dino: {img1_desc_dino.shape}")

        img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
        print(f"Shape of img1_desc (SD) normalized: {img1_desc.shape}")

        img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
        print(f"Shape of img1_desc_dino normalized: {img1_desc_dino.shape}")

        img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
        print(f"Shape of img1_desc (Fused) after fusion: {img1_desc.shape}")

    return img1_desc.cpu()
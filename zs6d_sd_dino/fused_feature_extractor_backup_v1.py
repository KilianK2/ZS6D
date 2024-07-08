import torch
from zs6d_sd_dino.sd_dino.extractor_sd import process_features_and_mask
from zs6d_sd_dino.sd_dino.extractor_dino import ViTExtractor
from zs6d_sd_dino.sd_dino.utils.utils_correspondence import resize

def get_fused_features( model, aug, img):
    PCA = True
    EDGE_PAD = False
    real_size = 480  # used to be 960
    img_size = 420  # # used to be 840 # if DINOV2 else 244
    model_type = 'dinov2_vitb14'
    layer = 11
    facet = 'token'
    stride = 14
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor_sd_dino = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor_sd_dino.model.patch_embed.patch_size[0]
    # num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    num_patches = int(patch_size / stride * (img_size // patch_size))

    input_text = None

    img_resized = resize(img, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img1 = resize(img_resized, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    with torch.no_grad():
        img1_desc = process_features_and_mask(model, aug, img_resized, input_text=input_text, mask=False,
                                              pca=PCA).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
        print(f"Shape of img1_desc (SD) features: {img1_desc.shape}")
        img1_batch = extractor_sd_dino.preprocess_pil(img1)
        print(f"Shape of img1_batch: {img1_batch.shape}")
        img1_desc_dino = extractor_sd_dino.extract_descriptors(img1_batch.to(device), layer, facet)
        print(f"Shape of img1_desc_dino: {img1_desc_dino.shape}")

        img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
        print(f"Shape of img1_desc (SD) normalized: {img1_desc.shape}")

        img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
        print(f"Shape of img1_desc_dino normalized: {img1_desc_dino.shape}")

        img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)

        print(f"Shape of img1_desc (Fused) after fusion: {img1_desc.shape}")

    return img1_desc.cpu()
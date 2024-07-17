import argparse
import os
import json
import numpy as np
import torch
from src.pose_extractor_sd_dino import PoseViTExtractorSdDino
from tools.ply_file_to_3d_coord_model import convert_unique
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from tqdm import tqdm
import cv2
from PIL import Image
from pose_utils import img_utils
from rendering.utils import get_rendering, get_sympose
from external.sd_dino.extractor_sd import load_model
from external.sd_dino.extractor_sd import process_features_and_mask
from external.sd_dino.utils.utils_correspondence import resize



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file',
                        default="./zs6d_configs/template_gt_preparation_configs/cfg_template_gt_generation_ycbv_sd_dino.json")

    args = parser.parse_args()


    """Adjust path to local structure"""
    #sys.path.append("/run/media/kilian/Extern_SSD/Robot_Vision/ZS6D/zs6d_sd_dino/sd_dino/third_party/ODISE")

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)

    with open(os.path.join(config['path_models_info_json']), 'r') as f:
        models_info = json.load(f)

    obj_poses = np.load(config['path_template_poses'])

    # Creating the output folder for the cropped templates and descriptors
    if not os.path.exists(os.path.join(config['path_output_templates_and_descs_folder'])):
        os.makedirs(os.path.join(config['path_output_templates_and_descs_folder']))

    # Creating the models_xyz folder
    if not os.path.exists(config['path_output_models_xyz']):
        os.makedirs(config['path_output_models_xyz'])

    # Preparing the object models in xyz format:
    print("Loading and preparing the object meshes:")
    norm_factors = {}
    for obj_model_name in tqdm(os.listdir(config['path_object_models_folder'])):
        if obj_model_name.endswith(".ply"):
            obj_id = int(obj_model_name.split("_")[-1].split(".ply")[0])
            input_model_path = os.path.join(config['path_object_models_folder'], obj_model_name)
            output_model_path = os.path.join(config['path_output_models_xyz'], obj_model_name)
            # if not os.path.exists(output_model_path):
            x_abs, y_abs, z_abs, x_ct, y_ct, z_ct = convert_unique(input_model_path, output_model_path)

            norm_factors[obj_id] = {'x_scale': float(x_abs),
                                    'y_scale': float(y_abs),
                                    'z_scale': float(z_abs),
                                    'x_ct': float(x_ct),
                                    'y_ct': float(y_ct),
                                    'z_ct': float(z_ct)}

    with open(os.path.join(config['path_output_models_xyz'], "norm_factor.json"), "w") as f:
        json.dump(norm_factors, f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """Setup for Fused_ZS6D"""
    stride = 14
    extractor = PoseViTExtractorSdDino(model_type='dinov2_vitb14', stride=stride, device=device)
    #extractor_dino = ViTExtractor(model_type='dinov2_vitb14', stride=stride, device=device)
    image_size_sd = 960
    image_size_dino = 840
    layer = 11
    facet = 'token'
    model, aug = load_model(diffusion_ver="v1-5", image_size=image_size_sd, num_timesteps=100)
    patch_size = extractor.model.patch_embed.patch_size[0]
    #img_size = extractor.model.patch_embed.img_size
    #num_patches = int(patch_size / stride * (image_size_dino // patch_size - 1) + 1)
    num_patches = int(patch_size / stride * (image_size_dino // patch_size))


    #num_patches_sd = int(patch_size / stride * (image_size_sd // patch_size - 1) + 1)


    cam_K = np.array(config['cam_K']).reshape((3, 3))

    ren = Renderer((config['template_resolution'][0], config['template_resolution'][1]), cam_K)

    template_labels_gt = dict()



    with torch.no_grad():

        for template_name in tqdm(os.listdir(config['path_templates_folder'])):

            path_template_folder = os.path.join(config['path_templates_folder'], template_name)

            if os.path.isdir(path_template_folder) and template_name != "models" and template_name != "models_proc":

                path_to_template_desc = os.path.join(config['path_output_templates_and_descs_folder'],
                                                     template_name)

                if not os.path.exists(path_to_template_desc):
                    os.makedirs(path_to_template_desc)

                obj_id = template_name.split("_")[-1]

                model_info = models_info[str(obj_id)]

                obj_model = Model3D()
                model_path = os.path.join(config['path_output_models_xyz'], f"obj_{int(obj_id):06d}.ply")

                # Some objects are scaled inconsistently within the dataset, these exceptions are handled here:
                obj_scale = config['obj_models_scale']
                obj_model.load(model_path, scale=obj_scale)

                files = os.listdir(path_template_folder)
                filtered_files = list(filter(lambda x: not x.startswith('mask_'), files))
                filtered_files.sort(key=lambda x: os.path.getmtime(os.path.join(path_template_folder, x)))

                tmp_list = []

                for i, file in enumerate(filtered_files):

                    # Preparing mask and bounding box [x,y,w,h]
                    mask_path = os.path.join(path_template_folder, f"mask_{file}")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    crop_size = max(w, h)

                    # Preparing cropped image and desc
                    img = cv2.imread(os.path.join(path_template_folder, file))

                    img_base = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img_crop_raw, crop_x, crop_y = img_utils.make_quadratic_crop(img_base, [x, y, w, h])
                    img_prep, img_crop, _ = extractor.preprocess(Image.fromarray(img_crop_raw), load_size=image_size_dino)

                    """SD-DINO"""

                    img_base = Image.fromarray(img_crop_raw).convert('RGB')

                    # Resizing
                    img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)


                    # Stable Diffusion
                    desc_sd = process_features_and_mask(model, aug, img_sd, input_text=None, mask=False, pca=True).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                    print(f"Shape of SD features: {desc_sd.shape}")

                    # DinoV2
                    desc_dino = extractor.extract_descriptors(img_prep.to(device), layer, facet)
                    print(f"Shape of DINO features: {desc_dino.shape}")

                    # normalization
                    desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
                    desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                    # fusion
                    desc_sd_dino = torch.cat((desc_sd, desc_dino), dim=-1)
                    print(f"Shape of SD-DINO features: {desc_sd_dino.shape}")


                    desc_sd_dino = desc_sd_dino.squeeze(0).squeeze(0).detach().cpu().numpy()

                    R = obj_poses[i][:3, :3]
                    t = obj_poses[i].T[-1, :3]
                    sym_continues = [0, 0, 0, 0, 0, 0]
                    keys = model_info.keys()

                    if ('symmetries_continuous' in keys):
                        sym_continues[:3] = model_info['symmetries_continuous'][0]['axis']
                        sym_continues[3:] = model_info['symmetries_continuous'][0]['offset']

                    rot_pose, rotation_lock = get_sympose(R, sym_continues)

                    img_uv, depth_rend, bbox_template = get_rendering(obj_model, rot_pose, t / 1000., ren)

                    img_uv = img_uv.astype(np.uint8)

                    img_uv, _, _ = img_utils.make_quadratic_crop(img_uv, [crop_y, crop_x, crop_size, crop_size])

                    # Storing template information:
                    tmp_dict = {"img_id": str(i),
                                "img_name": os.path.join(os.path.join(path_template_folder, file)),
                                "mask_name": os.path.join(os.path.join(path_template_folder, f"mask_{file}")),
                                "obj_id": str(obj_id),
                                "bbox_obj": [x, y, w, h],
                                "cam_R_m2c": R.tolist(),
                                "cam_t_m2c": t.tolist(),
                                "model_path": os.path.join(config['path_object_models_folder'],
                                                           f"obj_{int(obj_id):06d}.ply"),
                                "model_info": models_info[str(obj_id)],
                                "cam_K": cam_K.tolist(),
                                "img_crop": os.path.join(path_to_template_desc, file),
                                "img_desc": os.path.join(path_to_template_desc, f"{file.split('.')[0]}.npy"),
                                "uv_crop": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_uv.npy"),

                                }

                    tmp_list.append(tmp_dict)

                    # Saving all template crops and descriptors:
                    np.save(tmp_dict['uv_crop'], img_uv)
                    np.save(tmp_dict['img_desc'], desc_sd_dino)
                    img_crop.save(tmp_dict['img_crop'])

                template_labels_gt[str(obj_id)] = tmp_list

    with open(config['output_template_gt_file'], 'w') as f:
        json.dump(template_labels_gt, f)
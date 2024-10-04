import argparse
import json
import os
import torch
from tqdm import tqdm
import numpy as np
from src.pose_extractor_sd_dino import PoseViTExtractorSdDino
from pose_utils.data_utils import ImageContainer_masks
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import time
import pose_utils.eval_utils as eval_utils
import csv
import logging
from external.sd_dino.utils.utils_correspondence import resize
from external.sd_dino.extractor_sd import load_model
from external.sd_dino.extractor_sd import process_features_and_mask

# Setup logging
logging.basicConfig(level=logging.INFO, filename="pose_estimation.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    # adjust for ycbv
    parser.add_argument('--config_file', default="./zs6d_configs/bop_eval_configs/cfg_ycbv_inference_bop_sd_dino.json")

    args = parser.parse_args()

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)

    # Loading ground truth files:

    with open(os.path.join(config['templates_gt_path']), 'r') as f:
        templates_gt = json.load(f)

    with open(os.path.join(config['gt_path']), 'r') as f:
        data_gt = json.load(f)

    with open(os.path.join(config['norm_factor_path']), 'r') as f:
        norm_factors = json.load(f)

    # Set up a results csv file:
    csv_file = os.path.join('results', config['results_file'])

    # Column names for the CSV file
    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']

    # Create a new CSV file and write the headers
    with open(csv_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

    if config['debug_imgs']:
        debug_img_path = os.path.join("debug_imgs", config['results_file'].split(".csv")[0])
        if not os.path.exists(debug_img_path):
            os.makedirs(debug_img_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """Setup for Fused_ZS6D"""
    stride = 14

    extractor = PoseViTExtractorSdDino(model_type='dinov2_vitb14', stride=stride, device=device)
    print("Loading PoseViTExtractor is done!")

    image_size_sd = 960
    image_size_dino = 840
    layer = 11
    facet = 'token'
    model_sd, aug_sd = load_model(diffusion_ver="v1-5", image_size=image_size_sd, num_timesteps=100)
    patch_size = extractor.model.patch_embed.patch_size[0]
    num_patches = int(patch_size / stride * (image_size_dino // patch_size))

    # Loading templates into gpu
    templates_desc = {}
    templates_crops = {}
    tmpdic_per_obj = {}
    templates_gt_new = {}
    for obj_id, template_labels in tqdm(templates_gt.items()):
        try:
            templates_desc[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                                                for i, template_label in enumerate(template_labels) if
                                                i % config['template_subset'] == 0], dim=0)

            templates_gt_new[obj_id] = [template_label for i, template_label in enumerate(template_labels) if
                                        i % config['template_subset'] == 0]
        except Exception as e:
            logger.error(f"Error processing templates for object {obj_id}: {e}")

    print("Preparing templates finished!")

    print("Processing input images:")

    for all_id, img_labels in tqdm(data_gt.items()):
        scene_id = all_id.split("_")[0]
        img_id = all_id.split("_")[-1]

        # get data and crops for a single image

        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]

        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3, 3))



        img_data = ImageContainer_masks(img=img,
                                        img_name=img_name,
                                        scene_id=scene_id,
                                        cam_K=cam_K,
                                        crops=[],
                                        descs=[],
                                        x_offsets=[],
                                        y_offsets=[],
                                        obj_names=[],
                                        obj_ids=[],
                                        model_infos=[],
                                        t_gts=[],
                                        R_gts=[],
                                        masks=[])

        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]

            if bbox_gt[2] == 0 or bbox_gt[3] == 0:
                continue


            if bbox_gt != [-1, -1, -1, -1]:

                img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
                img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3, 3)))
                img_data.obj_ids.append(str(img_label['obj_id']))
                img_data.model_infos.append(img_label['model_info'])

                try:
                    # if True:
                    mask = img_utils.rle_to_mask(img_label['mask_sam'])

                    mask = mask.astype(np.uint8)

                    mask_3_channel = np.stack([mask] * 3, axis=-1)

                    bbox = img_utils.get_bounding_box_from_mask(mask)

                    img_crop_raw, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)

                    mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)


                    img_crop = cv2.bitwise_and(img_crop_raw, img_crop_raw, mask=mask_crop)



                    img_data.crops.append(Image.fromarray(img_crop))

                    img_prep, img_crop, _ = extractor.preprocess(Image.fromarray(img_crop), load_size=image_size_dino)

                    with torch.no_grad():

                        """SD-DINO"""

                        img_base = Image.fromarray(img_crop_raw).convert('RGB')
                        # Resizing
                        img_sd = resize(img_base, image_size_sd, resize=True, to_pil=True, edge=False)
                        # Stable Diffusion
                        desc_sd = process_features_and_mask(model_sd, aug_sd, img_sd, input_text=None, mask=False,
                                                            pca=True).reshape(1, 1, -1, num_patches ** 2).permute(0, 1,
                                                                                                                  3, 2)
                        print(f"Shape of SD features: {desc_sd.shape}")

                        # DinoV2
                        #img_dino_batch = extractor.preprocess_pil(img_dino)
                        desc_dino = extractor.extract_descriptors(img_prep.to(device), layer, facet)
                        print(f"Shape of DINO features: {desc_dino.shape}")

                        # normalization
                        desc_dino = desc_dino / desc_dino.norm(dim=-1, keepdim=True)
                        desc_sd = desc_sd / desc_sd.norm(dim=-1, keepdim=True)

                        desc_sd_dino = torch.cat((desc_sd, desc_dino), dim=-1)
                        print(f"Shape of SD-DINO features: {desc_sd_dino.shape}")

                        img_data.descs.append(desc_sd_dino.squeeze(0).squeeze(0).detach().cpu())

                    img_data.y_offsets.append(y_offset)
                    img_data.x_offsets.append(x_offset)
                    img_data.masks.append(mask_3_channel)

                except Exception as e:
                    logger.warning(
                        f"Loading mask and extracting descriptor failed for img {img_path} and object_id {obj_index}: {e}")
                    img_data.crops.append(None)
                    img_data.descs.append(None)
                    img_data.y_offsets.append(None)
                    img_data.x_offsets.append(None)
                    img_data.masks.append(None)

        for i in range(len(img_data.crops)):
            start_time = time.time()
            object_id = img_data.obj_ids[i]
            if img_data.crops[i] is not None:
                try:


                    matched_templates = utils.find_template_cpu(img_data.descs[i],
                                                                templates_desc[object_id],
                                                                num_results=config['num_matched_templates'])
                except Exception as e:

                    logger.error(
                        f"Template matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                min_err = np.inf
                pose_est = False
                for matched_template in matched_templates:

                    template = Image.open(templates_gt_new[object_id][matched_template[1]]['img_crop'])

                    try:
                        with torch.no_grad():
                            cropped_image = img_data.crops[i]
                            cropped_pil, _, _ = extractor.preprocess(cropped_image, load_size=image_size_dino)
                            template_image = template
                            template_pil, _, _ = extractor.preprocess(template_image, load_size=image_size_dino)

                            crop_size = img_data.crops[i].size[0]
                            scale_factor = crop_size / image_size_dino
                            #points1, points2, crop_pil, template_pil = extractor.find_correspondences_sd_dino6a(
                            #    cropped_image, cropped_pil, template_image, template_pil, model_sd, aug_sd,
                            #    image_size_sd, scale_factor, num_patches, num_pairs=20)
                            mask_template ="empty"
                            points1, points2, crop_pil, template_pil = extractor.find_correspondences_kmeans_sd_dino_v13(mask_crop, mask_template, cropped_image, cropped_pil,
                                                    template_image, template_pil,
                                                    model_sd, aug_sd, image_size_sd, scale_factor, num_patches)

                            #points1, points2, crop_pil, template_pil = extractor.find_correspondences_fastkmeans_sd_dino_v5(
                            #    input_image, input_pil, template_image, template_pil, num_patches, model_sd,
                            #    aug_sd, image_size_sd, scale_factor=scale_factor,
                            #    num_pairs=20)

                    except Exception as e:
                        logging.error(
                            f"Local correspondence matching failed for {img_data.img_name} and object_id {img_data.obj_ids[i]}: {e}")

                    try:
                        img_uv = np.load(templates_gt_new[object_id][matched_template[1]]['uv_crop'])

                        img_uv = img_uv.astype(np.uint8)

                        img_uv = cv2.resize(img_uv, img_data.crops[i].size)

                        R_est, t_est = utils.get_pose_from_correspondences(points1,
                                                                           points2,
                                                                           img_data.y_offsets[i],
                                                                           img_data.x_offsets[i],
                                                                           img_uv,
                                                                           img_data.cam_K,
                                                                           norm_factors[str(img_data.obj_ids[i])],
                                                                           config['scale_factor'])
                    except Exception as e:
                        logger.error(
                            f"Not enough correspondences could be extracted for {img_data.img_name} and object_id {object_id}: {e}")
                        R_est = None

                    if R_est is None:
                        R_est = np.array(templates_gt_new[object_id][matched_template[1]]['cam_R_m2c']).reshape((3, 3))
                        t_est = np.array([0., 0., 0.])

                    end_time = time.time()
                    err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)

                    if err < min_err:
                        min_err = err
                        R_best = R_est
                        t_best = t_est
                        pose_est = True

                if not pose_est:
                    R_best = np.array([[1.0, 0., 0.],
                                       [0., 1.0, 0.],
                                       [0., 0., 1.0]])

                    t_best = np.array([0., 0., 0.])
                    logger.warning(f"No pose could be determined for {img_data.img_name} and object_id {object_id}")
                    score = 0.
                else:
                    score = 0.

            else:
                R_best = np.array([[1.0, 0., 0.],
                                   [0., 1.0, 0.],
                                   [0., 0., 1.0]])

                t_best = np.array([0., 0., 0.])
                logger.warning(
                    f"No Pose could be determined for {img_data.img_name} and object_id {object_id} because no object crop available")
                score = 0.

            # Prepare for writing:
            R_best_str = " ".join(map(str, R_best.flatten()))
            t_best_str = " ".join(map(str, t_best * 1000))
            #elapsed_time = end_time - start_time
            # Write the detections to the CSV file

            # ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
            with open(csv_file, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                #csv_writer.writerow(
                #    [img_data.scene_id, img_data.img_name, object_id, score, R_best_str, t_best_str, elapsed_time])
                csv_writer.writerow(
                    [img_data.scene_id, img_data.img_name, object_id, score, R_best_str, t_best_str])

            if config['debug_imgs']:
                if i % config['debug_imgs'] == 0:
                    dbg_img = vis_utils.create_debug_image(R_best, t_best, img_data.R_gts[i], img_data.t_gts[i],
                                                           np.asarray(img_data.img),
                                                           img_data.cam_K,
                                                           img_data.model_infos[i],
                                                           config['scale_factor'],
                                                           image_shape=(config['image_resolution'][0],
                                                                        config['image_resolution'][1]),
                                                           colEst=(0, 255, 0))

                    dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)

                    if img_data.masks[i] is not None:
                        dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks[i]])
                    else:
                        dbg_img_mask = dbg_img

                    cv2.imwrite(os.path.join(debug_img_path, f"{img_data.img_name}_{img_data.obj_ids[i]}.png"),
                                dbg_img_mask)
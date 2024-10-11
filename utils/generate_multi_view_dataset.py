import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
import mvtec3d_util as mvt_util
import argparse
import render_utils
import glob
import shutil
import cv2
import time


def get_specific_resolution_pcd_rgb(organized_pc, rgb_image, width, height):
    resized_organized_pc = mvt_util.resize_organized_pc(organized_pc, target_height=height, target_width=width, tensor_out=False)
    resized_rgb_image = cv2.resize(rgb_image, (height, width))

    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(resized_organized_pc)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    o3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    none_zero_rgb = cv2.cvtColor(resized_rgb_image, cv2.COLOR_BGR2RGB)
    none_zero_rgb = none_zero_rgb.reshape(resized_rgb_image.shape[0] * resized_rgb_image.shape[1], resized_rgb_image.shape[2])
    none_zero_rgb = none_zero_rgb[nonzero_indices, :] / 255.

    return o3d_pc, resized_rgb_image, none_zero_rgb

# ALL DATA IS RESIZED TO 224
def preprocess_pc(multi_view_vis:render_utils.MultiViewRender, tiff_path, root_path, save_path):
    # READ FILES
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")

    gt_exists = os.path.isfile(gt_path)

    # READ Point Cloud
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    # calculate fpfh features and render multi-view images
    rgb_image = cv2.imread(rgb_path)

    _224_pcd, _224_rgb, _224_nonezero_rgb = get_specific_resolution_pcd_rgb(organized_pc, rgb_image, width=224, height=224)
    _ori_reso_pcd, _ori_reso_rgb, _ori_nonezero_rgb = get_specific_resolution_pcd_rgb(organized_pc, rgb_image, rgb_image.shape[0], rgb_image.shape[1])

    _ = multi_view_vis.calculate_fpfh_features(_ori_reso_pcd)
    images, points = multi_view_vis.multiview_render(_ori_reso_pcd, _ori_nonezero_rgb, _224_pcd)

    fpfh = multi_view_vis.calculate_fpfh_features(_224_pcd)
    # calculate the root and file paths
    save_rgb_path = rgb_path.replace(root_path, save_path)
    save_gt_path = gt_path.replace(root_path, save_path)

    save_rgb_root = os.path.split(save_rgb_path)[0]
    save_gt_root = os.path.split(save_gt_path)[0]
    tiff_spilt = os.path.split(tiff_path.replace(root_path, save_path))
    save_xyz_root = os.path.join(tiff_spilt[0], tiff_spilt[1][:-5])

    save_xyz_path = os.path.join(save_xyz_root, 'xyz.tiff')

    # MAKE DIRS
    os.makedirs(save_xyz_root, exist_ok=True)
    os.makedirs(save_rgb_root, exist_ok=True)
    if gt_exists:
        os.makedirs(save_gt_root, exist_ok=True)

    # copy images
    cv2.imwrite(save_rgb_path, _224_rgb)
    # shutil.copy(rgb_path, save_rgb_path)
    if gt_exists:
        gt_image = cv2.imread(gt_path)
        gt_image = cv2.resize(gt_image, (224, 224))
        cv2.imwrite(save_gt_path, gt_image)
        # shutil.copy(save_gt_path, save_gt_path)
    shutil.copy(tiff_path, save_xyz_path)

    # save render results, points, and fpfh features
    np.save(os.path.join(save_xyz_root, 'fpfh.npy'), fpfh)

    for idx, (image, point) in enumerate(zip(images, points)):

        # for visualization, check whether we get right results
        # viz_image = render_utils.draw_points3d_on_image(image.astype(np.uint8), point)
        # cv2.imshow('viz1', image.astype(np.uint8))
        # cv2.imshow('viz', viz_image)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(save_xyz_root, f"view_{idx:03d}.png"), image)
        np.save(os.path.join(save_xyz_root, f'view_{idx:03d}.npy'), point.astype(int))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('--dataset_path', type=str,
                        default='../../datasets/mvtec_3d_anomaly_detection',
                        help='The root path of the MVTec 3D-AD. The preprocessing is done inplace (i.e. the preprocessed dataset overrides the existing one)')
    parser.add_argument('--color-option', type=str,
                        default='UNIFORM', choices=['X', 'Y', 'Z', 'NORM', 'FPFH','UNIFORM','RGB'],
                        help='in [X,Y,Z,NORM,FPFH]')
    parser.add_argument('--category', type=str, default='bagel')
    parser.add_argument('--save-dir', type=str,
                        default='../../datasets/multi_view_rgb_mvtec_3d_anomaly_detection',
                        help='The save path for the generated multi-view-mvtec3d dataset')

    # NOTE: You should run the preprocessing.py first

    args = parser.parse_args()

    name = args.category


    color_option_dict = {'X':render_utils.MultiViewRender.COLOR_X,
                         'Y': render_utils.MultiViewRender.COLOR_Y,
                         'Z': render_utils.MultiViewRender.COLOR_Z,
                         'NORM': render_utils.MultiViewRender.COLOR_NORM,
                         'FPFH': render_utils.MultiViewRender.COLOR_FPFH,
                         'UNIFORM': render_utils.MultiViewRender.COLOR_UNIFORM,
                         'RGB': render_utils.MultiViewRender.COLOR_RGB
                         }

    dataset_path = args.dataset_path
    color_option = color_option_dict[args.color_option]
    save_dir = args.save_dir

    split = ['test']

    processed_number = 0
    multi_view_vis = render_utils.MultiViewRender('', color=color_option)


    print(f'processing {name}...')

    time_begin = time.time()
    for s in split:
        img_path = os.path.join(dataset_path, name, s)

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(img_path)

        for defect_type in defect_types:

            tiff_paths = glob.glob(os.path.join(img_path, defect_type, 'xyz') + "/*.tiff")
            tiff_paths.sort()

            for tiff_path in tiff_paths:
                preprocess_pc(multi_view_vis, tiff_path, dataset_path, save_dir)

                processed_number += 1
                if processed_number % 50 == 0:
                    cur_time = time.time()
                    print(f"Processed {processed_number} tiff files... using {cur_time - time_begin:.2f} s...")



import argparse
import numpy as np
import math
import json
import os
import random

import matplotlib.pyplot as plt

import evaluation_util

parser = argparse.ArgumentParser(
    description='Compute and plot errors of estimated poses wrt to pseudo ground truth.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_config', default="config_7scenes_sfm_pgt.json",
                    help='file containing the test case specification including paths to '
                         'pseudo ground truth and estimates of algorithms')

parser.add_argument('--nerf_gt', default="/home/n11373598/work/nerf-vloc/data/redkitchen/query_gt_poses_nerf.txt",
                    help='file containing the test case specification including paths to '
                         'pseudo ground truth and estimates of algorithms')

parser.add_argument('--scene', default="RedKitchen",
                    help='file containing the test case specification including paths to '
                         'pseudo ground truth and estimates of algorithms')

parser.add_argument('--error_threshold', type=float, default=5,
                    help='Error threshold when calculating recall, and bound for plotting error curves.')

parser.add_argument('--error_type', type=str, default='pose', choices=['pose', 'dcre_max', 'dcre_mean'],
                    help='Choice of error type.')

parser.add_argument('--error_max_images', type=int, default=-1,
                    help='Use at most x images when calculating error distribution for speed. -1 for using all.')

opt = parser.parse_args()


with open(opt.data_config, "r") as f:
    test_data = json.load(f)


algo_metrics = {}
dataset_folder = test_data["folder"]
rgb_image_width = test_data['image_width']

subplot_index = 321
for s_idx, scene_data in enumerate(test_data["scenes"]):

    if scene_data["name"] != opt.scene:
        continue

    scene_name = scene_data['name']
    scene_folder = scene_data['folder']

    print(f"\n{scene_name}")

    # load pseudo ground truth
    pgt_poses = evaluation_util.read_pose_data(scene_data["pgt"])
    pgt_poses_nerf = evaluation_util.read_pose_data(opt.nerf_gt)

    if 0 < opt.error_max_images < len(pgt_poses):
        keys = random.sample(pgt_poses.keys(), opt.error_max_images)
        pgt_poses = {k: pgt_poses[k] for k in keys}

    # iterate through algorithm estimates
    for estimate_data in scene_data["estimates"]:

        if estimate_data['algo'] not in ["vloc", "nerf-vloc", "Active Search"]:
            continue

        # initialise algorithm accumulated metrics
        if estimate_data['algo'] not in algo_metrics:
            algo_metrics[estimate_data['algo']] = []

        # load estimated poses
        est_poses = evaluation_util.read_pose_data(estimate_data["estimated_poses"])

        # main evaluation loop
        t_errors = np.ndarray((len(pgt_poses), ))
        r_errors = np.ndarray((len(pgt_poses), ))

        for i, query_file in enumerate(pgt_poses):
            try:
                if "nerf" in estimate_data['algo']:
                    pgt_pose, rgb_focal_length = pgt_poses_nerf[query_file]
                else:
                    pgt_pose, rgb_focal_length = pgt_poses[query_file]

                est_pose, _ = est_poses[query_file]
                t_err, r_err = evaluation_util.compute_error_rot_trans(pgt_pose, est_pose)
                t_errors[i] = t_err
                r_errors[i] = r_err

            except KeyError:
                # catching the case that an algorithm did not provide an estimate
                t_errors[i] = -1
                r_errors[i] = -1

        plt.subplot(subplot_index)
        plt.hist(t_errors, bins=100, range=[0, np.max(t_errors)])
        plt.title(f"{estimate_data['algo']} translation (cm)")
        plt.subplot(subplot_index+1)
        plt.title(f"{estimate_data['algo']} rotation (degree)")
        plt.hist(r_errors, bins=100, range=[0, np.max(r_errors)])
        subplot_index += 2
plt.show()

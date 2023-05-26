from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
# import json_tricks as json
import pickle
# import scipy.io as scio
# import logging
import copy
import os
# from collections import OrderedDict

from dataset.JointsDataset import JointsDataset   # 改
from cameras_cpu import camera_to_world_frame, project_pose
# import cv2

INF = 1e8
JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]


H36M_TO_PANOPTIC = [8, 9, 0, 11, 12, 13, 4, 5, 6, 14, 15, 16, 1, 2, 3]


def get_cam(camera):
    fx, fy = camera['f'][0][0][0], camera['f'][0][0][1]
    cx, cy = camera['c'][0][0][0], camera['c'][0][0][1]
    K = np.eye(3)
    K[0, 0] = fx.cpu().numpy()
    K[1, 1] = fy.cpu().numpy()
    K[0, 2] = cx.cpu().numpy()
    K[1, 2] = cy.cpu().numpy()
    camera['K'] = K
    return camera


def fastMT2mvpdatasets(img_key, transfromed_imgs, meta_data_list, num_views=4):
    for i in range(num_views):
        all_poses_3d = []
        all_poses_vis_3d = []
        all_poses = []
        all_poses_vis = []

        camera_i = get_cam(meta_data_list[i]['camera'])
        joints_3d = \
                camera_to_world_frame(
                    meta_data_list[i]['joints_3d'][:,:,0:3],
                    camera['R'],
                    camera['T'])[H36M_TO_PANOPTIC]
        import ipdb
        ipdb.set_trace()  # 确认一下[H36M_TO_PANOPTIC]是否提取成功

        joints_2d = project_pose(joints_3d, camera)

        joints_3d_vis = meta_data_list[i]['joints_3d'][0][:,3][H36M_TO_PANOPTIC]
        all_poses_3d.append(joints_3d)
        all_poses_vis_3d.append(joints_3d_vis)

        joints_2d_vis = joints_3d_vis  # 存疑
        all_poses.append(joints_2d)
        all_poses_vis.append(joints_2d_vis)

        meta_data_list[i]['joints_2d_ori'] = meta_data_list[i]['joints_2d']
        meta_data_list[i]['joints_3d'] = all_poses_3d
        meta_data_list[i]['joints_3d_vis'] = all_poses_vis_3d
        meta_data_list[i]['joints_2d'] = all_poses
        meta_data_list[i]['joints_2d_vis'] = all_poses_vis

        our_cam = {}
        our_cam['R'] = camera['R'][0].cpu().numpy()
        our_cam['T'] = camera['T'][0].cpu().numpy()
        our_cam['standard_T'] = -np.dot(camera['R'], camera['T'])
        import ipdb
        ipdb.set_trace()  # 确认一下上一步np.dot是否可行

        our_cam['K'] = camera['K']
        our_cam['fx'] = camera['f'][0][0][0].cpu().numpy()
        our_cam['fy'] = camera['f'][0][0][1].cpu().numpy()
        our_cam['cx'] = camera['c'][0][0][0].cpu().numpy()
        our_cam['cy'] = camera['c'][0][0][1].cpu().numpy()
        our_cam['k'] = camera['k'][0].reshape(3, 1).cpu().numpy()
        our_cam['p'] = camera['p'][0].reshape(2, 1).cpu().numpy()
        meta_data_list[i]['camera_ori'] = meta_data_list[i]['camera']
        meta_data_list[i]['camera'] = our_cam

        meta_data_list[i]['image_name'] = img_key[i][0][0]
        meta_data_list[i]['image'] = meta_data_list[i]['ori_img_keepsize']

        return meta_data_list # 下面进入Jointdataset处理阶段


    

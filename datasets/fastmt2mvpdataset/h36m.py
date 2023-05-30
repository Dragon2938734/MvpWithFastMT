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
import cv2
import torch
import torchvision.transforms as transforms
# from collections import OrderedDict

# from dataset.JointsDataset import JointsDataset   # 改
import sys
sys.path.append("/root/autodl-tmp/FastMETRO/src/datasets/fastmt2mvpdataset")
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
    fx, fy = camera['f'][0][0], camera['f'][1][0]
    cx, cy = camera['c'][0][0], camera['c'][1][0]
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    camera['K'] = K
    return camera


def get_scale(image_size, resized_size):
    w, h = image_size
    w_resized, h_resized = resized_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized
    scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

    return scale


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return np.array(b) + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if isinstance(scale, torch.Tensor):
        scale = np.array(scale.cpu())
    if isinstance(center, torch.Tensor):
        center = np.array(center.cpu())
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180
    if src_w >= src_h:
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
    else:
        src_dir = get_dir([src_h * -0.5, 0], rot_rad)
        dst_dir = np.array([dst_h * -0.5, 0], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift     # x,y
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def jointsdataset(meta_data,
                  image_size = np.array([320, 320]),
                  heatmap_size = np.array([80, 80]),
                  root_id = 2,
                  transform = None):
    image = meta_data['image']  # 此时图片在从tsv格式转为图片时已经为rgb了，所以不用再次调整
    # import ipdb
    # ipdb.set_trace()
    image = image[:1000]   # crop image from 1002 x 1000 to 1000 x 1000 for h36m
    
    joints = meta_data['joints_2d_mvp']
    joints_3d = meta_data['joints_3d_mvp']
    joints_vis = meta_data['joints_2d_vis']
    joints_3d_vis = meta_data['joints_3d_vis']
    # import ipdb
    # ipdb.set_trace()
    nposes = len(joints)

    height, width, _ = image.shape
    c = np.array([width / 2.0, height / 2.0])
    s = get_scale((width, height), image_size)
    r = 0  # NOTE: do not apply rotation augmentation
    trans = get_affine_transform(c, s, r, image_size, inv=0)
    # NOTE: this trans represents full image to cropped image,
    # not full image->heatmap
    input = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
    if transform:
        input = transform(input)
    
    for i in range(len(joints)):
        if joints_vis[i] > 0.0:
            joints[i, 0:2] = affine_transform(
                joints[i, 0:2], trans)
            if (np.min(joints[i, :2]) < 0 or
                    joints[i, 0] >= image_size[0] or
                    joints[i, 1] >= image_size[1]):
                joints_vis[i] = 0
    
    # NOTE: deal with affine transform
    # affine transform between origin img and heatmap
    aff_trans = np.eye(3, 3)
    aff_trans[0:2] = trans  # full img -> cropped img
    inv_aff_trans = np.eye(3, 3)
    inv_trans = get_affine_transform(c, s, r, image_size, inv=1)
    inv_aff_trans[0:2] = inv_trans
    # 3x3 data augmentation affine trans (scale rotate=0)
    # NOTE: this transformation contains both heatmap->image scale affine
    # and data augmentation affine
    aug_trans = np.eye(3, 3)
    aug_trans[0:2] = trans  # full img -> cropped img
    hm_scale = heatmap_size / image_size
    scale_trans = np.eye(3, 3)  # cropped img -> heatmap
    scale_trans[0, 0] = hm_scale[1]
    scale_trans[1, 1] = hm_scale[0]
    aug_trans = scale_trans @ aug_trans
    # NOTE: aug_trans is superset of affine_trans
    # 接下来原代码中对关节点的操作实际上就是增加了一个维度，如果后续网络需要，则将下列代码取消注释
    joints_u = joints[np.newaxis, :]
    joints_vis_u = joints_vis[np.newaxis, :]
    joints_3d_u = joints_3d[np.newaxis, :]
    joints_3d_vis_u = joints_3d_vis[np.newaxis, :]

    if isinstance(root_id, int):
        # import ipdb
        # ipdb.set_trace()
        roots_3d = joints_3d_u[:, root_id]  # 如果上面代码取消了注释，则改为 joints_3d_u[:, root_id]
    elif isinstance(root_id, list):
        roots_3d = np.mean([joints_3d_u[j] for j in root_id], axis=0)
    
    # NOTE: deal with camera
    cam = meta_data['camera']
    cam_intri = np.eye(3, 3)
    cam_intri[0, 0] = float(cam['fx'])
    cam_intri[1, 1] = float(cam['fy'])
    cam_intri[0, 2] = float(cam['cx'])
    cam_intri[1, 2] = float(cam['cy'])
    cam_R = cam['R']
    cam_T = cam['T']
    cam_standard_T = cam['standard_T']

    meta = {
        'image_name': meta_data['image_name'],
        'num_person': 1,
        'joints_3d': joints_3d_u,
        'joints_3d_vis': joints_3d_vis_u,
        'roots_3d': roots_3d,
        'joints': joints_u,
        'joints_vis': joints_vis_u,
        'center': c,
        'scale': s,
        'rotation': r,
        'camera': cam,
        'camera_Intri': cam_intri,
        'camera_R': cam_R,
        # for ray direction generation
        'camera_focal': np.stack([cam['fx'], cam['fy'], np.ones_like(cam['fy'])]),
        'camera_T': cam_T,
        'camera_standard_T': cam_standard_T,
        'affine_trans': aff_trans,
        'inv_affine_trans': inv_aff_trans,
        'aug_trans': aug_trans,
    }

    import ipdb
    ipdb.set_trace()

    return input, meta


def fastMT2mvpdatasets(img_key, meta_data_list, num_views=4):
    input = []
    meta = []

    for i in range(num_views):
        # all_poses_3d = []
        # all_poses_vis_3d = []
        # all_poses = []
        # all_poses_vis = []

        camera_i = get_cam(meta_data_list[i]['camera'])
        joints_3d = \
                camera_to_world_frame(
                    meta_data_list[i]['ori_joints_3d'][:,0:3],
                    camera_i['R'],
                    camera_i['T'])[H36M_TO_PANOPTIC]
        # import ipdb
        # ipdb.set_trace()  # 确认一下[H36M_TO_PANOPTIC]是否提取成功

        joints_2d = project_pose(joints_3d, camera_i)

        joints_3d_vis = meta_data_list[i]['ori_joints_3d'][:,3][H36M_TO_PANOPTIC]
        # all_poses_3d.append(joints_3d)
        # all_poses_vis_3d.append(joints_3d_vis)

        joints_2d_vis = joints_3d_vis  # 存疑
        # all_poses.append(joints_2d)
        # all_poses_vis.append(joints_2d_vis)

        # meta_data_list[i]['joints_2d_ori'] = meta_data_list[i]['ori_joints_2d']
        meta_data_list[i]['joints_3d_mvp'] = joints_3d
        meta_data_list[i]['joints_3d_vis'] = joints_3d_vis
        meta_data_list[i]['joints_2d_mvp'] = joints_2d
        meta_data_list[i]['joints_2d_vis'] = joints_2d_vis

        our_cam = {}
        our_cam['R'] = camera_i['R']
        our_cam['T'] = camera_i['T']
        our_cam['standard_T'] = -np.dot(camera_i['R'], camera_i['T'])
        our_cam['K'] = camera_i['K']
        our_cam['f'] = camera_i['f']
        our_cam['c'] = camera_i['c']
        our_cam['fx'] = camera_i['f'][0][0]
        our_cam['fy'] = camera_i['f'][1][0]
        our_cam['cx'] = camera_i['c'][0][0]
        our_cam['cy'] = camera_i['c'][1][0]
        our_cam['k'] = camera_i['k'].reshape(3, 1)
        our_cam['p'] = camera_i['p'].reshape(2, 1)
        meta_data_list[i]['camera_ori'] = meta_data_list[i]['camera']
        meta_data_list[i]['camera'] = our_cam
        meta_data_list[i]['image_name'] = img_key[i]
        # 下面进入Jointdataset处理阶段
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(),normalize,])

        input[i], meta[i] = jointsdataset(meta_data_list[i], transform=transform)
        # 下面继续按照Jointdataset中文件获得target、weight、target_3d、input_heatmap

    return meta_data_list 


    

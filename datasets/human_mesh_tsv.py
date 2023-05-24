# ----------------------------------------------------------------------------------------------
# METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
import code

import sys
sys.path.append('/root/autodl-tmp/FastMETRO') # 此处添加绝对路径，不知道就用pwd查

from src.utils.tsv_file import TSVFile, CompositeTSVFile
from src.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml
from src.utils.image_ops import img_from_base64, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms


class MeshTSVDataset(object):
    def __init__(self, img_file, label_file=None, hw_file=None, cameraparams_file=None,
                 linelist_file=None, is_train=True, cv2_output=False, scale_factor=1):

        self.img_file = img_file
        self.label_file = label_file
        self.hw_file = hw_file
        self.cameraparams_file = cameraparams_file
        # self.linelist_file = linelist_file
        self.linelistCam1_file = linelist_file[0]
        self.linelistCam2_file = linelist_file[1]
        self.linelistCam3_file = linelist_file[2]
        self.linelistCam4_file = linelist_file[3]

        self.img_tsv = self.get_tsv_file(img_file)
        self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
        self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)

        if self.is_composite:
            assert op.isfile(self.linelist_file)
            self.line_list = [i for i in range(self.hw_tsv.num_rows())]
        else:
            # self.line_list = load_linelist_file(linelist_file)
            self.lineCam1_list = load_linelist_file(linelist_file[0])
            self.lineCam2_list = load_linelist_file(linelist_file[1])
            self.lineCam3_list = load_linelist_file(linelist_file[2])
            self.lineCam4_list = load_linelist_file(linelist_file[3])

        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224

        self.image_keys = self.prepare_image_keys()

        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        self.pelvis_index = self.joints_definition.index('Pelvis')

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.linelist_file,
                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)

            a = TSVFile(tsv_path)
            # import ipdb
            # ipdb.set_trace()
            return TSVFile(tsv_path)  # 此处 tsv_path=='datasets/human3.6m/train.img.tsv'

    def get_valid_tsv(self):
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.label_tsv:
            return self.label_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_line_no(self, idx):
        # import ipdb
        # ipdb.set_trace()
        line_no_cam1 = self.lineCam1_list[idx]
        line_no_cam2 = self.lineCam2_list[idx]
        line_no_cam3 = self.lineCam3_list[idx]
        line_no_cam4 = self.lineCam4_list[idx]
        # 此处返回一组4个视角下的图片的索引
        return idx if self.line_list is None else [line_no_cam1,line_no_cam2,line_no_cam3,line_no_cam4]
        # return idx if self.line_list is None else self.line_list[idx]

    # def get_image(self, idx):  # 单视角的返回图片函数（一张图片）
    #     line_no = self.get_line_no(idx)
    #     row = self.img_tsv[line_no]
    #     # use -1 to support old format with multiple columns.
    #     cv2_im = img_from_base64(row[-1])
    #     if self.cv2_output:
    #         return cv2_im.astype(np.float32, copy=True)
    #     cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

    #     return cv2_im

    def get_image(self, idx):   #返回多目图片组（4张）
        line_no = self.get_line_no(idx)
        images = []
        for i in range(4):
            line_no_cam_i = line_no[i]
            row = self.img_tsv[line_no_cam_i]
            # use -1 to support old format with multiple columns.
            cv2_im = img_from_base64(row[-1])
            if self.cv2_output:
                return cv2_im.astype(np.float32, copy=True)
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            images.append(cv2_im)
        return images

    def get_camerasparams(self, img_key):
        camerasparams = []
        cam1para = {}
        cam2para = {}
        cam3para = {}
        cam4para = {}
        img_name = img_key[0]
        f = h5py.File(self.cameraparams_file, 'r')
        if img_name.find('S1') != -1:
            for item in f['subject1']['camera1'].keys():
                cam1para[item] = f['subject1']['camera1'][item][()]
            for item in f['subject1']['camera2'].keys():
                cam2para[item] = f['subject1']['camera2'][item][()]
            for item in f['subject1']['camera3'].keys():
                cam3para[item] = f['subject1']['camera3'][item][()]
            for item in f['subject1']['camera4'].keys():
                cam4para[item] = f['subject1']['camera4'][item][()]
        elif img_name.find('S5') != -1:
            for item in f['subject5']['camera1'].keys():
                cam1para[item] = f['subject5']['camera1'][item][()]
            for item in f['subject5']['camera2'].keys():
                cam2para[item] = f['subject5']['camera2'][item][()]
            for item in f['subject5']['camera3'].keys():
                cam3para[item] = f['subject5']['camera3'][item][()]
            for item in f['subject5']['camera4'].keys():
                cam4para[item] = f['subject5']['camera4'][item][()]
        elif img_name.find('S6') != -1:
            for item in f['subject6']['camera1'].keys():
                cam1para[item] = f['subject6']['camera1'][item][()]
            for item in f['subject6']['camera2'].keys():
                cam2para[item] = f['subject6']['camera2'][item][()]
            for item in f['subject6']['camera3'].keys():
                cam3para[item] = f['subject6']['camera3'][item][()]
            for item in f['subject6']['camera4'].keys():
                cam4para[item] = f['subject6']['camera4'][item][()]
        elif img_name.find('S7') != -1:
            for item in f['subject7']['camera1'].keys():
                cam1para[item] = f['subject7']['camera1'][item][()]
            for item in f['subject7']['camera2'].keys():
                cam2para[item] = f['subject7']['camera2'][item][()]
            for item in f['subject7']['camera3'].keys():
                cam3para[item] = f['subject7']['camera3'][item][()]
            for item in f['subject7']['camera4'].keys():
                cam4para[item] = f['subject7']['camera4'][item][()]
        elif img_name.find('S8') != -1:
            for item in f['subject8']['camera1'].keys():
                cam1para[item] = f['subject8']['camera1'][item][()]
            for item in f['subject8']['camera2'].keys():
                cam2para[item] = f['subject8']['camera2'][item][()]
            for item in f['subject8']['camera3'].keys():
                cam3para[item] = f['subject8']['camera3'][item][()]
            for item in f['subject8']['camera4'].keys():
                cam4para[item] = f['subject8']['camera4'][item][()]
        elif img_name.find('S9') != -1:
            for item in f['subject9']['camera1'].keys():
                cam1para[item] = f['subject9']['camera1'][item][()]
            for item in f['subject9']['camera2'].keys():
                cam2para[item] = f['subject9']['camera2'][item][()]
            for item in f['subject9']['camera3'].keys():
                cam3para[item] = f['subject9']['camera3'][item][()]
            for item in f['subject9']['camera4'].keys():
                cam4para[item] = f['subject9']['camera4'][item][()]
        elif img_name.find('S11') != -1:
            for item in f['subject11']['camera1'].keys():
                cam1para[item] = f['subject11']['camera1'][item][()]
            for item in f['subject11']['camera2'].keys():
                cam2para[item] = f['subject11']['camera2'][item][()]
            for item in f['subject11']['camera3'].keys():
                cam3para[item] = f['subject11']['camera3'][item][()]
            for item in f['subject11']['camera4'].keys():
                cam4para[item] = f['subject11']['camera4'][item][()]
        else:
            print('Subject no. wrong, no such subject!')
            raise AssertionError()
        return [cam1para, cam2para, cam3para, cam4para]  # 返回对应四个视角的相机参数

    # def get_annotations(self, idx): # 单视角返回标注
    #     line_no = self.get_line_no(idx)
    #     if self.label_tsv is not None:
    #         row = self.label_tsv[line_no]
    #         annotations = json.loads(row[1])
    #         return annotations
    #     else:
    #         return []

    def get_annotations(self, idx): # 多视角返回标注
        line_no = self.get_line_no(idx)
        annotations = []
        if self.label_tsv is not None:
            for i in range(4):
                line_no_cam_i = line_no[i]
                row = self.label_tsv[line_no_cam_i]
                annotation = json.loads(row[1])
                annotations.append(annotation)
            return annotations
        else:
            return []

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to 
        # decode the labels to specific formats for each task. 
        return annotations


    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        line_no_cam1 = line_no[0]
        line_no_cam2 = line_no[1]
        line_no_cam3 = line_no[2]
        line_no_cam4 = line_no[3]

        # based on the overhead of reading each row.
        if self.hw_tsv:
            return [self.hw_tsv[line_no_cam1][0], self.hw_tsv[line_no_cam2][0], self.hw_tsv[line_no_cam3][0], self.hw_tsv[line_no_cam4][0]]
            # return self.hw_tsv[line_no][0]  # 在此处返回图像名称'images/S1_Directions_1.54138969_000001.jpg'
        elif self.label_tsv:
            return self.label_tsv[line_no][0]
        else:
            return self.img_tsv[line_no][0]

    def __len__(self):
        if self.line_list is None:
            return self.img_tsv.num_rows() 
        else:
            return len(self.lineCam1_list) # 返回的数据长度是成套的包含4个视角的组数

    def __getitem__(self, idx):

        img = self.get_image(idx)  # 一个列表,包含4张图片
        img_key = self.get_img_key(idx) # 一个列表，包含4张图片的名称
        camerasparams = self.get_camerasparams(img_key) # 一个列表，包含4各视角的相机参数，每个相机参数为一个字典
        annotations = self.get_annotations(idx) # 一个列表，包含4张图片各自对应的标注
        
        # center_list = []
        # scale_list = []
        # has_2d_joints_list = []
        # has_3d_joints_list = []
        # joints_2d_list = []
        # joints_3d_list = []
        # has_smpl_list = []
        # pose_list = []
        # betas_list = []
        # gender_list = []
        transfromed_imgs = []
        meta_data_list = []

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        for i in range(4):
            annotations = annotations[i][0]
            center = annotations['center']
            scale = annotations['scale']
            has_2d_joints = annotations['has_2d_joints']
            has_3d_joints = annotations['has_3d_joints']
            joints_2d = np.asarray(annotations['2d_joints'])
            joints_3d = np.asarray(annotations['3d_joints'])
            # center_list.append(annotations[i]['center'])
            # scale_list.append(annotations[i]['scale'])
            # has_2d_joints_list.append(annotations[i]['has_2d_joints'])
            # has_3d_joints_list.append(annotations[i]['has_3d_joints'])
            # joints_2d = np.asarray(annotations[i]['2d_joints'])
            # joints_3d = np.asarray(annotations[i]['3d_joints'])
            if joints_2d.ndim==3:
                joints_2d = joints_2d[0]
            if joints_3d.ndim==3:
                joints_3d = joints_3d[0]
            # joints_2d_list.append(joints_2d)
            # joints_3d_list.append(joints_3d)


            # Get SMPL parameters, if available
            has_smpl = np.asarray(annotations['has_smpl'])
            # has_smpl_list.append(has_smpl)
            pose = np.asarray(annotations['pose'])
            # pose_list.append(pose)
            betas = np.asarray(annotations['betas'])
            # betas_list.append(betas)

            try:
                gender = annotations['gender']
            except KeyError:
                gender = 'none'

        # # Get augmentation parameters
        # flip,pn,rot,sc = self.augm_params()

            # Process image_i
            img[i] = self.rgb_processing(img[i], center, sc*scale, rot, flip, pn)
            img[i] = torch.from_numpy(img[i]).float()
            # Store image before normalization to use it in visualization
            transfromed_img_i = self.normalize_img(img[i])
            transfromed_imgs.append(transfromed_img_i)
            
            # normalize 3d pose by aligning the pelvis as the root (at origin)
            root_pelvis = joints_3d[self.pelvis_index,:-1]
            joints_3d[:,:-1] = joints_3d[:,:-1] - root_pelvis[None,:]
            # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
            joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
            # 2d pose augmentation
            joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)

            meta_data = {}
            meta_data['ori_img'] = img
            meta_data['cameras'] = camerasparams[i]
            meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
            meta_data['betas'] = torch.from_numpy(betas).float()
            meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
            meta_data['has_3d_joints'] = has_3d_joints
            meta_data['has_smpl'] = has_smpl

            # Get 2D keypoints and apply augmentation transforms
            meta_data['has_2d_joints'] = has_2d_joints
            meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
            meta_data['scale'] = float(sc * scale)
            meta_data['center'] = np.asarray(center).astype(np.float32)
            meta_data['gender'] = gender
            meta_data_list.append(meta_data)
        
        return img_key, transfromed_imgs, meta_data_list # 此处返回3个列表，包含4个视角下的图名、图片、标注(包括相机参数)



class MeshTSVYamlDataset(MeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, is_train=True, cv2_output=False, scale_factor=1):
        self.cfg = load_from_yaml_file(yaml_file)
        self.is_composite = self.cfg.get('composite', False)
        self.root = op.dirname(yaml_file)

        if self.is_composite==False:
            img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
            cameraparams_file = find_file_path_in_yaml(self.cfg.get('cameraparams',None), self.root)
            label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                                self.root)
            hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
            # linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
            #                                     self.root)
            linelistCam1_file = find_file_path_in_yaml(self.cfg.get('linelistCam1', None),
                                            self.root)
            linelistCam2_file = find_file_path_in_yaml(self.cfg.get('linelistCam2', None),
                                            self.root)
            linelistCam3_file = find_file_path_in_yaml(self.cfg.get('linelistCam3', None),
                                            self.root)
            linelistCam4_file = find_file_path_in_yaml(self.cfg.get('linelistCam4', None),
                                            self.root)
            linelist_file = [linelistCam1_file, linelistCam2_file, linelistCam3_file, linelistCam4_file]

            # import ipdb
            # ipdb.set_trace() 
        else:
            img_file = self.cfg['img']
            hw_file = self.cfg['hw']
            label_file = self.cfg.get('label', None)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)

        super(MeshTSVYamlDataset, self).__init__(
            img_file, label_file, hw_file, cameraparams_file, linelist_file, is_train, cv2_output=cv2_output, scale_factor=scale_factor)


if __name__ == '__main__':
    yaml_file = 'datasets/human3.6m/train.smpl.p1.yaml'
    dataset = MeshTSVYamlDataset(yaml_file, True, False, scale_factor=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1)
    
    for i, item in enumerate(data_loader):
        print('i:', i)
        import ipdb
        ipdb.set_trace()
        
'''
Author: zyq
Date: 2024-05-09 09:26:34
LastEditTime: 2024-05-09 16:34:41
FilePath: /CardiacAnalysis/utils/camus_processor.py
Description: camus dataset processor

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''

import os
import cv2
import random
import numpy as np
import SimpleITK as sitk

class CamusProcessor:
    def __init__(self, video_path, match_pattern='4CH'):
        if match_pattern != '4CH' and match_pattern != '2CH':
            #仅处理2CH和4CH两种左心室视图
            raise Exception('match_pattern must be 4CH or 2CH')

        self.video_path = video_path
        self.match_pattern = match_pattern
        self.sequence_type = match_pattern + '_sequence.mhd'
        self.cfg_info_dict = {}    
    
    def get_cfg_info(self, cfg_path):
        with open(cfg_path, 'r') as f:
            for i in f.readlines():
                info = i.strip().split(':')
                self.cfg_info_dict[info[0]] = str(info[1].strip())
        return self.cfg_info_dict
    
    def get_good_sequence_random(self):
        '''
        随机获取图像质量为good的视频帧序列
        '''
        sequence_list = self.__find_mhd_files(self.video_path)
        while(True):
            picked_one = random.choice(sequence_list)
            patient_id = picked_one.split('/')[-2]
            cfg_path = self.video_path + patient_id + '/Info_{}.cfg'.format(self.match_pattern)
            if self.__is_good_image(cfg_path):
                real_sequence_list = self.__reshape_img(picked_one)
                return real_sequence_list, cfg_path
    
    def get_all_good_sequence(self):
        '''
        获取CAMUS中所有图像质量为good的视频帧序列
        '''
        good_sequence_list = []
        good_sequence_cfg_list = []
        sequence_list = self.__find_mhd_files(self.video_path)
        for sequence in sequence_list:
            patient_id = sequence.split('/')[-2]
            cfg_path = self.video_path + patient_id + '/Info_{}.cfg'.format(self.match_pattern)
            if self.__is_good_image(cfg_path):
                good_sequence_list.append(self.__reshape_img(sequence))
                good_sequence_cfg_list.append(cfg_path)
        return good_sequence_list, good_sequence_cfg_list
    
    def __find_mhd_files(self, directory):
        mhd_files_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(self.sequence_type):
                    mhd_files_list.append(os.path.join(root, file))
        return mhd_files_list
    
    def __is_good_image(self, cfg_path):
        cfg_info_dict = self.get_cfg_info(cfg_path)
        if cfg_info_dict['ImageQuality'] == 'Good':
            return True
        return False
    
    def __reshape_img(self, mhd_image_path):
        img_lst = []
        origin_image = sitk.ReadImage(mhd_image_path, sitk.sitkFloat32)
        images_np = sitk.GetArrayFromImage(origin_image)
        for idx in range(len(images_np)):
            new_array = cv2.resize(images_np[idx,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
            new_array = np.reshape(new_array,(384,384,1))
            new_array = new_array/255
            img_lst.append(new_array)
        return img_lst
    
    
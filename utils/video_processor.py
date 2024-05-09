'''
Author: zyq
Date: 2024-05-09 08:55:20
LastEditTime: 2024-05-09 09:37:57
FilePath: /CardiacAnalysis/utils/video_processor.py
Description: 处理.avi视频文件

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_sequence_list = []
    
    def preprocess_video(self, is_split_frame=True, frame_size=(384, 384)):
        cap =  cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file. fpath=({})".format(self.video_path))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            # 读取下一帧
            ret, frame = cap.read()
        
            # 如果正确读取帧，ret为True
            if not ret:
                raise Exception("Failed to grab a frame")

            origin_image = np.mean(frame, axis=2)
            if is_split_frame:
                origin_image = origin_image[50:600,100:700] #需要根据图像本身的情况进行切割
            origin_image_float = origin_image.astype(np.float32)
            normalized_image = cv2.normalize(origin_image_float, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            new_array = cv2.resize(normalized_image, dsize=frame_size, interpolation=cv2.INTER_CUBIC)
            new_array = np.reshape(new_array,(frame_size[0],frame_size[1],1))
            self.video_sequence_list.append(new_array)
        return self.video_sequence_list
    
    def show_frame(self, fname='test.jpg'):
        if len(self.video_sequence_list) == 0:
            print('no frame to show. just return 0')
            return
        else:
            image_np = self.video_sequence_list[0]
            plt.figure(figsize=(5, 5))
            plt.imshow(image_np[:])
            plt.axis('off')
            plt.savefig(fname)
            
    
    
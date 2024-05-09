'''
Author: zyq
Date: 2024-05-09 10:57:17
LastEditTime: 2024-05-09 11:47:48
FilePath: /CardiacAnalysis/utils/utils.py
Description: lv utils

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''

import cv2
import numpy as np
import math

class LeftVentricularUtils:
    def __init__(self, lv_mask_label=1):
        self.lv_mask_label = lv_mask_label
    
    #private method
    def __get_lv_mask(self, mask_img):
        new_mask = np.array(mask_img)
        new_mask[new_mask != self.lv_mask_label] = 0
        return new_mask
    
    def __get_lv_area(self, lv_mask):
        lv_mask = lv_mask.astype(np.uint8)
        contours, _ = cv2.findContours(lv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max([cv2.contourArea(c) for c in contours])
        max_contour = max(contours, key=cv2.contourArea)
        return max_area, max_contour
    
    def __cal_lv_vol(self, lv_area, lv_long_axis_length):
        res = (8 * lv_area * lv_area) / (3 * math.pi * lv_long_axis_length)
        return res

    def __split_dots(self, points, ellipse):
        quad_1_dots = []
        quad_2_dots = []
        quad_3_dots = []
        quad_4_dots = []
        for point in points:
            # 提取椭圆参数
            cx, cy = ellipse[0]
            
            # 点的坐标
            x, y = point
            # 点相对于椭圆中心的坐标
            relative_x = x - cx
            relative_y = y - cy
            
            # 将角度转换为弧度
            angle_rad = np.deg2rad(-ellipse[2])  # 使用相反的角度进行逆旋转
            
            # 创建逆旋转矩阵
            rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                        [np.sin(angle_rad),  np.cos(angle_rad)]])
            
            # 应用逆旋转矩阵来调整点的坐标
            adjusted_x, adjusted_y = rotation_matrix.dot([relative_x, relative_y])
            
            # 判断象限
            if adjusted_x > 0 and adjusted_y > 0:
                quad_1_dots.append([x, y])
            elif adjusted_x < 0 and adjusted_y > 0:
                quad_2_dots.append([x, y])
            elif adjusted_x < 0 and adjusted_y < 0:
                quad_3_dots.append([x, y])
            elif adjusted_x > 0 and adjusted_y < 0:
                quad_4_dots.append([x, y])
            else:
                # 如果点在坐标轴上，可以根据具体情况决定如何分配象限
                # 这里我们假设点在第一象限
                quad_1_dots.append([x, y])
        return quad_1_dots, quad_2_dots, quad_3_dots, quad_4_dots

    #public method
    def cal_ef_error(self, real_ef, pred_ef):
        error_percentage = ((pred_ef * 100 - real_ef) / real_ef) *100
        return error_percentage

    def val_ef_result(self, ef_error_list):
        abs_ef_error_list = np.array([abs(i) for i in ef_error_list])
        max_error = np.max(abs_ef_error_list)
        min_error = np.min(abs_ef_error_list)
        mean_error = np.mean(abs_ef_error_list)
        return max_error, min_error, mean_error

    #pred lvef value
    def get_pred_lvef(self, predict_mask_list):
        areas_list = []
        ellipse_list = []
        
        for predict_mask in predict_mask_list:
            #1.get lv mask
            predict_mask = np.array(predict_mask)
            lv_mask = self.__get_lv_mask(predict_mask).astype(np.uint8)

            #2.get lv area and ellipse info
            max_area, max_contour = self.__get_lv_area(lv_mask)
            ellipse = cv2.fitEllipse(max_contour)
            areas_list.append(max_area)
            ellipse_list.append(ellipse)
            real_max_areas = max(areas_list)
            real_min_areas = min(areas_list)

        #3.cal lvef value
        max_areas_idx = areas_list.index(real_max_areas)
        min_areas_idx = areas_list.index(real_min_areas)
        #long axis
        max_long_axis = ellipse_list[max_areas_idx][1][1]
        min_long_axis = ellipse_list[min_areas_idx][1][1] 

        pred_edv = self.__cal_lv_vol(real_max_areas, max_long_axis)
        pred_esv = self.__cal_lv_vol(real_min_areas, min_long_axis)
        pred_ef = (pred_edv - pred_esv) / pred_edv

        return pred_ef, max_areas_idx, min_areas_idx
    
    def get_predict_mask_x_info(self, predict_mask):
        #从mask图像中获取lv信息
        lv_predict_mask = self.__get_lv_mask(predict_mask).astype(np.uint8)
        _, contours = self.__get_lv_area(lv_predict_mask)

        #拟合椭圆, 获取中心点坐标(x,y)
        ellipse = cv2.fitEllipse(contours)
        center = ellipse[0]
        points = contours.reshape(-1, 2)

        quad_1_dots, quad_2_dots, quad_3_dots, quad_4_dots = self.__split_dots(points, ellipse)

        left_up_x_means = np.mean(np.array([dot[0] for dot in quad_1_dots]))
        left_down_x_means = np.mean(np.array([dot[0] for dot in quad_4_dots]))
        left_x_means = np.mean(np.array([dot[0] for dot in quad_1_dots] + [dot[0] for dot in quad_4_dots]))
        return left_up_x_means, left_down_x_means, left_x_means
    

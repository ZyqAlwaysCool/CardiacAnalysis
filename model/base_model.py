'''
desc: 分割模型抽象类, 具体模型需实现抽象类方法
author: zyq
date: 2024/6/27
'''
from abc import ABC, abstractmethod

class BaseSegModel(ABC):
    @abstractmethod
    def load_model(self, model_path):
        '''
        use to load model
        '''
        pass

    @abstractmethod
    def get_img_masks(self, input_img):
        '''
        use to get image mask
        '''
        pass
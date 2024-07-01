## README
模型均需继承自BaseSegModel(base_model.py), 由SegModelFactory生产出具体的模型供推理使用. SAM_Med2D和VM_UnetV2均为第三方库, 只使用其推理部分代码. 原仓库见:
* SAM_Med2D: https://github.com/OpenGVLab/SAM-Med2D
* VM_UnetV2: https://github.com/nobodyplayer1/VM-UNetV2

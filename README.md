<!--
 * @Author: zyq
 * @Date: 2024-05-09 08:56:51
 * @LastEditTime: 2024-05-09 18:02:04
 * @FilePath: /CardiacAnalysis/README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by zyq, All Rights Reserved. 
-->
## README
### 1.仓库说明
本仓库使用在camus数据集上训练的unet模型为例，对视频帧进行逐帧分割，并对分割结果进行指标评估，目前包括：左心室射血分数值预测、左心室室壁运动振幅两项指标。支持以下两种类型输入:
* camus数据集中的病人用例视频帧，即以2CH_sequence.mhd或4CH_sequence.mhd为后缀的文件
* 用户准备的心动检查视频，类型需为a2c或a4c的.avi格式视频，提供视频预处理步骤转换输入
  * 注：对于用户提供的心动检查视频，用户需自行决定是否对视频帧的内容做切割，以去掉图像上的无用信息，保留核心的心动检查图像

左心室射血分数值预测思路：模型分割出左心室图像，找出像素面积最大和最小帧，认定为舒张末期和收缩末期。基于opencv找到最大联通像素面积和最大轮廓，基于最大轮廓拟合椭圆，找到椭圆长轴并按照ES值的计算公式预测LVES值。实验结果表明，camus数据集中60%的数据lvef预测误差在10%以内，若要进一步减少误差，需提高分割模型精度。

左心室室壁运动振幅预测思路：找到舒张末期和收缩末期帧，分别拟合椭圆，基于椭圆长短轴将轮廓坐标点划分为四个象限，通过椭圆左侧的轮廓坐标点，计算出舒张末期和收缩末期x轴平均运动距离，与正常心超的运动距离区间做对比，给出一个经验值粗判。

### 2.快速开始
* 安装依赖包:`pip install -r requirements.txt`
* 跑camus数据集用例(需下载数据集):`python main.py -c`

### 3.参考
* camus数据集: https://paperswithcode.com/dataset/camus
  * 原始数据集地址(尝试下载慢)：https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8
  * kaggle已预处理数据集地址：https://www.kaggle.com/datasets/toygarr/camus-dataset
  * 原始数据集地址(可下载版)：https://aistudio.baidu.com/datasetdetail/66872
* unet-keras: https://github.com/gabrielbaltazarmw2/CAMUS-dataset
  * 使用该仓库中提到的kaggle版预处理后的camus数据集进行模型训练，实测可直接跑通训练
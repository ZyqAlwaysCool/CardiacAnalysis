## 心脏指标分析-多模型对比实验

**数据来源：**[https://www.kaggle.com/datasets/toygarr/camus-dataset](https://www.kaggle.com/datasets/toygarr/camus-dataset)<br />**数据构成：**

1. unet-camus：默认使用，统一训练数据为4ch视图，训练集样本量：675，验证集样本量：225，使用4ch中的各病例视频sequence作为测试样本集，模拟真实医生输入
2. large-camus：yolo系列模型存在无法收敛或推理报错的问题，此时使用备用数据集large-camus。基于原始camus数据集，提取4ch和2ch标注图像，并根据8:2比例划分的数据集，图像尺寸不一，训练数据：1443，验证集363

**项目路径：**/home/kemove/zyq/giit/cardiac/CardiacAnalysis。git链接：[https://git.giitllm.cn/ai/CardiacAnalysis](https://git.giitllm.cn/ai/CardiacAnalysis)<br />**评测数据：**原camus数据集中所有未标注的视频序列帧，只取图像质量为good的case。默认为心尖4腔室视频<br />**评测步骤：**

1. 训练分割模型，获取最佳权重，例如.../best.pt
2. 修改/home/kemove/zyq/giit/cardiac/CardiacAnalysis/configs/model_configs.py，更新评测模型对应使用的权重文件
3. 修改/home/kemove/zyq/giit/cardiac/CardiacAnalysis/eval_model/eval_model.py的待评测模型名称model_name
4. 切虚拟环境：conda activate py311-yolo
5. python /home/kemove/zyq/giit/cardiac/CardiacAnalysis/eval_model/eval_model.py

**评价指标：** 根据ef预测值误差比例作为模型性能指标，分析流水线：分割模型做左心室分割--->连通区域基于cv2拟合椭圆--->计算ef预测值。各模型统一下游处理流程，**仅变更分割模块**

<a name="G69Hh"></a>
### Unet
<a name="phHBe"></a>
#### 描述
unet网络学习：[https://zhuanlan.zhihu.com/p/44958351](https://zhuanlan.zhihu.com/p/44958351)<br />unet-keras版本：[https://github.com/gabrielbaltazarmw2/CAMUS-dataset](https://github.com/gabrielbaltazarmw2/CAMUS-dataset)<br />使用unet作为医学图像分割模型进行测试，训练数据使用camus数据集的心尖4腔室图像，模型在验证集上的精度为90.3%。实测使用单心尖4腔/单心尖2腔或是混合数据，以unet作为分割模型的精度均在90%左右。<br />训练代码：[http://10.10.20.106:8888/notebooks/CAMUS-dataset/UNET_CAMUS.ipynb](http://10.10.20.106:8888/notebooks/CAMUS-dataset/UNET_CAMUS.ipynb)

<a name="X1gkH"></a>
#### 实验
基于unet-camus数据集训练，实验结果：<br />误差低于5%的数据有: 30.0%<br />误差低于10%的数据有: 55.00000000000001%<br />误差低于15%的数据有: 74.23076923076923%<br />误差低于20%的数据有: 86.15384615384616%

<a name="HPPEa"></a>
#### 总结
优点：实验效果不错<br />缺点：泛化性差，模型只支持一个尺寸的图像输入，其他尺寸需要修改模型结构，重新训练
<a name="iuVYt"></a>
### YOLOV8
<a name="Zaolt"></a>
#### 描述
模型结构修改参考：[https://blog.csdn.net/weixin_62371528/article/details/136353061](https://blog.csdn.net/weixin_62371528/article/details/136353061)<br />各层解析：[https://blog.csdn.net/MacWx/article/details/135389379](https://blog.csdn.net/MacWx/article/details/135389379)

<a name="zdlWS"></a>
#### 实验
使用数据集：

- unet-camus：/home/kemove/zyq/giit/dataset/camus-datasets-yolo/4ch
- large-camus：/home/kemove/zyq/giit/cardiac/yolo/datasets/camus/partition820
<a name="YFICW"></a>
##### 非预训练版本多模型结构对比
| **模型名** | **描述** | **实验结果** | **备注** |
| --- | --- | --- | --- |
| yolov8-origin-scale-n | yolov8原生模型，无pretrained-weight，scale参数设置为n | 误差低于5%的数据有: 7.307692307692308%<br />误差低于10%的数据有: 14.23076923076923%<br />误差低于15%的数据有: 23.46153846153846%<br />误差低于20%的数据有: 30.76923076923077% | <br /> |
| yolov8-origin-scale-x | yolov8原生模型，无pretrained-weight，scale参数设置为x | 误差低于5%的数据有: 6.923076923076923%<br />误差低于10%的数据有: 13.076923076923078%<br />误差低于15%的数据有: 18.076923076923077%<br />误差低于20%的数据有: 22.30769230769231% | <br /> |
| yolov8-unetencoder-maxpool2d-scale-n | 在原yolov8结构第三个卷积层后添加一个unet编码模块，带maxpool2d层(剔除冗余特征，加速推理，减少过拟合)。scale参数设置为n | 误差低于5%的数据有: 6.153846153846154%<br />误差低于10%的数据有: 11.538461538461538%<br />误差低于15%的数据有: 16.538461538461537%<br />误差低于20%的数据有: 21.153846153846153% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-unet-encoder-maxpool.yaml |
| yolov8-unetencoder-no-maxpool2d-scale-n | 在原yolov8结构最后两个c2f层前添加unet编码模块，不带maxpool2d层，只加unet的卷积模块，提取特征。scale参数设置为n | 误差低于5%的数据有: 3.4615384615384617%<br />误差低于10%的数据有: 4.615384615384616%<br />误差低于15%的数据有: 10.0%<br />误差低于20%的数据有: 14.615384615384617% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-unet-encoder-no-maxpool.yaml |
| yolov8-unetencoder-maxpool2d-scale-x | 使用yolov8x scale，unet模块配置同上 | **误差低于5%的数据有: 24.23076923076923%**<br />误差低于10%的数据有: 40.0%<br />误差低于15%的数据有: 53.46153846153846%<br />误差低于20%的数据有: 58.84615384615385% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-unet-encoder-maxpool.yaml |
| yolov8-focus-c3str | 增加focus层，替换c2f为基于transformer的c3str层 | 1.基于备注2所训练的模型，训练设置：(训练=1443，测试=363.图像尺寸不固定，epoch=500)<br />误差低于5%的数据有: 20.384615384615383%<br />误差低于10%的数据有: 38.07692307692307%<br />**误差低于15%的数据有: 55.769230769230774%**<br />误差低于20%的数据有: 68.07692307692308% | 基于网上博主提供的目标识别模型的backbone部分，原博主称此架构改造效果较好，在2个基于camus生成的数据集上进行训练：<br />1.训练=675，测试=225. 图像尺寸=384*384，epoch=100，无法收敛，map值很低<br />2.训练=1443，测试=363.图像尺寸不固定，epoch=500，可以收敛<br />基于实验结果来看，1不收敛的原因可能是：训练数据量太少，无法收敛拟合<br />/home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-c3str.yaml |
| yolov8-focus-c3str-unet-encoder | 增加focus层，替换c2f为基于transformer的c3str层，加入net特征提取模块 | 基于large-camus训练<br />误差低于5%的数据有: 20.76923076923077%<br />**误差低于10%的数据有: 40.76923076923077%**<br />误差低于15%的数据有: 54.61538461538461%<br />**误差低于20%的数据有: 69.23076923076923%** | 1.基于unet-camus数据训练的模型在推理时报错，考虑忽略，使用large-camus再训练，测试一次<br /><br />/home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-c3str-unet-encoder.yaml |
| yolov8-gam-attention | yolov8的backbone中加入gam-attention层 | / | 加入attention后，训练指标为nan，推测可能是训练数据量不够或者epoch不够或无法收敛导致。可进一步做实验确认原因<br />/home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-focus-gam.yaml |

> 总结

1. 非预训练版本的各改进模型结构整体表现均较差，说明从头训练模型的方式不可取。尽可能在预训练好的模型权重基础上进行微调。例如使用官方的yolov8x.pt
2. 自定义的unet的特征提取模块具备一定正向收益
   1. 相较于未预训练的原生yolov8，加入unet的编码器结构后效果有提升不少(yolov8-origin-scale-x和yolov8-unetencoder-maxpool2d-scale-x)，说明unet的encoder模块在特征提取阶段有一定正向作用
   2. focus-c3str改进架构在unet-camus数据集上训练模型时均存在问题，出现无法收敛、推理报错的问题，而在large-camus上训练后正常，推测原因是训练数据集少，无法收敛拟合。所以控制变量，统一在large-camus上进行训练，测试ef指标计算性能。从结果上看，yolov8-focus-c3str-unet-encoder和yolov8-focus-3str差异点在多了一个unet-encoder特征提取模块，带unet编码模块的结构效果好于不带的。同样反映出unet的encoder模块在特征提取阶段有一定正向作用

<a name="OFdmX"></a>
##### 预训练版本多模型结构对比
| **模型名** | **描述** | **实验结果** | **备注** |
| --- | --- | --- | --- |
| yolov8-pretrained | yolov8原生模型+pretrained-weight(yolov8x-seg.pt) | **误差低于5%的数据有: 32.30769230769231%**<br />误差低于10%的数据有: 51.92307692307693%<br />误差低于15%的数据有: 70.38461538461539%<br />误差低于20%的数据有: 83.07692307692308% | yolov8-pretrained |
| yolov8-unetencoder-maxpool2d-scale-x-pretrained | 使用yolov8x scale，添加unet特征提取模块 | 误差低于5%的数据有: 25.0%<br />误差低于10%的数据有: 50.0%<br />误差低于15%的数据有: 68.46153846153847%<br />误差低于20%的数据有: 83.07692307692308% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-unet-encoder-maxpool.yaml |
| yolov8-focus-c3str-unet-encoder | 增加focus层，替换c2f为基于transformer的c3str层，加入net特征提取模块. **pretrained版本. **amp=False | 误差低于5%的数据有: 14.23076923076923%<br />误差低于10%的数据有: 30.384615384615383%<br />误差低于15%的数据有: 49.23076923076923%<br />误差低于20%的数据有: 62.30769230769231% | 基于large-camus训练<br />/home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-c3str-unet-encoder.yaml |
| yolov8-focus-c3str-pretrained | 增加focus层，替换c2f为基于transformer的c3str层，**pretrained版本** | 误差低于5%的数据有: 24.23076923076923%<br />误差低于10%的数据有: 43.46153846153846%<br />误差低于15%的数据有: 59.23076923076923%<br />误差低于20%的数据有: 73.84615384615385% | 基于large-camus训练<br /><br />/home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-c3str.yaml |
| yolov8s_GAM_seg | 有预训练,head对17,21,25层进行cat，amp=False。加入GAM-attention层，**pretrained版本** | 误差低于5%的数据有: 26.53846153846154%<br />误差低于10%的数据有: 52.69230769230769%<br />误差低于15%的数据有: 69.61538461538461%<br />误差低于20%的数据有: 78.46153846153847% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8s_GAM_seg.yaml<br /> |
| Yolov8s_swin_seg | 有预训练，替换backbone，amp=False。**pretrained版本** | 误差低于5%的数据有: 30.384615384615383%<br />误差低于10%的数据有: 51.53846153846153%<br />误差低于15%的数据有: 67.3076923076923%<br />误差低于20%的数据有: 78.46153846153847% |  /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8s_swin_seg.yaml  |
| yolov8x_swin_seg | **pretrained版本**。amp=True | 误差低于5%的数据有: 29.230769230769234%<br />误差低于10%的数据有: 50.0%<br />误差低于15%的数据有: 66.53846153846153%<br />误差低于20%的数据有: 77.3076923076923% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_swin_seg.yaml |
| yolov8x_swin_seg | **pretrained版本.** amp=False | 误差低于5%的数据有: 28.846153846153843%<br />误差低于10%的数据有: 55.00000000000001%<br />**误差低于15%的数据有: 71.53846153846153%**<br />误差低于20%的数据有: 82.6923076923077% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_swin_seg.yaml |
| yolov8x_c2_seg | **pretrained版本. **amp=False | 误差低于5%的数据有: 29.615384615384617%<br />误差低于10%的数据有: 52.307692307692314%<br />误差低于15%的数据有: 67.6923076923077%<br />误差低于20%的数据有: 81.15384615384616% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_c2_seg.yaml |
| yolov8x_c2_DW_seg | **pretrained版本. **amp=False | 误差低于5%的数据有: 30.0%<br />误差低于10%的数据有: 55.38461538461539%<br />误差低于15%的数据有: 70.76923076923077%<br />误差低于20%的数据有: 83.84615384615385% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_c2_DW_seg.yaml<br /><br />博文作者提到dwconv是减少精度，提升训练速度。而实测其最终误差小于仅改c2层的模型 |
| yolov8x-unet-encoder-focus-swin-sppc-seg | **pretrained版本. **amp=False | 误差低于5%的数据有: 30.76923076923077%<br />误差低于10%的数据有: 48.46153846153846%<br />误差低于15%的数据有: 66.92307692307692%<br />误差低于20%的数据有: 80.76923076923077% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-unet-encoder-maxpool-swin.yaml |
| yolov8x-focus-swin-sppc-seg | **pretrained版本. **amp=False | 误差低于5%的数据有: 31.92307692307692%<br />误差低于10%的数据有: 51.92307692307693%<br />误差低于15%的数据有: 66.53846153846153%<br />误差低于20%的数据有: 78.46153846153847% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-c3str-sppc-c2.yaml<br /><br />低于5%误差的数据相较于其他改动版本模型达到最优，但仍低于原生yolov8-pretrained版本 |
| yolov8x-p6-trans-c2-seg | **pretrained版本.** amp=False | 误差低于5%的数据有: 21.153846153846153%<br />误差低于10%的数据有: 39.61538461538461%<br />误差低于15%的数据有: 54.230769230769226%<br />误差低于20%的数据有: 64.23076923076924% | /home/kemove/zyq/giit/cardiac/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8x-seg-p6-trans-c3str-c2.yaml<br /><br />p6架构官方说是使用的预测特征图为 P3, P4, P5, P6，增加对大目标的分割能力，实测在本数据集上val效果不佳，不记录实验结果 |
| Yolov8xswin_seg | **pretrained版本.**amp=false | 误差低于5%的数据有: 29.230769230769234%<br />误差低于10%的数据有: 50.0%<br />误差低于15%的数据有: 66.53846153846153%<br />误差低于20%的数据有: 77.3076923076923% | backbone是swintransformer block，之前的swin是transformer block<br />/home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8xswin_seg.yaml<br /> |
| yolov8x_DW_C3TR_focus_seg | <br /> | 误差低于5%的数据有: 28.846153846153843%<br />误差低于10%的数据有: 54.61538461538461%<br />误差低于15%的数据有: 67.6923076923077%<br />误差低于20%的数据有: 80.38461538461539% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_DW_C3TR_focus_seg.yaml |
| yolov8x_DW_swin_FOCUS-3_seg | <br /> | 误差低于5%的数据有: 31.538461538461537%<br />**误差低于10%的数据有: 58.07692307692308%**<br />**误差低于15%的数据有: 73.84615384615385%**<br />**误差低于20%的数据有: 85.76923076923076%** | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_DW_swin_FOCUS-3_seg.yaml |
| yolov8x_DW_swin_FOCUS2_seg | <br /> | 误差低于5%的数据有: 30.0%<br />误差低于10%的数据有: 52.307692307692314%<br />误差低于15%的数据有: 70.0%<br />误差低于20%的数据有: 83.07692307692308% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_DW_swin_FOCUS2_seg.yaml |
| yolov8x_DW_FOCUS_seg.yaml | <br /> | 误差低于5%的数据有: 28.076923076923077%<br />误差低于10%的数据有: 55.38461538461539%<br />误差低于15%的数据有: 71.53846153846153%<br />误差低于20%的数据有: 82.3076923076923% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_DW_FOCUS_seg.yaml |
| yolov8x_DW_FOCUS_sppc_seg.yaml | <br /> | 误差低于5%的数据有: 25.769230769230766%<br />误差低于10%的数据有: 52.69230769230769%<br />误差低于15%的数据有: 69.23076923076923%<br />误差低于20%的数据有: 81.15384615384616% | /home/kemove/hyx/ultralytics_STF/ultralytics/cfg/models/v8/yolov8_DW_FOCUS_sppc_seg.yaml |

> 总结

1. amp=False能够提高最终分析结果的精度。看yolov8x_swin_seg(amp=False)和yolov8x_swin_seg(amp=True)的实验结果
2. yolov8x_c2_DW_seg和yolov8x_swin_seg两个改进结构相较于原生yolov8x-pretrained版本，在不同数据分布上有一定提升，分析两个模型结构：
   1. yolov8x_c2_DW_seg修改的是卷积层和head的c2f层。即它的改动主要在分割结果输出部分
   2. yolov8x_swin_seg修改的是backbone的结构。即它的改动主要在特征提取部分，仅增加了transformer块
3. 从实验结果上看，模型结构的少量改动相较于纯堆砌层可能更具优化效果
<a name="ZagGU"></a>
#### 总结
评测结果以心脏指标ef分析误差作为最终评判结果，与真值比较，计算偏差。由于整个步骤中除了图像分割，还包含一系列下游处理逻辑，反映的误差并非代表改进模型的真实分割效果。


<a name="uQ34L"></a>
### SAM-Med2D
<a name="grX68"></a>
#### 描述
meta所开源的SAM模型目前是图像分割领域的一个研究进展，通过交互式提示（如：点和边界框）分割区域，做到自动化分割。而直接将预训练好的SAM应用于医学图像分割效果不佳，原因是自然图像和医学图像有差异。因此，以SAM模型作为基座进行医学影像微得到SAM-Med2D模型，且开源了一个医学图像分割数据集名为SAM-Med2D-20M。

<a name="dm6M4"></a>
#### 实验
sam-med2d虽然已是基于SAM模型，经过医学影像数据集微调后的模型，但由于其预训练的数据集中没有心脏相关的数据集，所以本身的预训练权重评测效果较差，因此通过unet-camus数据集微调后参与评测，实验结果：<br />误差低于5%的数据有: 29.615384615384617%<br />误差低于10%的数据有: 58.46153846153847%<br />误差低于15%的数据有: 73.84615384615385%<br />误差低于20%的数据有: 83.07692307692308%

<a name="WYhGZ"></a>
#### 总结
优点：泛化性较强，在少量数据上微调即可得到较好的实验结果<br />缺点：交互式分割模型，若不经过微调，则需要对图像做坐标点标记后才能分割
<a name="SqdJo"></a>
### VM-UnetV2
<a name="m1Ytt"></a>
#### 描述
unet的变体模型，github链接：[https://github.com/nobodyplayer1/VM-UNetV2](https://github.com/nobodyplayer1/VM-UNetV2)

<a name="le9Cq"></a>
#### 实验

1. 训练情况
   1. 2024-06-26 19:41:58 - test of best model, loss: 0.1203,miou: 0.8754462393086319, f1_or_dsc: 0.9335871335148035, accuracy: 0.9878949351369599。miou指标偏低
   2. 配置文件：/home/kemove/zyq/giit/cardiac/CardiacAnalysis/model/VM_UNetV2/configs/config_setting_v2.py
2. 实验结果

误差低于5%的数据有: 27.307692307692307%<br />误差低于10%的数据有: 46.15384615384615%<br />误差低于15%的数据有: 64.61538461538461%<br />误差低于20%的数据有: 77.6923076923077%

<a name="fJSfr"></a>
#### 总结
模型可复现，但在专有数据集上表现不佳



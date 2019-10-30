[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## 项目概述

欢迎来到卷积神经网络（CNN）项目！在这一项目中，你将学到如何建立一个处理现实生活中的，用户提供的图像的算法。给你一个狗的图像，你的算法将会识别并估计狗的品种，如果提供的图像是人，代码将会识别最相近的狗的品种。

![Sample Output][image1]

在学习用于分类的最先进的 CNN 模型的同时，你将会为用户体验做出重要的设计与决定。我们的目标是，当你完成这一项目时，你将可以理解，通过将一系列模型拼接在一起，设计数据处理管道完成各式各样的任务所面临的挑战。每个模型都有它的优点与缺点，并且设计实际应用时，经常会面对解决许多没有最优解的问题。尽管你的解答不是最优的，但你的设计将带来愉快的用户体验！


## 项目指南

### 步骤

1. 克隆存储库并打开下载的文件夹。


2. 下载[狗狗数据集](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/dogImages.zip) ，并将数据集解压大存储库中，地点为`项目路径/dogImages`. 

3. 下载[人类数据集](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/lfw.zip)。并将数据集解压大存储库中，位置为`项目路径/lfw `。

4. 为狗狗数据集下载 [VGG-16关键特征](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/DogVGG16Data.npz) 并将其放置于存储库中，位置为`项目路径/bottleneck_features `。

5. 安装必要的 Python 依赖包


opencv-python
h5py
matplotlib
numpy
scipy
tqdm
keras
scikit-learn
pillow
ipykernel
tensorflow


## 常见坑和问题
* 由于使用已有的模型需要下载对应的权重模型数据集 如碰到需要下载
* xxxxx_weights_tf_dim_ordering_tf_kernels_notop.h5格式的文件时
* 可通过复制下来对应的下载链接到迅雷下载可快速下载下来 并将其保存在自定义的路径下 
* 在加载对应模型时直接传入对应路径的地址就行
* 多种模型结构详细解析请看：https://blog.csdn.net/u011746554/article/details/74394211
* 模型权重.h5文件的官方下载地址：https://github.com/fchollet/deep-learning-models/releases


xxxxData.npz文件可百度自行下载


* VGG-19 bottleneck features  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz
* ResNet-50 bottleneck features  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz
* Inception bottleneck features  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz
* Xception bottleneck features   https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz




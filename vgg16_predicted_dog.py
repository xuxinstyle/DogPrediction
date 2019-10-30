from extract_bottleneck_features import *
from glob import glob
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import tensorflow as tf


def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
def predict_breed(img_path):
    from keras.applications.vgg16 import VGG16, preprocess_input
    tensor = path_to_tensor(img_path)
    predicted_index = VGG16_model.predict(preprocess_input(tensor))
    return dog_names[np.argmax(predicted_index)]
result = VGG16_predict_breed('static/img/timg.jpg')
X = tf.placeholder(tf.string, name = 'input')
Y = tf.nn.softmax(tf.matmul(VGG16_predict_breed, out_weights))

result1 = predict_breed('static/img/timg.jpg')
print(result+"\n")
print(result1)
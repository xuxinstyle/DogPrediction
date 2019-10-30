from extract_bottleneck_features import *
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import tensorflow as tf
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')
def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']
# Keras中的数据预处理过程
# train_tensors = paths_to_tensor(train_files).astype('float32')/255
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
# test_tensors = paths_to_tensor(test_files).astype('float32')/255
InceptionV3_model = Sequential()
#InceptionV3_model.add(MaxPooling2D(input_shape=train_InceptionV3.shape[1:]))
#InceptionV3_model.add(GlobalAveragePooling2D(input_shape=(224, 224, 3)))
InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
# InceptionV3_model.add(Flatten())
InceptionV3_model.add(Dense(133, activation='softmax'))

InceptionV3_model.summary()

## 编译模型

InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

## 训练模型

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5',
                               verbose=1, save_best_only=True)
# 减小BATCH_SIZE的数量，同时增加EPOCH的数量，可以提高最终的准确率，但是训练的时间也就相应的加长了；
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
InceptionV3_model.fit(train_InceptionV3, train_targets,
          validation_data=(valid_InceptionV3, valid_targets),
          epochs=16, batch_size=64, callbacks=[checkpointer], verbose=1)

## 加载具有最好验证loss的模型
converter = tf.lite.TFLiteConverter.from_keras_model(InceptionV3_model)
tflite_model = converter.convert()
open("tensorflow_lite/tflite_model.tflite", "wb").write(tflite_model)
InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
# 获取测试数据集中每一个图像所预测的狗品种的index
InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

# 报告测试准确率

test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

def InceptionV3_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# result1 = predict_breed('static/img/timg.jpg')
result = InceptionV3_predict_breed('static/img/timg.jpg')
print("result:"+result+"\n")
# print("result1:" + result1)
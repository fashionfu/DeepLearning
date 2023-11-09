# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task2_VGG16_Apple.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/15 18:55 
'''
#去掉AVX warning
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 任务：根据original_data样本，建立模型，对test_data的图片进行普通/其他苹果判断
# 1.数据增强，扩充确认为普通苹果的样本数量
# 2.特征提取，使用VGG16模型提取图像特征
# 3.图片批量处理
# 4.Kmeans模型尝试普通、其他苹果聚类
# 5.基于标签数据矫正结果，并可视化
# 6.Meanshift模型提升模型表现
# 7.数据降维PCA处理，提升模型表现

#数据增强
from keras.preprocessing.image import ImageDataGenerator
path='original_data'#图片加强的文件路径
dst_path='gen_data'#设置存储路径

#创建示例完成图像增强的配置
datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.02,
    horizontal_flip=True,
    vertical_flip=True
)

#设置图像增强后图像的路径
gen=datagen.flow_from_directory(
    path,#此处要在path路径中的文件夹下建立子文件夹，每个子文件夹代表一个类别，等价于打标签
    target_size=(224,224),
    batch_size=2,#每个批次可以产生多少张图片
    save_to_dir=dst_path,
    save_prefix='gen',#意为生成的图像
    save_format='jpg'
)

#这一步只需要进行一次后，在gen_data文件夹中就会产生200张图像增强后的图片了
# for i in range(100):
#     gen.next()

#====================进行单张图片的特征提取====================
#load the image(single image='1.jpg')
from keras.utils.image_utils import load_img,img_to_array
img_path='1.jpg'
img=load_img(img_path,target_size=(224,224))#224×224图像用于VGG16模型应用来进行特征提取
# print(type(img))#<class 'PIL.Image.Image'>

#visualize the image
from matplotlib import pyplot as plt
fig1=plt.figure(figsize=(5,5))
plt.imshow(img)
# plt.show()

img=img_to_array(img)#将图像转换为矩阵，方便输入到VGG模型中进行特征提取
# print(img.shape)#(224, 224, 3)
# print(type(img))#<class 'numpy.ndarray'>

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
model_vgg=VGG16(weights='imagenet',include_top=False)
X=np.expand_dims(img,axis=0)#将X[0]扩展，表示为图片的张数
X=preprocess_input(X)#进行数组的预处理，把它转换成VGG16可以使用的格式
# print(X.shape)#(1, 224, 224, 3)

#特征提取
features=model_vgg.predict(X)#此时使用的model_vgg模型已经去除了输出层，提取出来的就是对应所需特征
# print(features.shape)#(1, 7, 7, 512)

#flatten
features=features.reshape(1, 7 * 7 * 512 )
# print(features.shape)#(1, 25088)
#====================完成单张图片的特征提取====================

#list all the names of the data
import os
folder='train_data'
dirs=os.listdir(folder)
# print(dirs)#['10.jpg', '11.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg', '17.jpg', '18.jpg', '19.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg', '36_100.jpg', '38_100.jpg', '3_100.jpg', '4-0.jpg', '4-1.jpg', '4-2.jpg', '4-3.jpg', '4-4.jpg', '4-5.jpg', '4-6.jpg', '4-7.jpg', '4-8.jpg', '4-9.jpg', '8.jpg', '9.jpg', 'gen_0_1173818.jpg', 'gen_0_1939178.jpg', 'gen_0_2389907.jpg', 'gen_0_2757481.jpg', 'gen_0_3482860.jpg', 'gen_0_413611.jpg', 'gen_0_4759760.jpg', 'gen_0_589938.jpg', 'gen_0_7095525.jpg', 'gen_0_7490538.jpg', 'gen_0_8022296.jpg', 'gen_0_8076560.jpg', 'gen_0_8153913.jpg', 'gen_0_8171627.jpg', 'gen_0_8176886.jpg', 'gen_0_8374543.jpg', 'gen_0_8560193.jpg', 'gen_0_8628425.jpg', 'gen_0_865515.jpg', 'gen_0_9536397.jpg', 'gen_1_1177085.jpg', 'gen_1_200464.jpg', 'gen_1_2735359.jpg', 'gen_1_3152310.jpg', 'gen_1_3489309.jpg', 'gen_1_3628492.jpg', 'gen_1_3874934.jpg', 'gen_1_4610773.jpg', 'gen_1_5004692.jpg', 'gen_1_5089191.jpg', 'gen_1_5506123.jpg', 'gen_1_5537016.jpg', 'gen_1_6614229.jpg', 'gen_1_6687675.jpg', 'gen_1_8121142.jpg', 'gen_1_8199107.jpg', 'gen_1_94237.jpg', 'gen_1_9879536.jpg', 'gen_1_9958411.jpg', 'gen_1_9974622.jpg', 'gen_2_1025646.jpg', 'gen_2_1114055.jpg', 'gen_2_1295036.jpg', 'gen_2_1449151.jpg', 'gen_2_1563279.jpg', 'gen_2_2388955.jpg', 'gen_2_3689068.jpg', 'gen_2_3705174.jpg', 'gen_2_3729669.jpg', 'gen_2_427808.jpg', 'gen_2_4339819.jpg', 'gen_2_5145490.jpg', 'gen_2_6865321.jpg', 'gen_2_7347490.jpg', 'gen_2_7719855.jpg', 'gen_2_8159591.jpg', 'gen_2_865058.jpg', 'gen_2_901014.jpg', 'gen_2_9107562.jpg', 'gen_2_9797151.jpg', 'gen_3_1637940.jpg', 'gen_3_205126.jpg', 'gen_3_2660446.jpg', 'gen_3_273910.jpg', 'gen_3_2875244.jpg', 'gen_3_3489460.jpg', 'gen_3_3637757.jpg', 'gen_3_4880160.jpg', 'gen_3_5006202.jpg', 'gen_3_5823241.jpg', 'gen_3_5856796.jpg', 'gen_3_6023118.jpg', 'gen_3_6640131.jpg', 'gen_3_7063140.jpg', 'gen_3_8185280.jpg', 'gen_3_8618547.jpg', 'gen_3_8774167.jpg', 'gen_3_9015167.jpg', 'gen_3_9672745.jpg', 'gen_3_9749246.jpg', 'gen_4_1530550.jpg', 'gen_4_1595157.jpg', 'gen_4_1754719.jpg', 'gen_4_1835666.jpg', 'gen_4_247090.jpg', 'gen_4_3540178.jpg', 'gen_4_3638993.jpg', 'gen_4_5280779.jpg', 'gen_4_6026647.jpg', 'gen_4_6297558.jpg', 'gen_4_6523481.jpg', 'gen_4_6547423.jpg', 'gen_4_697927.jpg', 'gen_4_70659.jpg', 'gen_4_7124008.jpg', 'gen_4_7352881.jpg', 'gen_4_8956156.jpg', 'gen_4_897614.jpg', 'gen_4_9276022.jpg', 'gen_4_9912020.jpg', 'gen_5_1333224.jpg', 'gen_5_1521602.jpg', 'gen_5_204051.jpg', 'gen_5_2234335.jpg', 'gen_5_227189.jpg', 'gen_5_2289544.jpg', 'gen_5_2348096.jpg', 'gen_5_2778348.jpg', 'gen_5_2829005.jpg', 'gen_5_2842957.jpg', 'gen_5_3251289.jpg', 'gen_5_3712172.jpg', 'gen_5_4424607.jpg', 'gen_5_4785401.jpg', 'gen_5_5028503.jpg', 'gen_5_6542455.jpg', 'gen_5_6943692.jpg', 'gen_5_7165283.jpg', 'gen_5_7358288.jpg', 'gen_5_8578063.jpg', 'gen_6_1183460.jpg', 'gen_6_1644590.jpg', 'gen_6_1832113.jpg', 'gen_6_2375269.jpg', 'gen_6_2395520.jpg', 'gen_6_2663293.jpg', 'gen_6_2928206.jpg', 'gen_6_2996849.jpg', 'gen_6_3008926.jpg', 'gen_6_4123922.jpg', 'gen_6_4715816.jpg', 'gen_6_5451189.jpg', 'gen_6_5812659.jpg', 'gen_6_6064641.jpg', 'gen_6_6999312.jpg', 'gen_6_7269044.jpg', 'gen_6_913437.jpg', 'gen_6_9145962.jpg', 'gen_6_9259116.jpg', 'gen_6_9574245.jpg', 'gen_7_1191212.jpg', 'gen_7_1593784.jpg', 'gen_7_2031208.jpg', 'gen_7_2766501.jpg', 'gen_7_2827720.jpg', 'gen_7_3431326.jpg', 'gen_7_3489226.jpg', 'gen_7_3752436.jpg', 'gen_7_4103112.jpg', 'gen_7_6082080.jpg', 'gen_7_6181207.jpg', 'gen_7_7441899.jpg', 'gen_7_854805.jpg', 'gen_7_857795.jpg', 'gen_7_8635501.jpg', 'gen_7_8968343.jpg', 'gen_7_9536330.jpg', 'gen_7_9555974.jpg', 'gen_7_9560045.jpg', 'gen_7_9838799.jpg', 'gen_8_1412646.jpg', 'gen_8_1460151.jpg', 'gen_8_1844965.jpg', 'gen_8_1870731.jpg', 'gen_8_1932998.jpg', 'gen_8_3577061.jpg', 'gen_8_4626841.jpg', 'gen_8_4746710.jpg', 'gen_8_4752320.jpg', 'gen_8_5823938.jpg', 'gen_8_621841.jpg', 'gen_8_6278845.jpg', 'gen_8_6331058.jpg', 'gen_8_6348273.jpg', 'gen_8_6514581.jpg', 'gen_8_7271910.jpg', 'gen_8_7837449.jpg', 'gen_8_8092634.jpg', 'gen_8_9129815.jpg', 'gen_8_9200743.jpg', 'gen_9_1343246.jpg', 'gen_9_2554013.jpg', 'gen_9_3073875.jpg', 'gen_9_3475555.jpg', 'gen_9_362950.jpg', 'gen_9_3870657.jpg', 'gen_9_4771425.jpg', 'gen_9_4829784.jpg', 'gen_9_5441774.jpg', 'gen_9_5577948.jpg', 'gen_9_5583312.jpg', 'gen_9_5599161.jpg', 'gen_9_676781.jpg', 'gen_9_6898632.jpg', 'gen_9_7820564.jpg', 'gen_9_8155447.jpg', 'gen_9_8191524.jpg', 'gen_9_9033414.jpg', 'gen_9_919808.jpg', 'gen_9_9261162.jpg']
#名称合并
img_path=[]#创建一个空list来存储名称
for i in dirs:
    if os.path.splitext(i)[1]=='.jpg':#确保后缀名为.jpg
        img_path.append(i)
img_path=[folder + "//" + i for i in img_path]#进行字符串衔接
# print(img_path)#['train_data//10.jpg', 'train_data//11.jpg', 'train_data//13.jpg', 'train_data//14.jpg', 'train_data//15.jpg', 'train_data//16.jpg', 'train_data//17.jpg', 'train_data//18.jpg', 'train_data//19.jpg', 'train_data//20.jpg', 'train_data//21.jpg', 'train_data//22.jpg', 'train_data//23.jpg', 'train_data//24.jpg', 'train_data//25.jpg', 'train_data//36_100.jpg', 'train_data//38_100.jpg', 'train_data//3_100.jpg', 'train_data//4-0.jpg', 'train_data//4-1.jpg', 'train_data//4-2.jpg', 'train_data//4-3.jpg', 'train_data//4-4.jpg', 'train_data//4-5.jpg', 'train_data//4-6.jpg', 'train_data//4-7.jpg', 'train_data//4-8.jpg', 'train_data//4-9.jpg', 'train_data//8.jpg', 'train_data//9.jpg', 'train_data//gen_0_1173818.jpg', 'train_data//gen_0_1939178.jpg', 'train_data//gen_0_2389907.jpg', 'train_data//gen_0_2757481.jpg', 'train_data//gen_0_3482860.jpg', 'train_data//gen_0_413611.jpg', 'train_data//gen_0_4759760.jpg', 'train_data//gen_0_589938.jpg', 'train_data//gen_0_7095525.jpg', 'train_data//gen_0_7490538.jpg', 'train_data//gen_0_8022296.jpg', 'train_data//gen_0_8076560.jpg', 'train_data//gen_0_8153913.jpg', 'train_data//gen_0_8171627.jpg', 'train_data//gen_0_8176886.jpg', 'train_data//gen_0_8374543.jpg', 'train_data//gen_0_8560193.jpg', 'train_data//gen_0_8628425.jpg', 'train_data//gen_0_865515.jpg', 'train_data//gen_0_9536397.jpg', 'train_data//gen_1_1177085.jpg', 'train_data//gen_1_200464.jpg', 'train_data//gen_1_2735359.jpg', 'train_data//gen_1_3152310.jpg', 'train_data//gen_1_3489309.jpg', 'train_data//gen_1_3628492.jpg', 'train_data//gen_1_3874934.jpg', 'train_data//gen_1_4610773.jpg', 'train_data//gen_1_5004692.jpg', 'train_data//gen_1_5089191.jpg', 'train_data//gen_1_5506123.jpg', 'train_data//gen_1_5537016.jpg', 'train_data//gen_1_6614229.jpg', 'train_data//gen_1_6687675.jpg', 'train_data//gen_1_8121142.jpg', 'train_data//gen_1_8199107.jpg', 'train_data//gen_1_94237.jpg', 'train_data//gen_1_9879536.jpg', 'train_data//gen_1_9958411.jpg', 'train_data//gen_1_9974622.jpg', 'train_data//gen_2_1025646.jpg', 'train_data//gen_2_1114055.jpg', 'train_data//gen_2_1295036.jpg', 'train_data//gen_2_1449151.jpg', 'train_data//gen_2_1563279.jpg', 'train_data//gen_2_2388955.jpg', 'train_data//gen_2_3689068.jpg', 'train_data//gen_2_3705174.jpg', 'train_data//gen_2_3729669.jpg', 'train_data//gen_2_427808.jpg', 'train_data//gen_2_4339819.jpg', 'train_data//gen_2_5145490.jpg', 'train_data//gen_2_6865321.jpg', 'train_data//gen_2_7347490.jpg', 'train_data//gen_2_7719855.jpg', 'train_data//gen_2_8159591.jpg', 'train_data//gen_2_865058.jpg', 'train_data//gen_2_901014.jpg', 'train_data//gen_2_9107562.jpg', 'train_data//gen_2_9797151.jpg', 'train_data//gen_3_1637940.jpg', 'train_data//gen_3_205126.jpg', 'train_data//gen_3_2660446.jpg', 'train_data//gen_3_273910.jpg', 'train_data//gen_3_2875244.jpg', 'train_data//gen_3_3489460.jpg', 'train_data//gen_3_3637757.jpg', 'train_data//gen_3_4880160.jpg', 'train_data//gen_3_5006202.jpg', 'train_data//gen_3_5823241.jpg', 'train_data//gen_3_5856796.jpg', 'train_data//gen_3_6023118.jpg', 'train_data//gen_3_6640131.jpg', 'train_data//gen_3_7063140.jpg', 'train_data//gen_3_8185280.jpg', 'train_data//gen_3_8618547.jpg', 'train_data//gen_3_8774167.jpg', 'train_data//gen_3_9015167.jpg', 'train_data//gen_3_9672745.jpg', 'train_data//gen_3_9749246.jpg', 'train_data//gen_4_1530550.jpg', 'train_data//gen_4_1595157.jpg', 'train_data//gen_4_1754719.jpg', 'train_data//gen_4_1835666.jpg', 'train_data//gen_4_247090.jpg', 'train_data//gen_4_3540178.jpg', 'train_data//gen_4_3638993.jpg', 'train_data//gen_4_5280779.jpg', 'train_data//gen_4_6026647.jpg', 'train_data//gen_4_6297558.jpg', 'train_data//gen_4_6523481.jpg', 'train_data//gen_4_6547423.jpg', 'train_data//gen_4_697927.jpg', 'train_data//gen_4_70659.jpg', 'train_data//gen_4_7124008.jpg', 'train_data//gen_4_7352881.jpg', 'train_data//gen_4_8956156.jpg', 'train_data//gen_4_897614.jpg', 'train_data//gen_4_9276022.jpg', 'train_data//gen_4_9912020.jpg', 'train_data//gen_5_1333224.jpg', 'train_data//gen_5_1521602.jpg', 'train_data//gen_5_204051.jpg', 'train_data//gen_5_2234335.jpg', 'train_data//gen_5_227189.jpg', 'train_data//gen_5_2289544.jpg', 'train_data//gen_5_2348096.jpg', 'train_data//gen_5_2778348.jpg', 'train_data//gen_5_2829005.jpg', 'train_data//gen_5_2842957.jpg', 'train_data//gen_5_3251289.jpg', 'train_data//gen_5_3712172.jpg', 'train_data//gen_5_4424607.jpg', 'train_data//gen_5_4785401.jpg', 'train_data//gen_5_5028503.jpg', 'train_data//gen_5_6542455.jpg', 'train_data//gen_5_6943692.jpg', 'train_data//gen_5_7165283.jpg', 'train_data//gen_5_7358288.jpg', 'train_data//gen_5_8578063.jpg', 'train_data//gen_6_1183460.jpg', 'train_data//gen_6_1644590.jpg', 'train_data//gen_6_1832113.jpg', 'train_data//gen_6_2375269.jpg', 'train_data//gen_6_2395520.jpg', 'train_data//gen_6_2663293.jpg', 'train_data//gen_6_2928206.jpg', 'train_data//gen_6_2996849.jpg', 'train_data//gen_6_3008926.jpg', 'train_data//gen_6_4123922.jpg', 'train_data//gen_6_4715816.jpg', 'train_data//gen_6_5451189.jpg', 'train_data//gen_6_5812659.jpg', 'train_data//gen_6_6064641.jpg', 'train_data//gen_6_6999312.jpg', 'train_data//gen_6_7269044.jpg', 'train_data//gen_6_913437.jpg', 'train_data//gen_6_9145962.jpg', 'train_data//gen_6_9259116.jpg', 'train_data//gen_6_9574245.jpg', 'train_data//gen_7_1191212.jpg', 'train_data//gen_7_1593784.jpg', 'train_data//gen_7_2031208.jpg', 'train_data//gen_7_2766501.jpg', 'train_data//gen_7_2827720.jpg', 'train_data//gen_7_3431326.jpg', 'train_data//gen_7_3489226.jpg', 'train_data//gen_7_3752436.jpg', 'train_data//gen_7_4103112.jpg', 'train_data//gen_7_6082080.jpg', 'train_data//gen_7_6181207.jpg', 'train_data//gen_7_7441899.jpg', 'train_data//gen_7_854805.jpg', 'train_data//gen_7_857795.jpg', 'train_data//gen_7_8635501.jpg', 'train_data//gen_7_8968343.jpg', 'train_data//gen_7_9536330.jpg', 'train_data//gen_7_9555974.jpg', 'train_data//gen_7_9560045.jpg', 'train_data//gen_7_9838799.jpg', 'train_data//gen_8_1412646.jpg', 'train_data//gen_8_1460151.jpg', 'train_data//gen_8_1844965.jpg', 'train_data//gen_8_1870731.jpg', 'train_data//gen_8_1932998.jpg', 'train_data//gen_8_3577061.jpg', 'train_data//gen_8_4626841.jpg', 'train_data//gen_8_4746710.jpg', 'train_data//gen_8_4752320.jpg', 'train_data//gen_8_5823938.jpg', 'train_data//gen_8_621841.jpg', 'train_data//gen_8_6278845.jpg', 'train_data//gen_8_6331058.jpg', 'train_data//gen_8_6348273.jpg', 'train_data//gen_8_6514581.jpg', 'train_data//gen_8_7271910.jpg', 'train_data//gen_8_7837449.jpg', 'train_data//gen_8_8092634.jpg', 'train_data//gen_8_9129815.jpg', 'train_data//gen_8_9200743.jpg', 'train_data//gen_9_1343246.jpg', 'train_data//gen_9_2554013.jpg', 'train_data//gen_9_3073875.jpg', 'train_data//gen_9_3475555.jpg', 'train_data//gen_9_362950.jpg', 'train_data//gen_9_3870657.jpg', 'train_data//gen_9_4771425.jpg', 'train_data//gen_9_4829784.jpg', 'train_data//gen_9_5441774.jpg', 'train_data//gen_9_5577948.jpg', 'train_data//gen_9_5583312.jpg', 'train_data//gen_9_5599161.jpg', 'train_data//gen_9_676781.jpg', 'train_data//gen_9_6898632.jpg', 'train_data//gen_9_7820564.jpg', 'train_data//gen_9_8155447.jpg', 'train_data//gen_9_8191524.jpg', 'train_data//gen_9_9033414.jpg', 'train_data//gen_9_919808.jpg', 'train_data//gen_9_9261162.jpg']

#define the method to extract the features
def modelProcess(img_path,model):
    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    X=np.expand_dims(img,axis=0)
    X=preprocess_input(X)#对图像数组进行预处理，以便应用于VGG16对图像进行特征提取
    X_VGG = model.predict(X)#得到去除vgg-16模型输出层后的全部所需特征（全连接层前）
    X_VGG=X_VGG.reshape(1,7*7*512)
    return X_VGG#返回图形的特征

#图像批量处理
features_train=np.zeros([len(img_path),7*7*512])
for i in range(len(img_path)):
    features_i=modelProcess(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    features_train[i]=features_i
# print(features_train[0])
# print(features_train.shape)#(230, 25088)

#define X; 30 ➡ 10普通苹果 ➡ +200普通苹果，希望以数量多的为一类（普通，210），其他为一类（其他）
#希望以众数来矫正无监督式聚类模型的结果
X=features_train

#set up Kmeans unsupervised model
from sklearn.cluster import KMeans
cnn_kmeans = KMeans(n_clusters=2,max_iter=2000)#最多迭代次数不超过2000
cnn_kmeans.fit(X)

#make prediction
y_predict_kmeans=cnn_kmeans.predict(X)
print(y_predict_kmeans)

from collections import Counter
print(Counter(y_predict_kmeans))#预测结果统计：Counter({1: 120, 0: 110})，非常差的模型结果❌

normal_apple_id=1
#visualize the result
fig2=plt.figure(figsize=(10,40))
for i in range(45):
    for j in range(5):
        img=load_img(img_path[i*5+j])#read the image
        plt.subplot(45,5,i*5+j+1)
        plt.title('apple' if y_predict_kmeans[i*5+j] == normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
# plt.show()

#测试数据集的读取
import os
folder_test='test_data'
dirs_test=os.listdir(folder_test)
# print(dirs_test)
#名称合并
img_path_test=[]#创建一个空list来存储名称
for i in dirs_test:
    if os.path.splitext(i)[1]=='.jpg':#确保后缀名为.jpg
        img_path_test.append(i)
img_path_test=[folder_test + "//" + i for i in img_path_test]#进行字符串衔接
# print(img_path_test)#['test_data//1.jpg', 'test_data//3.jpg', 'test_data//xx_0_1133520.jpg', 'test_data//xx_12_5963907.jpg', 'test_data//xx_14_6969294.jpg', 'test_data//xx_1_586816.jpg', 'test_data//xx_2_1588073.jpg', 'test_data//xx_3_3378731.jpg', 'test_data//xx_5_4481562.jpg', 'test_data//xx_7_4698934.jpg', 'test_data//xx_8_5910781.jpg', 'test_data//xx_9_3279522.jpg']

#图像批量处理——test
features_test=np.zeros([len(img_path_test),7*7*512])
for i in range(len(img_path_test)):
    features_i=modelProcess(img_path_test[i],model_vgg)
    print('preprocessed:',img_path_test[i])
    features_test[i]=features_i
X_test=features_test
# print(features_test.shape,X_test.shape)#(12, 25088) (12, 25088)

y_predict_kmeans_test=cnn_kmeans.predict(X_test)
print(y_predict_kmeans_test)

fig3=plt.figure(figsize=(10,40))
for i in range(3):
    for j in range(4):
        img=load_img(img_path_test[i*4+j])#read the image
        plt.subplot(3,4,i*4+j+1)
        plt.title('apple' if y_predict_kmeans_test[i*4+j] == normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
# plt.show()

#因为采用Kmeans聚类出来的结果不好，考虑采用meanshift模型进行聚类
from sklearn.cluster import MeanShift,estimate_bandwidth#自动找出期望的范围宽度
#obtain the bandwidth
bw=estimate_bandwidth(X,n_samples=140)
# print(bw)#1324.8336121551652

#set up meanshift model
cnn_meanshift=MeanShift(bandwidth=bw)
cnn_meanshift.fit(X)

#make prediction
y_predict_ms=cnn_meanshift.predict(X)
print(y_predict_ms)
# [ 8  4  9  0  5  0  3  6  0  0  2  0  7  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  1 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#此时证明将多数正确的普通苹果分为了一类，其他每个自成一类，模型结果较好
print(Counter(y_predict_ms))#Counter({0: 220, 8: 1, 4: 1, 9: 1, 5: 1, 3: 1, 6: 1, 2: 1, 7: 1, 1: 1, 10: 1})

normal_apple_id=0
fig4=plt.figure(figsize=(10,40))
for i in range(45):
    for j in range(5):
        img=load_img(img_path[i*5+j])#read the image
        plt.subplot(45,5,i*5+j+1)
        plt.title('apple' if y_predict_ms[i*5+j] == normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
# plt.show()

y_predict_ms_test=cnn_meanshift.predict(X_test)

fig5=plt.figure(figsize=(10,40))
for i in range(3):
    for j in range(4):
        img=load_img(img_path_test[i*4+j])
        plt.subplot(3,4,i*4+j+1)
        plt.title('apple'if y_predict_ms_test[i*4+j]==normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
# plt.show()

#PCA降维,考虑从数据开始下手，加强预处理(对于整个数据集进行处理)
from sklearn.preprocessing import StandardScaler
stds=StandardScaler()
X_norm=stds.fit_transform(X)
#PCA analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=200)
X_pca=pca.fit_transform(X_norm)
print(X_pca.shape,X.shape)#(230, 200) (230, 25088)

#calculate the variance ratio of each components
var_ratio=pca.explained_variance_ratio_
print(np.sum(var_ratio))#0.9841585272943283

#obtain the new bandwidth
bw_pca=estimate_bandwidth(X_pca,n_samples=140)

#set up meanshift new model
cnn_pca_meanshift=MeanShift(bandwidth=bw_pca)
cnn_pca_meanshift.fit(X_pca)

y_predict_ms_pca=cnn_pca_meanshift.predict(X_pca)
print(Counter(y_predict_ms_pca))

fig6=plt.figure(figsize=(10,40))
for i in range(45):
    for j in range(5):
        img=load_img(img_path[i*5+j])
        plt.subplot(45,5,i*5+j+1)
        plt.title('apple'if y_predict_ms_pca[i*5+j]==normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
# plt.show()

X_norm_test=stds.transform(X_test)
X_pca_test=pca.transform(X_norm_test)
y_predict_ms_pca_test=cnn_pca_meanshift.predict(X_pca_test)

fig7=plt.figure(figsize=(10,40))
for i in range(3):
    for j in range(4):
        img=load_img(img_path_test[i*4+j])
        plt.subplot(3,4,i*4+j+1)
        plt.title('apple'if y_predict_ms_pca_test[i*4+j]==normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
plt.show()

# 普通/其他苹果检测实战summary:
# 1.通过搭建混合模型，实现了监督+无监督、机器+深度学习技术的有机结合，并在少样本情况下建立起了有效区分普通苹果与其他苹果的模型
# 2.针对少样本任务，掌握了生成新数据的数据增强方法
# 3.更熟练的掌握了拆分经典模型VGG16模型并用于提取图像特征的方法
# 4.完成了图像的批量处理
# 5.回顾了无监督聚类算法：Kmeans、Meanshift，并通过标签数据分布实现数据类别矫正
# 6.成功引入PCA数据降维技术，剔除了数据中的噪音信息、降低了模型复杂度、减少了模型训练时间，并最终提高了模型表现
# 7.可以考虑一些其他方法去完成任务，比如异常数据检测技术

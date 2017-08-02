## 最有趣？
机器学习是不是很无聊，用来用去都是识别字体。能不能帮我找到颜值高的妹子，顺便提高一下姿势水平。

FaceRank 基于 TensorFlow CNN 模型，提供了一些图片处理的工具集，后续还会提供训练好的模型。给 FaceRank 一个妹子，他给你个分数。

从此以后筛选简历，先把头像颜值低的去掉；自动寻找女主颜值高的小电影；自动关注美女；自动排除负分滚粗的相亲对象。从此以后升职加薪，迎娶白富美，走上人生巅峰。

苍老师镇楼：

![1cbf16b28aa949acadeeff4398829328_th.jpg](http://upload-images.jianshu.io/upload_images/76451-2deffc054e0e3452.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 项目开源：
GitHub:https://github.com/fendouai/FaceRank

## 依赖库：
* Tensorflow
安装：pip install tensorflow
简介：Tensorflow 是谷歌的机器学习框架，FaceRank 使用了基于它的 CNN 模型。
http://www.tensorflownews.com/2017/07/28/installing-tensorflow-tensorflow/
* face_recognition
简介：这个库在项目中，用来从图片中截出人脸，并保存为新文件，方便生成数据集。
这个库比较难装，如果直接安装失败，建议使用 docker.
The world's simplest facial recognition api for Python and the command line
安装：pip install face_recognition

## 训练数据集生成工具
* 文件夹截图
![fileinfo.png](http://upload-images.jianshu.io/upload_images/76451-dbe34a6ab015b96d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 标注说明 
文件夹中  1-2.jpg 表明这是 1分的图片，2是第2张。也就是 “-”前面的数字就是分数。

*  find_faces_in_picture.py
这个脚本使用了  face_recognition 来扣人脸，它会从  上图中的 web_image 读取图片，抠图之后保存到 face_image 文件夹。

* resize_image.py
这个脚本会读取  face_image 文件夹，并将图片统一处理为 128*128像素。

## 训练
一切都准备好了，直接运行 train_model.py
这部分内容在 Github 有比较详细说明：
https://github.com/fendouai/FaceRank/

## 模型使用
* FaceRank 内置了模型保存功能，训练之后，以后都可以直接运行  run_model.py 。也就是可以封装成函数或者类库使用，非常方便。

## 学习流程
如果看到这里有很多不懂的话，建议：
*  Hello World
https://zhuanlan.zhihu.com/p/27963600
*  基本概念
https://zhuanlan.zhihu.com/p/27986689
*  卷积神经网络
https://zhuanlan.zhihu.com/p/28161292
*  训练好模型参数的保存和恢复代码
https://zhuanlan.zhihu.com/p/27912379
* TensorFlowNews 专栏
https://zhuanlan.zhihu.com/TensorFlownews
*  TensorFlowNews 博客
http://www.tensorflownews.com/

欢迎关注我的博客，因为我也还在学习中，现有的教程经常比较大，涉及到的只是比较多，我会经常拆分出小的知识点，我的博客也会把这些小的知识点记录下来。
FaceRank，带你走进 TensorFlow 的世界。

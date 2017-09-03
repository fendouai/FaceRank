## FaceRank-人脸打分基于 TensorFlow 的 CNN 模型

## 结果图片
如有侵权，请通知删除，结果由 FaceRank AI 输出。
![Result Pic](https://github.com/fendouai/FaceRank/blob/master/cang.jpg)

## 隐私
因为隐私问题，训练图片集并不提供，稍微可能会放一些卡通图片。


## 数据集
* 130张 128*128 张网络图片，图片名： 1-3.jpg 表示 分值为 1 的第 3 张图。
你可以把符合这个格式的图片放在 resize_images 来训练模型。

## 模型
人脸打分基于 TensorFlow 的 CNN 模型 代码参考 : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

## 运行
安装好 TensorFlow 之后，直接运行 train_model.py.
* 训练模型
* 保存模型到 model 文件夹

## 测试
运行完 train_model.py 之后,直接运行 run_model.py 来测试.

## 下载
训练好的模型可以在以下网址下载：
http://www.tensorflownews.com/

## 模型效果
* 训练过程
你可以看训练过程： Train_Result.md ,这里有损失函数和准确率变化过程。
* 测试结果
结果并不非常好，但是增加数据集之后有所改善。

```
(?, 128, 128, 24)
(?, 64, 64, 24)
(?, 64, 64, 96)
(?, 32, 32, 96)

['1-1.jpg', '1-2.jpg', '10-1.jpg', '10-2.jpg', '2-1.jpg', '2-2.jpg', '3-1.jpg', '3-2.jpg', '4-1.jpg', '4-2.jpg', '5-1.jpg', '5-2.jpg', '6-1.jpg', '6-2.jpg', '7-1.jpg', '7-2.jpg', '8-1.jpg', '8-2.jpg', '9-1.jpg', '9-2.jpg']
20
(10, 128, 128, 3)
[3 2 8 6 5 8 0 4 7 7]
(10, 128, 128, 3)
[2 6 6 6 5 8 7 8 7 5]
Test Finished!
```
## 支持
* 提交 issue
* QQ 群: 522785813
* 知乎:https://zhuanlan.zhihu.com/TensorFlownews
* 博客:http://www.tensorflownews.com/

##后续计划
* 图片像素要提高
* 增加数据集
* 在临近的层次，用公用的图片：比如1-3；4-6；7-9 用相似或者相同图片。

## 微信群：
![Result Pic](https://github.com/fendouai/FaceRank/blob/master/wechatgroup.jpg)

如果二维码过期，请到这里 http://www.tensorflownews.com/ 会保持更新。

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from PIL import Image
import os

list = os.listdir("./resize_image/")
print(list)
print(len(list))
for batch_id in range(1, 10):
	batch = list[batch_id * 10:batch_id * 10 + 10]
	batch_xs=[]
	batch_ys=[]
	for image in batch:
		id_tag = image.find("-")
		score = image[0:id_tag]
		# print(score)
		img = Image.open("./resize_image/" + image)
		img_ndarray = numpy.asarray(img, dtype='float32')
		img_ndarray = numpy.reshape(img_ndarray, [128, 128, 3])
		# print(img_ndarray.shape)
		batch_x = img_ndarray
		batch_xs.append(batch_x)
		#print(batch_xs)
		batch_y = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		# print(type(score))
		batch_y[int(score) - 1] = 1
		# print(batch_y)
		batch_y = numpy.reshape(batch_y, [10,])
		batch_ys.append(batch_y)
		#print(batch_ys)
	batch_xs=numpy.asarray(batch_xs)
	print(batch_xs.shape)
	batch_ys = numpy.asarray(batch_ys)
	print(batch_ys.shape)
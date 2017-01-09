# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:52:41 2016

@author: zhouc
"""
# library imports
import caffe
import cv2
import numpy as np
# local imports
from style import StyleTransfer
caffe.set_mode_gpu()
style_img_path = 'images/style/the_scream.jpg'
content_img_path = 'images/content/tubingen.jpg'
img_style = caffe.io.load_image(style_img_path)
img_content = caffe.io.load_image(content_img_path)
args = {"length": 600, "ratio": 2e5, "n_iter": 32, "init": "content"}
st = StyleTransfer()
st.transfer_style(img_style, img_content, **args)
img_out = st.get_generated()
# show the image
cv2.imshow("Style", cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyWindow("Style")
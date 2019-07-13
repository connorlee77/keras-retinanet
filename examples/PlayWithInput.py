from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

sample_path = 'pictures'
image = read_image_bgr(os.path.join(sample_path, '1.png'))

A = 0
incident = 1.0
transmitted = 1.1
tau = 2
new_image = np.uint8(image*np.exp(-tau) + A*(1-np.exp(-tau)))
print(new_image.shape)
plt.figure()
plt.axis('off')
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

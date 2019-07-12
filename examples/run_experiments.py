import keras
from keras import backend as K

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def detect_classify(image):

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    detections = {}
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        name = labels_to_names[label]
        if name not in detections:
            detections[name] = [(score, box)]
        else:
            detections[name].append([(score, box)])

    return image, detections

def dfdtau(grad_wrt_input, I, A, tau):
    dIdtau = (A - I)*np.exp(-tau)
    return np.multiply(grad_wrt_input, dIdtau).sum()

def dfdA(grad_wrt_input, I, A, tau):
    dIdA = 1 - np.exp(-tau)
    return np.multiply(grad_wrt_input, dIdA).sum()

def getGradients(image, A, tau):

    # preprocess image for network
    image = preprocess_image(image)
    processed_image, scale = resize_image(image)

    print(processed_image) 
    print(processed_image.shape) 
    
    grad_wrt_input = iterate([np.expand_dims(processed_image, axis=0)])
    grad_wrt_input = np.squeeze(grad_wrt_input)

    grad_wrt_tau = dfdtau(grad_wrt_input, processed_image, A, tau)
    grad_wrt_A = dfdA(grad_wrt_input, processed_image, A, tau)

    return processed_image, grad_wrt_tau, grad_wrt_A


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
# ## Load RetinaNet model

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
sample_path = 'pictures'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)
# print(model.summary())

# Get symbolic gradient
gradient = keras.backend.gradients(model.output, model.input)[0]
# create function for retrieving input-specific gradient
iterate = keras.backend.function([model.input], [gradient]) 


# Read image
filename = os.path.join(sample_path, 'porsche.jpg');
image = read_image_bgr(filename)
tau = 0
A = 0 
for i in range(10):
    ### Process images
    image, detections = detect_classify(image)
    # processed_image, grad_wrt_tau, grad_wrt_A = getGradients(image, A, tau)
    print(detections)
    tau += 0.1
    image = image*np.exp(-tau) + A*(1 - np.exp(-tau))



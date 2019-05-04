# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('logos', 'snapshots', 'resnet50_csv_160.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = {1: 'Ferrari-Pict', 2: 'FC_Barcelona', 3: 'Adidas-Pict', 4: 'Audi-Pict', 5: 'Adidas-Text', 6: 'Allianz-Pict', 7: 'Aldi', 8: 'Burger_king', 9: 'Audi-Text', 10: 'eBay', 11: 'Allianz-Text', 12: 'CocaCola', 13: 'Atletico_Madrid', 14: 'Facebook-Text', 15: 'FC_Bayern_Munchen', 16: 'Apple', 17: 'Ferrari-Text', 18: 'BMW', 19: 'Facebook-Pict', 20: 'Amazon'}

# load image
image = read_image_bgr('logos/images/apple.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    print(box)
    print(score)
    print(labels_to_names[label])
    
    # scores are sorted so we can break
    if score < 0.6:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(7, 7))
plt.axis('off')
plt.imshow(draw)
plt.show()
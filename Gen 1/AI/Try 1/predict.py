import cv2
import tensorflow as tf
import os
from keras.preprocessing import image
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

CATEGORIES = ["no", "yes"]

folder_path = r'C:/Users/mihir/Desktop/Mihir/Multiple Sclerosis AI/AI/pred/yes/2/'

IMG_SIZE = 150

images = []

for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    images.append(img)

images = np.vstack(images)

model = tf.keras.models.load_model("C:/Users/mihir/Desktop/Mihir/Multiple Sclerosis AI/AI/Try 1/Model.h5")

images = images/255.0

prediction = model.predict(images, batch_size=32)
print(prediction)  # will be a list in a list.
with open('yes_2.txt', 'w') as f:
    for item in prediction:
        f.write("%s\n" % item)
import cv2
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

CATEGORIES = ["no", "yes"]

folder_path = r'C:/Users/mihir/Desktop/Mihir/Multiple Sclerosis AI/AI/pred/yes/3/1825.jpg'

def prepare(filepath):
    IMG_SIZE = 150  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("C:/Users/mihir/Desktop/Mihir/Multiple Sclerosis AI/AI/Try 1/Model.h5")

prediction = model.predict([prepare(folder_path)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
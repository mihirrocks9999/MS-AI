import numpy
import os
from tabnanny import verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

def decode(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    height, width, depth = 181, 217, 181

    features = tf.io.parse_single_example(
        serialized_example,
        features={'train/image': tf.io.FixedLenFeature([height, width, depth, 1], tf.float32),
                  'train/label': tf.io.FixedLenFeature([], tf.int64)})

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['train/image'], features['train/label']

def predict(printPredict, batch_size, prefetch_size):
    model = tf.keras.models.load_model("Gen 2/Code/CNN/Custom/bestClassification.h5")
    print("Starting Prediction")

    length = sum(1 for _ in tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/testset.tfrecords"))
    print("Length is: " + str(length))
    # Define data loaders.
    full_dataset = tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/testset.tfrecords").map(decode)
    test_dataset = (
            full_dataset.shuffle(length)
            .batch(batch_size)
            .prefetch(prefetch_size)
        )

    prediction = model.predict(test_dataset, verbose=1)
    results = []
    i = 0
    for next_element in test_dataset:
        text = next_element[1].numpy()
        label = text[0]
        label = "Actual Value: " + str(label)
        results.append([prediction[i][0], label])
        i = i + 1
    results = numpy.array(results)
    if printPredict:
        print(results)
    accuracy = 0
    for num in results:
        renum = num[1]
        renum = renum.replace('Actual Value: ','')
        renum = float(renum)
        acnum = float(num[0])
        temp = abs(renum-acnum)
        accuracy = accuracy + temp
    accuracy = float(accuracy/(length/batch_size))
    print("Predicition Accuracy is: " + str(accuracy))
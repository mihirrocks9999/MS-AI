import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import random
import nibabel as nib
from scipy import ndimage

def createTF(noMS_dir, MS_dir, height, width, depth):
    normal_scan_paths = [
        os.path.join(os.getcwd(), noMS_dir, x)
        for x in os.listdir(noMS_dir)
        ]
    abnormal_scan_paths = [
        os.path.join(os.getcwd(), MS_dir, x)
        for x in os.listdir(MS_dir)
        ]
    scan_paths = normal_scan_paths + abnormal_scan_paths

    print("MRI scans with no Multiple Sclerosis: " + str(len(normal_scan_paths)))
    print("MRI scans with Multiple Sclerosis: " + str(len(abnormal_scan_paths)))
    print("Height is " + str(height) + ". " + "Width is " + str(width) + ". " + "Depth is " + str(depth) + "." )

    # path to save the TFRecords file
    train_filename = "Gen 2/Code/CNN/Custom/dataset.tfrecords"

    # open the file
    writer = tf.io.TFRecordWriter(train_filename)

    # iterate through all .mnc files:
    for scan in scan_paths:

        # Load the image and label
        img = nib.load(scan).get_fdata()
        img = rotate(img)
        label = 2
        if "noMS" in scan:
            label = 0
        else:
            label = 1
    
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _float_feature(img.ravel())}
               
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()

def createTestTF(noMS_dir, MS_dir, height, width, depth):
    normal_scan_paths = [
        os.path.join(os.getcwd(), noMS_dir, x)
        for x in os.listdir(noMS_dir)
        ]
    abnormal_scan_paths = [
        os.path.join(os.getcwd(), MS_dir, x)
        for x in os.listdir(MS_dir)
        ]
    scan_paths = normal_scan_paths + abnormal_scan_paths

    print("MRI scans with no Multiple Sclerosis: " + str(len(normal_scan_paths)))
    print("MRI scans with Multiple Sclerosis: " + str(len(abnormal_scan_paths)))
    print("Height is " + str(height) + ". " + "Width is " + str(width) + ". " + "Depth is " + str(depth) + "." )

    # path to save the TFRecords file
    train_filename = "Gen 2/Code/CNN/Custom/testset.tfrecords"

    # open the file
    writer = tf.io.TFRecordWriter(train_filename)

    # iterate through all .mnc files:
    for scan in scan_paths:

        # Load the image and label
        img = nib.load(scan).get_fdata()
        img = rotate(img)
        label = 2
        if "noMS" in scan:
            label = 0
        else:
            label = 1
    
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _float_feature(img.ravel())}
               
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        return volume
    augmented_volume = scipy_rotate(volume)
    #augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
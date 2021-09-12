import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob

# Constants
SEED = 139
SEP = os.path.sep
IMAGE_SIZE = 256
MODEL_PATH = "generator.h5"

# Set seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Data Path
DATA_PATH = "input_data"
pre_image_path = os.path.join(DATA_PATH, "pre_disaster_images")
targets_path = os.path.join(DATA_PATH, "targets")

pre_images_names = sorted([path.split("/")[-1]
                           for path in glob.glob(os.path.join(pre_image_path, "*.png"))])
targets_names = sorted([path.split("/")[-1]
                        for path in glob.glob(os.path.join(targets_path, "*.png"))])

# Load the model
gan_one = tf.keras.models.load_model(MODEL_PATH)

# Defining functions




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

'''Constants'''
SEED = 139
SEP = os.path.sep
IMAGE_SIZE = 256
MODEL_PATH = "generator.h5"
OUTPUT_PATH = "output_images"

'''Set seeds'''
tf.random.set_seed(SEED)
np.random.seed(SEED)

'''Data Path'''
DATA_PATH = "input_data"
pre_image_path = os.path.join(DATA_PATH, "pre_disaster_images")
targets_path = os.path.join(DATA_PATH, "targets")

pre_images_names = sorted([path.split("/")[-1]
                           for path in glob.glob(os.path.join(pre_image_path, "*.png"))])
targets_names = sorted([path.split("/")[-1]
                        for path in glob.glob(os.path.join(targets_path, "*.png"))])

'''Load the model'''
gan_one = tf.keras.models.load_model(MODEL_PATH)

'''Create output folder'''
if not os.path.exists("output_images"):
    os.makedirs("output_images")
    os.makedirs("output_images/images")
    os.makedirs("output_images/images/level_1")
    os.makedirs("output_images/images/level_2")
    os.makedirs("output_images/images/level_3")
    os.makedirs("output_images/images/level_4")
    os.makedirs("output_images/targets")
    os.makedirs("output_images/targets/level_1")
    os.makedirs("output_images/targets/level_2")
    os.makedirs("output_images/targets/level_3")
    os.makedirs("output_images/targets/level_4")


'''Define functions'''


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UnsupportedTypeError(Error):
    """Exception raised for unsupported types.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class UnsupportedLevelError(Error):
    """Exception raised for unsupported types.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def change_destruction_level(image, destruction_level):
    """
    Changes the destruction level of a pre_disaster mask to create a post_disaster_mask.

    Parameters
    ----------
    image : numpy.ndarray or Tensorflow Tensor
        Contains the image of the pre disaster mask
    destruction_level : str or int
        Level of destruction to be created from the pre disaster mask

    Returns
    -------
    numpy.ndarray or Tensorflow Tensor
        A Tensor or numpy array to signify a post disaster mask

    Raises
    ------
    UnsupportedLevelError
        Raised when the user provides a destruction level that isn't supported
    UnsupportedTypeError
        Raised when the user provides the destruction level that isn't a string or an integer
        Raised when the user provides an image that isn't a numpy array or a Tensorflow Tensor
    """

    destruction_levels = {
        "no damage": 1,
        "minor": 2,
        "major": 3,
        "destroyed": 4
    }

    if isinstance(destruction_level, str):
        try:
            destruction_level = destuction_levels[destruction_level]
        except KeyError:
            raise UnsupportedLevelError(
                f"{destruction_level} is not a supported destruction level. The only supported string levels are 'no damage', 'minor', 'major', and 'destroyed'.")
    elif isinstance(destruction_level, int):
        if destruction_level > 4 or destruction_level < 1:
            raise UnsupportedLevelError(
                f"{destruction_level} is not a supported destruction level. The only supported integer levels are 1, 2, 3, and 4.")
    else:
        raise UnsupportedTypeError(
            "Destruction level must be a string or an integer.")

    if isinstance(image, np.ndarray):
        return np.where(image != 0, destruction_level, image).astype(np.float32)
    elif tf.is_tensor(image):
        return tf.cast(tf.where(image != 0, destruction_level, image), tf.float32)
    else:
        raise UnsupportedTypeError(
            "This function only supports Tensors and numpy.ndarray objects.")


'''Resize image function'''


def resize(input_image, image_size=IMAGE_SIZE):
    input_image = tf.image.resize(input_image, [
                                  image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


'''Normalize images [0,254] to [-1,+1]'''


def normalize(input_image):
    input_image = (input_image/127.5) - 1

    return input_image


'''Augmentation of data: Random Crop + flip'''


def random_jitter(input_image):
    input_image = resize(input_image, 286)

    stacked_image = tf.stack([input_image, input_image], axis=0)

    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMAGE_SIZE, IMAGE_SIZE, 4])

    input_image = cropped_image[0]

    return input_image


'''Load the images'''


def load_image(filename, augment=True, destruction_level=0):
    '''
     input1 := Pre-image
     input2 := Mask post-image

     target := Post image
    '''

    filename_input1 = filename
    filename_input2 = filename.split(".png")[0] + "_target.png"

    pre_image = tf.cast(tf.image.decode_png(tf.io.read_file(
        pre_image_path + SEP + filename_input1), channels=3), tf.float32)
    mask_image = tf.cast(tf.image.decode_png(tf.io.read_file(
        targets_path + SEP + filename_input2), channels=1), tf.float32)

    modified_mask = change_destruction_level(mask_image, destruction_level)
    new_input = tf.experimental.numpy.dstack((modified_mask, pre_image))

    new_input = resize(new_input, IMAGE_SIZE)

    if augment:
        new_input = random_jitter(new_input)

    new_input = normalize(new_input)

    pre_image = resize(pre_image, IMAGE_SIZE)
    mask_image = resize(mask_image, IMAGE_SIZE)

    return pre_image, mask_image, modified_mask, new_input


'''Generate the new data'''


def create_new_data(train_pre_images_names, train_masks_names, model=gan_one, num_of_images=100, height=512, width=512):
    for i in range(num_of_images):
        # Get the respective names of the images
        pre_image_name = train_pre_images_names[i]
        post_image_name = pre_image_name.replace("post", "pre")

        post_mask_name = train_masks_names[i]

        # Dictionary
        scaling_values = {1: 255, 2: 127.5, 3: 85, 4: 63.75}

        for j in range(1, 4 + 1):
            # Load the pre disaster images and post disaster mask
            _, _, modified_mask, new_input = load_image(
                filename=pre_image_name, destruction_level=j)

            # Call the model to generate the post disaster image
            generated_post = model(tf.expand_dims(
                new_input, axis=0), training=True)

            # Resize the images to heightxwidth
            modified_mask_resized = tf.image.resize(
                modified_mask,  [height, width], method="area").numpy()
            generated_post_resized = tf.image.resize(
                generated_post, [height, width], method="area").numpy()

            post_image_dest = os.path.join(OUTPUT_PATH, "images", f"level_{j}")
            targets_dest = os.path.join(OUTPUT_PATH, "targets", f"level_{j}")

            plt.imsave(os.path.join(targets_dest, f'{post_mask_name}_level_{j}.png'), tf.squeeze(
                modified_mask_resized)/scaling_values[j], format="png", cmap="gray", vmax=j)

            plt.imsave(os.path.join(post_image_dest, f'{post_image_name}_level_{j}.png'), tf.squeeze(
                generated_post_resized).numpy()*0.5 + 0.5, format="png")


'''Main Function'''


def main():
    create_new_data(pre_images_names, targets_names)


if __name__ == "__main__":
    main()

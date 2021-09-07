import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class GanOne:
    def __init__(self, data_folder_path):
        self.__IMAGE_SIZE = 256
        self.__OUTPUT_CHANNELS = 3
        self.__LAMBDA = 100

        self.__loss_function = BinaryCrossentropy(from_logits=True)
        self.__generator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
        self.__discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

        self.__generator_losses = list()
        self.__discriminator_losses = list()

        # Train Paths
        self.__pre_image_path_train = os.path.join(
            data_folder_path, "train_pre/images")
        self.__masks_path_train = os.path.join(
            data_folder_path, "train_post/targets")
        self.__post_image_path_train = os.path.join(
            data_folder_path, "train_post/images")

        # Test Paths
        self.__pre_image_path_test = os.path.join(
            data_folder_path, "test_pre/images")
        self.__masks_path_test = os.path.join(
            data_folder_path, "test_post/targets")
        self.__post_image_path_test = os.path.join(
            data_folder_path, "test_post/images")

        # Train Filenames
        self.__train_pre_image_names = sorted([filepath.split(
            "/")[-1] for filepath in glob.glob(os.path.join(data_folder_path, "train_pre/images/*.png"))])
        self.__train_masks_names = sorted([filepath.split("/")[-1] for filepath in glob.glob(
            os.path.join(data_folder_path, "train_post/targets/*.png"))])
        self.__train_post_images_names = sorted([filepath.split(
            "/")[-1] for filepath in glob.glob(os.path.join(data_folder_path, "train_post/targets/*.png"))])

        # Test Filenames
        self.__test_pre_image_names = sorted([filepath.split(
            "/")[-1] for filepath in glob.glob(os.path.join(data_folder_path, "test_pre/images/*.png"))])
        self.__test_masks_names = sorted([filepath.split("/")[-1] for filepath in glob.glob(
            os.path.join(data_folder_path, "test_post/targets/*.png"))])
        self.__test_post_images_names = sorted([filepath.split(
            "/")[-1] for filepath in glob.glob(os.path.join(data_folder_path, "test_post/targets/*.png"))])

        self.gen = self.generator()
        self.disc = self.discriminator()

    def __resize(self, input_image, target_image, image_size=256):
        input_image = tf.image.resize(input_image, [
                                      image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        target_image = tf.image.resize(target_image, [
                                       image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, target_image

    def __normalize(self, input_image, target_image):
        input_image = (input_image / 127.5) - 1
        target_image = (target_image / 127.5) - 1

        return input_image, target_image

    def __random_jitter(self, input_image, target_image):
        input_image, target_image = self.__resize(
            input_image, target_image, 286)

        stacked_image = tf.stack([input_image, target_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.__IMAGE_SIZE, self.__IMAGE_SIZE, 4])

        input_image, target_image = cropped_image[0], cropped_image

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            target_image = tf.image.flip_left_right(target_image)

        return input_image, target_image

    def __load_image(self, filename, augment=True, from_training=True):
        '''
        input1 := Pre-image
        input2 := Mask post-image

        target := Post image
        '''
        if len(tf.strings.split(filename, sep="pre")) == 2:
            filename_input1 = tf.strings.split(filename, sep="pre")[
                0] + 'pre' + tf.strings.split(filename, sep="pre")[-1]
            filename_target = tf.strings.split(filename, sep="pre")[
                0] + 'post' + tf.strings.split(filename, sep="pre")[-1]
            filename_input2 = tf.strings.split(filename_target, sep=".png")[
                0] + "_target.png"

        elif len(tf.strings.split(filename, sep="post")) == 2:
            filename_input1 = tf.strings.split(filename, sep="post")[
                0] + 'pre' + tf.strings.split(filename, sep="post")[-1]
            filename_target = tf.strings.split(filename, sep="post")[
                0] + 'post' + tf.strings.split(filename, sep="post")[-1]
            filename_input2 = tf.strings.split(filename_target, sep=".png")[
                0] + "_target.png"
        else:
            filename_input1 = ""
            filename_input2 = ""
            filename_target = ""

        if from_training:
            pre_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__pre_image_path_train + '/' + filename_input1), channels=3), tf.float32)
            mask_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__masks_path_train + '/' + filename_input2), channels=1), tf.float32)

            target_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__post_image_path_train + '/' + filename_target), channels=4), tf.float32)
        else:
            pre_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__pre_image_path_test + '/' + filename_input1), channels=3), tf.float32)
            mask_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__masks_path_test + '/' + filename_input2), channels=1), tf.float32)

            target_image = tf.cast(tf.image.decode_png(tf.io.read_file(
                self.__post_image_path_test + '/' + filename_target), channels=4), tf.float32)

        concat = tf.experimental.numpy.dstack((mask_image, pre_image))

        concat, target_image = self.__resize(
            concat, target_image, self.__IMAGE_SIZE)

        if augment:
            concat, target_image = self.__random_jitter(concat, target_image)

        concat, target_image = self.__normalize(concat, target_image)

        target_image, _ = tf.experimental.numpy.split(
            target_image, [3],  axis=-1
        )

        return concat, target_image

    def __load_train_image(self, filename):
        return self.__load_image(filename=filename, augment=True, from_training=True)

    def __load_test_image(self, filename):
        return self.__load_image(filename=filename, augment=False, from_training=False)

    def __downsample(self, filters, size, apply_batchnorm=True):
        init = tf.random_normal_initializer(0., 0.02)
        result = Sequential()
        result.add(Conv2D(filters=filters, kernel_size=size, strides=2,
                          padding="same", kernel_initializer=init, use_bias=False))
        if apply_batchnorm == True:
            result.add(BatchNormalization())

        result.add(LeakyReLU())
        return result

    def __upsample(self, filters, size, apply_dropout=False):
        init = tf.random_normal_initializer(0., 0.02)
        result = Sequential()
        result.add(Conv2DTranspose(filters=filters, kernel_size=size,
                                   strides=2, padding="same", kernel_initializer=init, use_bias=False))
        result.add(BatchNormalization())

        if apply_dropout == True:
            result.add(Dropout(0.5))

        result.add(ReLU())
        return result

    def generator(self):
        inputs = Input(shape=[self.__IMAGE_SIZE, self.__IMAGE_SIZE, 4])

        down_stack = [
            self.__downsample(filters=64,  size=4, apply_batchnorm=False),
            self.__downsample(filters=128, size=4),
            self.__downsample(filters=256, size=4),
            self.__downsample(filters=512, size=4),
            self.__downsample(filters=512, size=4),
            self.__downsample(filters=512, size=4),
            self.__downsample(filters=512, size=4),
            self.__downsample(filters=512, size=4)
        ]

        up_stack = [
            self.__upsample(filters=512, size=4, apply_dropout=True),
            self.__upsample(filters=512, size=4, apply_dropout=True),
            self.__upsample(filters=512, size=4, apply_dropout=True),
            self.__upsample(filters=512, size=4),
            self.__upsample(filters=256, size=4),
            self.__upsample(filters=128, size=4),
            self.__upsample(filters=64, size=4),
        ]
        init = tf.random_normal_initializer(0., 0.02)

        last = Conv2DTranspose(filters=self.__OUTPUT_CHANNELS, kernel_size=4, strides=2,
                               padding="same", kernel_initializer=init, activation="tanh")
        x = inputs

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)
        return Model(inputs=inputs, outputs=x)

    def __generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.__loss_function(tf.ones_like(
            disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.__LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator(self):
        init = tf.random_normal_initializer(0., 0.02)

        inp = Input(shape=[self.__IMAGE_SIZE,
                           self.__IMAGE_SIZE, 4], name="input_image")
        tar = Input(shape=[self.__IMAGE_SIZE,
                           self.__IMAGE_SIZE, 3], name="target_image")

        x = Concatenate()([inp, tar])

        down1 = self.__downsample(64, 4, False)(x)
        down2 = self.__downsample(128, 4)(down1)
        down3 = self.__downsample(256, 4)(down2)

        zero_pad1 = ZeroPadding2D()(down3)
        conv = Conv2D(512, 4, strides=1, kernel_initializer=init,
                      use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = LeakyReLU()(batchnorm1)
        zero_pad2 = ZeroPadding2D()(leaky_relu)
        last = Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2)
        return Model(inputs=[inp, tar], outputs=last)

    def __discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.__loss_function(tf.ones_like(
            disc_real_output), disc_real_output)
        generated_loss = self.__loss_function(tf.zeros_like(
            disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def save_images(self, model, test_input, target, epoch):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))
        display_list = [test_input[0], target[0], prediction[0]]
        title = ["Input Image (Pre-disaster + Post-Mask)",
                 "Ground Truth(Post Disaster)", "Prediction Image (Post-Disaster)"]
        for i in range(3):
            if i == 0:
                print(
                    f"test_input shape = {test_input[0].shape}\t\ttarget shape = {target[0].shape}\t\tprediction shape = {prediction[0].shape}")
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis("off")

        if epoch % 25 == 0:
            plt.savefig(
                f"drive/My Drive/Colab Notebooks/modified_data/pre_and_post_joined/output/epoch_{epoch}.png")

            plt.show()
            plt.close()

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.gen(input_image, training=True)

            disc_real_output = self.disc([input_image, target], training=True)
            disc_generated_output = self.disc(
                [input_image, gen_output], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.__generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.__discriminator_loss(
                disc_real_output, disc_generated_output)
            generator_gradients = gen_tape.gradient(
                gen_total_loss, self.gen.trainable_variables)
            discriminator_gradients = disc_tape.gradient(
                disc_loss, self.disc.trainable_variables)
            self.__generator_optimizer.apply_gradients(
                zip(generator_gradients, self.gen.trainable_variables))
            self.__discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.disc.trainable_variables))
            return gen_total_loss, disc_loss

    def fit(self, train_ds, epochs, test_ds):
        for epoch in range(1501, epochs+1):
            start = time.time()
            test_ds = test_ds.shuffle(10)
            for input_, target in test_ds.take(1):
                self.save_images(self.gen, input_, target, epoch)

            # Train
            print(f"Epoch {epoch}")
            for n, (input_, target) in train_ds.enumerate():
                gen_loss, disc_loss = self.__train_step(input_, target, epoch)

            print("Generator loss {:.2f} Discriminator loss {:.2f}".format(
                gen_loss, disc_loss))
            print("Time take for epoch {} is {} sec\n".format(
                epoch+1, time.time() - start))

            self.__generator_losses.append(gen_loss)
            self.__discriminator_losses.append(disc_loss)

            if epoch % 25 == 0:
                self.gen.save(
                    f"drive/My Drive/Colab Notebooks/modified_data/pre_and_post_joined/saved_modelsB/generator/gen_model_{epoch}.h5")
                self.disc.save(
                    f"drive/My Drive/Colab Notebooks/modified_data/pre_and_post_joined/saved_modelsB/discriminator/disc_model_{epoch}.h5")

    def predict_from_path(self, pre_image_path, mask_image_path):
        pre_image = cv2.imread(pre_image_path)
        mask_image = cv2.imread(mask_image_path)

        pre_image_resized = cv2.resize(
            pre_image, (256, 256), interpolation=cv2.INTER_AREA)
        mask_image_resized = cv2.resize(
            mask_image, (256, 256), interpolation=cv2.INTER_AREA)

        stacked_image = np.dstack((pre_image_resized, mask_image_resized))
        post_image = self.gen(stacked_image, training=True)
        post_image = tf.squeeze(post_image)

        return post_image

    def predict(self, pre_image, mask_image):
        if tf.is_tensor(pre_image):
            pre_image = pre_image.numpy()
        if tf.is_tensor(mask_image):
            mask_image = mask_image.numpy()

        pre_image_resized = cv2.resize(
            pre_image, (256, 256), interpolation=cv2.INTER_AREA)
        mask_image_resized = cv2.resize(
            mask_image, (256, 256), interpolation=cv2.INTER_AREA)

        stacked_image = np.dstack((pre_image_resized, mask_image_resized))
        post_image = self.gen(stacked_image, training=True)
        post_image = tf.squeeze(post_image)

        return post_image

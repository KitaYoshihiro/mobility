from __future__ import print_function, division

import tensorflow

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, MaxPooling2D

#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import keras.backend as K

import pickle
from hdmsgenerator import HDMSGenerator

import matplotlib.pyplot as plt
import os
import numpy as np

class SGAN:
    def __init__(self, traindata_generator):
        self.traindata_generator = traindata_generator
        self.img_rows = 192
        self.img_cols = 1024
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 5

        optimizer = tensorflow.keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # self.discriminator.compile(
        #     loss=['binary_crossentropy', 'categorical_crossentropy'],
        #     loss_weights=[0.5, 0.5], #ここのバランスを変えてみる（初期は0.5, 0.5）
        #     optimizer=optimizer,
        #     metrics=['accuracy']
        # )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        model = Model(noise, valid)
        model = tensorflow.contrib.tpu.keras_to_tpu_model(
            model, strategy=tensorflow.contrib.tpu.TPUDistributionStrategy(
                tensorflow.contrib.cluster_resolver.TPUClusterResolver(
                    tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
                    )
                )
        )
        self.combined = model
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

        self.discriminator = tensorflow.contrib.tpu.keras_to_tpu_model(
            self.discriminator, strategy=tensorflow.contrib.tpu.TPUDistributionStrategy(
                tensorflow.contrib.cluster_resolver.TPUClusterResolver(
                    tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
                    )
                )
        )
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5], #ここのバランスを変えてみる（初期は0.5, 0.5）
            optimizer=optimizer,
            metrics=['accuracy']
        )

        init = tensorflow.initialize_all_variables()


    def build_generator(self):

        model = Sequential()
        model.add(Dense(64 * 48 * 256, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((48, 256, 64)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        #model.add(Activation("tanh"))
        model.add(Activation("sigmoid"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        model = Model(noise, img)

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        inputImage = Input(shape=self.img_shape)
        
        features = model(inputImage)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)


        # img = Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same")(inputImage)
        # img = BatchNormalization(momentum=0.8)(img)
        # img = LeakyReLU(alpha=0.2)(img)
        # img = Dropout(0.25)(img)

        # img= MaxPooling2D()(img)
        # img_1 = Conv2D(32, kernel_size=3, strides=1, padding="same")(img)
        # img_1 = BatchNormalization(momentum=0.8)(img_1)
        # img = Concatenate()([img_1, img])
        # img = LeakyReLU(alpha=0.2)(img)
        # img = Dropout(0.25)(img)

        # img = MaxPooling2D()(img)
        # img_1 = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        # img_1 = BatchNormalization(momentum=0.8)(img_1)
        # img = Concatenate()([img_1, img])
        # img = LeakyReLU(alpha=0.2)(img)
        # img = Dropout(0.25)(img)

        # features = Flatten()(img)

        # valid = Dense(1, activation="sigmoid")(features)
        # pre_label = Dense(50, activation="relu")(features)
        # pre_labe2 = Dense(50, activation="relu")(pre_label)
        # label = Dense(self.num_classes+1, activation="softmax")(pre_labe2)

        model = Model(inputImage, [valid, label])
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # X_train, y_train = 

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1, 1)

        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = batch_size // 2
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            #idx = np.random.randint(0, X_train.shape[0], batch_size)
            #imgs = X_train[idx]
            imgs, [y_train, _] = next(self.traindata_generator)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # One-hot encoding of labels
            #labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            #labels = to_categorical(y_train, num_classes=self.num_classes+1)
            labels = np.concatenate((y_train, np.zeros((batch_size, 1))), axis=1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2]) * 1000

            # Plot the progress
            print ("%d [D loss: %f, D loss_real: %f, D loss_fake: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]"
             % (epoch, d_loss[0], d_loss_real[0], d_loss_fake[0], 100*d_loss[3], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 4, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        figsize_px = np.array([12, 6])
        fig, axs = plt.subplots(r, c, figsize=figsize_px, dpi=600)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='jet', interpolation='nearest')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/content/drive/Shared drives/ML/HDMS/images/mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "mnist_sgan_generator")
        save(self.discriminator, "mnist_sgan_discriminator")
        save(self.combined, "mnist_sgan_adversarial")


if __name__ == '__main__':
    with open('../20190719_bin_1_ndarray.pickle', mode='rb') as f:
        X = pickle.load(f)
        X = X[0:72,4:196,] # analysis #0-71 (72 samples)
        X = X + np.ones_like(X)
        X = np.log(X)
        m = np.max(np.max(X, axis=2, keepdims=True), axis=1, keepdims=True)
        X = X/m
        #X = (X - np.ones_like(X) * 0.5) * 2

    y = np.array([
        1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 
        1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 
        1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 
    ]) # 24 samples x 3 injections

    batchsize = 8
    gen = HDMSGenerator(train_X=X, train_y=y)
    g = gen.generate()

    sgan = SGAN(traindata_generator = g)
    sgan.train(epochs=20000, batch_size=batchsize, sample_interval=100)


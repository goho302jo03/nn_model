import cv2
import os
import glob
import numpy as np
from keras.layers import Input, Reshape, Dense, Flatten, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from sklearn.utils import shuffle
from keras.datasets import mnist


class GAN():

    def __init__(self):
        self.latent_dim = 100
        self.img_rows = 28
        self.img_cols = 28
        self.img_channel = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channel)

        self.generator, _ = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        self.discriminator.trainable = False
        input = Input(shape=(self.latent_dim, ))
        self.combine = Model(inputs = input, outputs = self.discriminator(self.generator(input)))
        self.combine.summary()
        self.combine.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    def build_generator(self):
        input = Input(shape=(self.latent_dim, ))
        dense1 = Dense(256)(input)
        dense1 = LeakyReLU(alpha=0.2)(dense1)
        dense1 = BatchNormalization(momentum=0.8)(dense1)
        dense2 = Dense(512)(dense1)
        dense2 = LeakyReLU(alpha=0.2)(dense2)
        dense2 = BatchNormalization(momentum=0.8)(dense2)
        dense3 = Dense(1024)(dense2)
        dense3 = LeakyReLU(alpha=0.2)(dense3)
        dense3 = BatchNormalization(momentum=0.8)(dense3)
        dense4 = Dense(np.prod(self.img_shape), activation='tanh')(dense3)
        logits = Reshape(self.img_shape)(dense4)
        model = Model(inputs = input, outputs = logits)

        return model, logits

    def build_discriminator(self):
        input = Input(shape=self.img_shape)
        reshape_input = Reshape((np.prod(self.img_shape), ))(input)
        dense1 = Dense(512, activation='relu')(reshape_input)
        dense1 = LeakyReLU(alpha=0.2)(dense1)
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = LeakyReLU(alpha=0.2)(dense2)
        logits = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs = input, outputs = logits)

        return model

    def train(self, epochs, batch_size):
        # imgs = []
        # data_path = os.path.join('./data/images/', '*g')
        # files = glob.glob(data_path)
        #
        # for file in files:
        #     img = cv2.imread(file)
        #     imgs.append(img)
        #
        # imgs = np.array(imgs)
        (real_imgs_x, _), (_, _) = mnist.load_data()
        real_imgs_x = np.reshape(real_imgs_x, (-1,28,28,1))/127.5 - 1
        # real_imgs_x = imgs
        valid_y = np.ones(batch_size)
        fake_y = np.zeros(batch_size)

        for epoch in range(epochs):
            print('----------------------------------')
            print('epoch: %d' %(epoch))
            print('----------------------------------')

            # train discriminator
            print('discriminator')

            # for k in range(5):
            idx = np.random.randint(0, np.shape(real_imgs_x)[0], batch_size)
            imgs = real_imgs_x[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            fake_imgs = self.generator.predict(noise)
            dis_loss_real = self.discriminator.train_on_batch(imgs, valid_y)
            dis_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_y)
            print(np.shape(fake_imgs))
            print('loss: ', 0.5 * (dis_loss_real[0] + dis_loss_fake[0]), ', accuracy: ', 0.5 * (dis_loss_real[1] + dis_loss_fake[1]))

            # train generator
            print('generator')
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.combine.train_on_batch(noise, valid_y)
            print('loss: ', gen_loss[0], ', accuracy: ', gen_loss[1])

            if epoch%1000 == 0:
                for i in range(10):
                    cv2.imwrite('./output/epoch%d_%d.jpg' %(epoch, i+1), (fake_imgs[i]+1)*127.5)

if __name__ == '__main__':
    gan = GAN()
    gan.train(100000, 32)


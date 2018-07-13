import cv2
import os
import glob
import numpy as np
from keras.layers import Input, Reshape, Dense, Flatten, Activation
from keras.models import Model
from sklearn.utils import shuffle

class GAN():

    def __init__(self):
        self.latent_dim = 50
        self.img_rows = 64
        self.img_cols = 64
        self.img_channel = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channel)

        self.generator, _ = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.discriminator.trainable = False
        input = Input(shape=(self.latent_dim, ))
        self.combine = Model(inputs = input, outputs = self.discriminator(self.generator(input)))
        self.combine.summary()
        self.combine.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def build_generator(self):
        input = Input(shape=(self.latent_dim, ))
        dense1 = Dense(256, activation='relu')(input)
        dense2 = Dense(512, activation='relu')(dense1)
        dense3 = Dense(1024, activation='relu')(dense2)
        dense4 = Dense(np.prod(self.img_shape), activation='sigmoid')(dense3)
        logits = Reshape(self.img_shape)(dense4)
        model = Model(inputs = input, outputs = logits)

        return model, logits

    def build_discriminator(self):
        input = Input(shape=self.img_shape)
        reshape_input = Reshape((np.prod(self.img_shape), ))(input)
        dense1 = Dense(512, activation='relu')(reshape_input)
        dense2 = Dense(1024, activation='relu')(dense1)
        dense3 = Dense(256, activation='relu')(dense2)
        logits = Dense(1, activation='sigmoid')(dense3)
        model = Model(inputs = input, outputs = logits)

        return model

    def train(self, epochs):
        imgs = []
        data_path = os.path.join('./data/images/', '*g')
        files = glob.glob(data_path)

        for file in files:
            img = cv2.imread(file)
            imgs.append(img)
        real_imgs_x = np.array(imgs)
        real_imgs_y = np.ones(np.shape(real_imgs_x)[0])
        noise_x = np.random.normal(0, 1, (np.shape(real_imgs_x)[0], self.latent_dim))
        noise_y = np.zeros(np.shape(real_imgs_x)[0])

        for epoch in range(epochs):
            print('epoch: %d' %(epoch))
            # train discriminator
            print('discriminator')
            fake_img = self.generator.predict(noise_x)
            discriminator_x = np.concatenate((fake_img, real_imgs_x), axis=0)
            discriminator_y = np.concatenate((noise_y, real_imgs_y))
            discriminator_x, discriminator_y = shuffle(discriminator_x, discriminator_y, random_state=0)
            self.discriminator.fit(discriminator_x, discriminator_y, batch_size=2048, epochs=1, verbose=1)

            # train generator
            print('generator')
            noise_x, noise_y = shuffle(noise_x, noise_y, random_state=0)
            self.combine.fit(noise_x, real_imgs_y, batch_size=2048, epochs=20, verbose=1)

            if epoch%100 == 0:
                for i in range(10):
                    cv2.imwrite('./output/epoch%d_%d.jpg' %(epoch, i+1), fake_img[i])
                    cv2.imwrite('./output/t_epoch%d_%d.jpg' %(epoch, i+1), real_imgs_x[i])

if __name__ == '__main__':
    gan = GAN()
    gan.train(10000)


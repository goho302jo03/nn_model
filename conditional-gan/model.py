#!/usr/bin/env python3
import cv2
import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Input, Reshape, Dense, Flatten, Activation, BatchNormalization, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

class GAN():

    def __init__(self):
        self.latent_dim = 100
        self.label_dim = 22
        self.img_rows = 64   # mnist: 28
        self.img_cols = 64   # mnist: 28
        self.img_channel = 3 # mnist: 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channel)

        self.generator, _ = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        self.discriminator.trainable = False
        input = Input(shape=(self.latent_dim, ))
        label = Input(shape=(self.label_dim, ))
        self.combine = Model(inputs = [input, label], outputs = self.discriminator([self.generator([input, label]), label]))
        self.combine.summary()
        self.combine.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    def build_generator(self):
        input = Input(shape=(self.latent_dim, ))
        label = Input(shape=(self.label_dim, ))
        concat = Concatenate()([input, label])
        dense1 = Dense(256)(concat)
        dense1 = LeakyReLU(alpha=0.2)(dense1)
        dense1 = BatchNormalization(momentum=0.8)(dense1)
        dense2 = Dense(512)(dense1)
        dense2 = LeakyReLU(alpha=0.2)(dense2)
        dense2 = BatchNormalization(momentum=0.8)(dense2)
        dense3 = Dense(1024)(dense2)
        dense3 = LeakyReLU(alpha=0.2)(dense3)
        dense3 = BatchNormalization(momentum=0.8)(dense3)
        logits = Dense(np.prod(self.img_shape), activation='tanh')(dense3)
        # logits = Reshape(self.img_shape)(dense4)
        model = Model(inputs = [input, label], outputs = logits)

        return model, logits

    def build_discriminator(self):
        input = Input(shape=(np.prod(self.img_shape), ))
        label = Input(shape=(self.label_dim, ))
        concat = Concatenate()([input, label])
        dense1 = Dense(512, activation='relu')(concat)
        dense1 = LeakyReLU(alpha=0.2)(dense1)
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = LeakyReLU(alpha=0.2)(dense2)
        logits = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs = [input, label], outputs = logits)

        return model

    def train(self, epochs, batch_size):

        # dataset: 2D face
        data = {}
        data_path = os.path.join('../dataset/anime-face/images/', '*g')
        files = glob.glob(data_path)

        for file in files:
            img = cv2.imread(file)
            key = file.split('/')[-1].replace('.jpg', '')
            data[key] = {}
            data[key]['img'] = img

        with open('../dataset/anime-face/tags.csv') as f:
            content = f.readlines()
        for line in content:
            line = line.split(',')
            style = line[1].split(' ')
            data[line[0]]['hair'] = style[0]
            data[line[0]]['eye'] = style[2]

        hair = []
        eye = []
        real_imgs_x = []
        for i in range(len(data)):
            real_imgs_x.append(data[str(i)]['img'])
            hair.append(data[str(i)]['hair'])
            eye.append(data[str(i)]['eye'])

        le_hair = LabelEncoder()
        le_eye = LabelEncoder()
        hair = le_hair.fit_transform(hair)
        eye = le_eye.fit_transform(eye)

        one_hair = OneHotEncoder()
        one_eye = OneHotEncoder()
        hair = one_hair.fit_transform(np.reshape(hair, (-1,1))).toarray()
        eye = one_eye.fit_transform(np.reshape(eye, (-1,1))).toarray()

        # dataset: mnist
        # (real_imgs_x, _), (_, _) = mnist.load_data()
        # real_imgs_x = np.reshape(real_imgs_x, (-1, 28, 28, 1))

        real_imgs_x = np.array(real_imgs_x)/127.5 - 1
        real_imgs_x = np.reshape(real_imgs_x, (len(real_imgs_x), -1))
        real_label = np.concatenate((hair, eye), axis=1)

        valid_y = np.ones(batch_size)
        fake_y = np.zeros(batch_size)

        for epoch in range(epochs):
            print('----------------------------------')
            print('epoch: %d' %(epoch))
            print('----------------------------------')

            # train discriminator
            print('discriminator')

            for k in range(3):
                idx = np.random.randint(0, np.shape(real_imgs_x)[0], batch_size)
                imgs = real_imgs_x[idx]
                batch_real_label = real_label[idx]

                hair_noise = np.random.randint(0, len(list(le_hair.classes_)), batch_size)
                hair_noise = one_hair.transform(np.reshape(hair_noise, (-1, 1))).toarray()
                eye_noise = np.random.randint(0, len(list(le_eye.classes_)), batch_size)
                eye_noise = one_eye.transform(np.reshape(eye_noise, (-1, 1))).toarray()

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                label_noise = np.concatenate((hair_noise, eye_noise), axis=1)

                fake_imgs = self.generator.predict([noise, label_noise])

                dis_loss_real = self.discriminator.train_on_batch([imgs, batch_real_label], valid_y)
                dis_loss_fake = self.discriminator.train_on_batch([fake_imgs, label_noise], fake_y)
                print('loss: ', 0.5 * (dis_loss_real[0] + dis_loss_fake[0]), ', accuracy: ', 0.5 * (dis_loss_real[1] + dis_loss_fake[1]))

            # train generator
            print('generator')
            hair_noise = np.random.randint(0, len(list(le_hair.classes_)), batch_size)
            hair_noise = one_hair.transform(np.reshape(hair_noise, (-1, 1))).toarray()
            eye_noise = np.random.randint(0, len(list(le_eye.classes_)), batch_size)
            eye_noise = one_eye.transform(np.reshape(eye_noise, (-1, 1))).toarray()

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            label_noise = np.concatenate((hair_noise, eye_noise), axis=1)

            gen_loss = self.combine.train_on_batch([noise, label_noise], valid_y)
            print('loss: ', gen_loss[0], ', accuracy: ', gen_loss[1])

            if epoch%1000 == 0:
                for i in range(10):
                    img = np.reshape(fake_imgs[i], (self.img_shape))
                    cv2.imwrite('./output/epoch%d_%d.jpg' %(epoch, i), img*127.5)
        self.generator.save('generator.h5')
        le_hair_mapping = dict(zip(le_hair.classes_, le_hair.transform(le_hair.classes_)))
        le_eye_mapping = dict(zip(le_eye.classes_, le_eye.transform(le_eye.classes_)))
        with open('le_hair.pkl', 'wb') as f:
            pickle.dump(le_hair_mapping, f)
        with open('le_eye.pkl', 'wb') as f:
            pickle.dump(le_eye_mapping, f)

if __name__ == '__main__':
    gan = GAN()
    gan.train(100, 128)


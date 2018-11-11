import pickle
import cv2
import sys
import numpy as np
from keras.models import load_model

def main():
    model = load_model('./public/' + sys.argv[1])
    latent_dim = 100
    img_shape = (64, 64, 3)

    with open('./public/le_eye.pkl', 'rb') as f:
        le_eye = pickle.load(f)
    with open('./public/le_hair.pkl', 'rb') as f:
        le_hair = pickle.load(f)

    hair_idx = le_hair[sys.argv[2]]
    eye_idx = le_eye[sys.argv[3]]
    hair_hot = [0]*12
    eye_hot = [0]*10
    hair_hot[hair_idx] = 1
    eye_hot[eye_idx] = 1

    for i in range(10):
        noise = np.random.normal(0, 1, (1, latent_dim))
        hair_hot = np.reshape(hair_hot, (1, -1))
        eye_hot = np.reshape(eye_hot, (1, -1))
        label = np.concatenate((hair_hot, eye_hot), axis=1)

        img = np.reshape(model.predict([noise, label]) , img_shape)
        cv2.imwrite('./public/result/%d.jpg' %i, img*127.5)


if __name__ == '__main__':
    main()

import pickle
import cv2
import numpy as np
from keras.models import load_model

def main():
    model = load_model('generator.h5')
    latent_dim = 100
    img_shape = (64, 64, 3)

    with open('le_eye.pkl', 'rb') as f:
        le_eye = pickle.load(f)
    with open('le_hair.pkl', 'rb') as f:
        le_hair = pickle.load(f)

    print('\nPlease Enter Image Style: Hair color/Eye color')
    print('Hair Color: ', le_hair.keys())
    print('Eye Color: ', le_eye.keys())
    i = 1
    while True:
        query = input()
        q_list = query.split()
        hair_idx = le_hair[q_list[0]]
        eye_idx = le_eye[q_list[1]]
        hair_hot = [0]*12
        eye_hot = [0]*10
        hair_hot[hair_idx] = 1
        eye_hot[eye_idx] = 1

        noise = np.random.normal(0, 1, (1, latent_dim))
        hair_hot = np.reshape(hair_hot, (1, -1))
        eye_hot = np.reshape(eye_hot, (1, -1))
        label = np.concatenate((hair_hot, eye_hot), axis=1)

        img = np.reshape(model.predict([noise, label]) , img_shape)
        cv2.imwrite('./result/%d.jpg' %i, img*127.5)
        i += 1

        print('\nSuccessfully Gererate Image!!!')
        print('==============================')
        print('Please Enter Image Style: Hair color/Eyes color')
        print('Hair Color: ', le_hair.keys())
        print('Eye Color: ', le_eye.keys())

if __name__ == '__main__':

    main()

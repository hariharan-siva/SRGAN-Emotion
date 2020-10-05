import sys
import os
import matplotlib.pyplot as plt
from model.srgan import generator, discriminator
from model import resolve_single
from utils import load_image
import tensorflow as tf
import cv2
from model.srgan import generator, discriminator
from model import resolve_single
from utils import load_image
import PIL
from PIL import Image
from tensorflow.python.types import core as core_tf_types

weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)
os.makedirs(weights_dir, exist_ok=True)

gan_generator = generator()
gan_generator.load_weights(weights_file('gan_generator.h5'))

frame_number = 0
folder = r"C:\Users\Hari\Desktop\super-resolution-master\data"
current_path = "C:\\Users\\Hari\\Desktop\\super-resolution-master\\save\\"

for filename in os.listdir(folder):
    frame_number += 1
    print(frame_number)
    lr = load_image(os.path.join(folder,filename))
    gan_sr = resolve_single(gan_generator, lr)
    tf.keras.preprocessing.image.save_img(current_path + str(frame_number) + ".png",gan_sr)
    #save = cv2.resize(lr, (100,100))
    #cv2.imwrite(current_path + str(frame_number) + ".png",save)
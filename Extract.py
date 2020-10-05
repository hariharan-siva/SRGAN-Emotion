import sys
import os
import time

import tensorflow as tf

import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS

import numpy as np

import cv2
import face_recognition

import keras
from keras import models

from model.srgan import generator, discriminator
from model import resolve_single
from utils import load_image

import PIL
from PIL import Image
from numpy import asarray

model = models.load_model("C:/model_v6_23.hdf5")
emotion_dict= {'Angry': 0, 'Sad': 2, 'Surprise': 3, 'Happy': 1}

folder = r"#PATH to Videos"
save_1 = r"#Save Path"

face_locations = []
counter = 0
frame_number = 0

for filename in os.listdir(folder):
	fvs = FileVideoStream(os.path.join(folder,filename)).start()
	print("[INFO] starting video file thread:" + os.path.join(folder,filename))
	time.sleep(1.0)
	fps = FPS().start()
	while fvs.more():
		frame_number += 1
		frame = fvs.read()
		if frame_number%4 != 0:
			continue
		
		if frame is None:
			break
		
		frame = imutils.resize(frame, width=450)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#frame = np.dstack([frame, frame, frame])
		print(frame_number)
		
		face_locations = face_recognition.face_locations(frame)
		if not face_locations:
			continue
			
		for (top, right, bottom, left) in face_locations:
			crop_img = frame1 = frame[top:bottom, left:right]
			crop_img = np.dstack([crop_img, crop_img, crop_img])
			
			frame1 = cv2.resize(frame1, (48,48))
			temp = np.reshape(frame1, [1, frame1.shape[0], frame1.shape[1], 1])
			predicted_class = np.argmax(model.predict(temp))
			label_map = dict((v,k) for k,v in emotion_dict.items())
			predicted_label = label_map[predicted_class]
			
			cv2.imwrite(save_1 + predicted_label + str(counter) + ".png",crop_img)
			counter = counter + 1			
		cv2.waitKey(1)
		fps.update()

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
cv2.destroyAllWindows()
fvs.stop()
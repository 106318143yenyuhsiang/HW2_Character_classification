import cv2
import os
import numpy as np
from keras.models import load_model
import pandas as pd


def read_images(path):
    images=[]
    for i in range(990):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
        images.append(image)

    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]]
        label_str.append(temp)

    return label_str
path='test/test/'
images = read_images(path)
model = load_model('train.h5')
predict = model.predict(images)
y_classes = predict.argmax(axis=1)
print(y_classes)
label_str=transform(np.loadtxt('listdir.txt',dtype='str'),y_classes,images.shape[0])

pd.DataFrame({"id": list(range(1,len(label_str)+1)),"character": label_str}).to_csv('test_score.csv', index=False, header=True)

# HW2_Character_classification
# 引用相關套件
	import os
	from sklearn.cross_validation import train_test_split
	from keras.models import Sequential, Model
	from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
	import numpy as np
	from keras import applications
	import cv2
	from keras.optimizers import SGD
	from keras.utils import np_utils
	from keras.preprocessing.image import ImageDataGenerator
# 創建圖片標籤的矩陣
	images=[]
	labels= []
	listdir= []
# 讀取圖片函式
	def read_images_labels(path,i):
    	for file in os.listdir(path):
        	abs_path = os.path.abspath(os.path.join(path, file))   
        	if os.path.isdir(abs_path):
            	i+=1                                              
           		temp = os.path.split(abs_path)[-1]                 
            	listdir.append(temp)
            	read_images_labels(abs_path,i)    
            	print(i,temp)
        	else:  
            	if file.endswith('.jpg'):
                	image=cv2.resize(cv2.imread(abs_path),(64,64)) 
                	images.append(image)                           
                	labels.append(i-1)                             
    return images, labels ,listdir
# 讀取訓練圖片
	path='train/characters-20'
	read_images_labels(path,0)
# 圖片數據標準化
	images=np.array(images,dtype=np.float32)/255
# 標籤轉成1其餘為0的矩陣形式
	labels = np_utils.to_categorical(labels, num_classes=20)
# 儲存標籤內容
	np.savetxt('listdir.txt', listdir, delimiter = ' ',fmt="%s")
# 將訓練數據拆成訓練與測試集
	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
# 引用VGG模型與後面自行生成全連接層
	base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
	add_model = Sequential()
	add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	add_model.add(Dense(64, activation='relu'))
	add_model.add(Dense(20, activation='sigmoid'))
	model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# 產生縮放圖形以供訓練
	datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True,)
	datagen.fit(X_train)
# 訓練模型
	model.fit_generator(datagen.flow(X_train,y_train,batch_size=64), epochs=35,steps_per_epoch=100)
# 儲存模型與驗證
	model.save('train.h5')
	print(model.evaluate(X_test, y_test, verbose=0))
# 實驗過程
![image](https://github.com/106318143yenyuhsiang/HW2_Character_classification/blob/master/report2.JPG)
# Kaggle排名
![image](https://github.com/106318143yenyuhsiang/HW2_Character_classification/blob/master/rank.JPG)
# 分析
	使用VGG包好的函式訓練,因為電腦是一般家用電腦,因此訓練一次模型就要使用超過半個小時的時間,因此使用保存權重再訓練的方式
# 檢討改進

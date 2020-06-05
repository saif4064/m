

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





from keras.preprocessing.image import ImageDataGenerator
immag=ImageDataGenerator()
train=immag.flow_from_directory("C:/Users/Lenovo/covid/h",class_mode="categorical")





data=pd.read_csv("metadata.csv")





data


# In[4]:


s='images/'
x=[]
for i in data.index:
    #data['filename'].astype('str')
    data['filename'][i]=s+data['filename'][i]
    
    





x=[]
for i in data.filename:
    img=plt.imread(''+i)
    #img.astype('str')
    x.append(img)
X=np.array(x)





X.shape





from sklearn.preprocessing import LabelEncoder
m=LabelEncoder()
data['finding']=m.fit_transform(data['finding'])





from keras.utils import np_utils
y=data.finding
yy=np_utils.to_categorical(y)





yy





data.head()






X.shape[0]





from skimage.transform import resize
image=[]
for i in range(0,X.shape[0]):
    a=resize(X[i],preserve_range=True,output_shape=(100,100,3))
    image.append(a)
XX=np.array(image)





XX.shape





import pickle
model=pickle.dump(XX,open('traffic6.pkl','wb'))





from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(XX,yy,test_size=0.20,random_state=1)





xtrain.shape




from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,MaxPool2D,Flatten,InputLayer





model=Sequential()

model.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=(100,100,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(70,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(90,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))

model.add(Dense(units=11, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(xtrain,ytrain, epochs=80, validation_data=(xtest,ytest))





model2=pickle.dump(model,open('wb1.pkl','wb'))





import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image1(filename):
    
    img = load_img(filename, target_size=(100, 100))
    
    img = img_to_array(img)
    
    img = img.reshape(1, 100, 100, 3)
    
    return img


def run_example():
   
    img = load_image1(r'C:\Users\Lenovo\covid\static\ards-secondary-to-tiger-snake-bite.pngprint(img)
    model=pickle.load(open("wb1.pkl",'rb'))
    
    result = model.predict_classes(img)
    print(result[0])
run_example()













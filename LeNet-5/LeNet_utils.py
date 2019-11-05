import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import pickle
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

class LossAndErrorPrintingCallback(Callback):
    def __init__(self, ):
        self.loss_arr = []
    def on_train_batch_end(self, batch, logs=None):
        #print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
        self.loss_arr.append(logs['loss'])
    def on_epoch_end(self, epoch, logs=None):
        x = [i for i in range(0,len(self.loss_arr))]
        y = self.loss_arr
        plt.figure(num="train_one_epoch")
        plt.plot(x, y)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def randomShowPics():
    info = unpickle('dataset/batches.meta')
    label_names = info['label_names']
    data = unpickle('dataset/data_batch_1')
    imgs = data['data'].reshape(-1,3,32,32)
    labels = data['labels']
    imgs = np.transpose(imgs, (0, 2, 3,1))
    
    r = range(10000)
    random_list = random.sample(r,10)

    plt.figure(num='Show pictures',figsize=(12, 6))
    
    for i in range(10):
        plt.subplot(2,5,i+1)
        idx = random_list[i]
        plt.imshow(imgs[idx])
        plt.xlabel(label_names[labels[idx]])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    '''
    for i in range(10):
        idx = random.randint(0,9999)
        img = imgs[idx]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(label_names[labels[idx]],0);
        cv2.resizeWindow(label_names[labels[idx]], 300, 300);
        cv2.imshow(label_names[labels[idx]],img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    '''
def loadDataset():
    data = unpickle('dataset/data_batch_1')
    X_train = data['data'].reshape(-1,3,32,32)
    Y_train = np.array(data['labels']).reshape(-1,1)
    for i in range(2,6):
        data = unpickle('dataset/data_batch_'+str(i))
        X = data['data'].reshape(-1,3,32,32)
        Y = np.array(data['labels']).reshape(-1,1)
        X_train = np.concatenate((X_train,X), axis=0)
        Y_train = np.concatenate((Y_train,Y), axis=0)
    X_train = np.transpose(X_train, (0, 2, 3, 1)) 
    
    data_test = unpickle('dataset/test_batch')
    X_test = data_test['data'].reshape(-1,3,32,32)
    X_test = np.transpose(X_test, (0, 2, 3, 1)) 
    Y_test = np.array(data_test['labels']).reshape(-1,1)
    return X_train, Y_train, X_test, Y_test

def printHyperparameter():
    print('hyperparameters:')
    print('batch size:16')
    print('learning rate:0.001')
    print('optimizer: SGD')

def createLeNetModel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5),strides=(1, 1), padding='valid',activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPool2D(pool_size=(2, 2),padding='valid'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5),strides=(1, 1), padding='valid',activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120,activation='relu'))
    model.add(layers.Dense(84,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model

def showTrainingResult():
    img = cv2.imread('result.png')
    cv2.namedWindow("Result",0);
    cv2.resizeWindow("Result", 727, 604);
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train_one_epoch_gui():
    X_train, Y_train, X_test, Y_test = loadDataset()
    train_one_epoch(X_train,Y_train,X_test,Y_test)

def train_one_epoch(X_train,Y_train,X_test,Y_test):
    model = createLeNetModel()
    history = model.fit(X_train, Y_train, epochs=1,batch_size=128,callbacks=[LossAndErrorPrintingCallback()])
    #loss, acc = model.evaluate(x=X_test, y=Y_test)
    #print("loss: %s, acc: %s" % (loss, acc))
    
    
def train(X_train, Y_train, X_test, Y_test):
    checkpoint_dir = "model/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = createLeNetModel()
    model.summary()
    if latest:
        print('Train from '+latest)
        model.load_weights(latest)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_freq=1)
    history = model.fit(X_train, Y_train, epochs=1,batch_size=16,callbacks=[cp_callback])
    print(history)
    #loss, acc = model.evaluate(x=X_test, y=Y_test)
    #print("loss: %s, acc: %s" % (loss, acc))

def predict(index):
    info = unpickle('dataset/batches.meta')
    label_names = info['label_names']
    X_train, Y_train, X_test, Y_test = loadDataset()
    X = np.array([X_test[index]], dtype="float32")
    checkpoint_dir = "model/adam85"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = createLeNetModel()
    if latest:
        model.load_weights(latest)
    predict = model.predict(X)
    prob = predict[0].tolist()
    idx = prob.index(max(prob))
    
    plt.figure(num='Predict',figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.bar(label_names,predict[0])
    plt.subplot(2,1,2)
    plt.imshow(X_test[index])
    plt.xlabel(label_names[idx])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
if __name__ == '__main__':
    #X_train, Y_train, X_test, Y_test = loadDataset()
    #train_one_epoch(X_train, Y_train, X_test, Y_test)
    #predict(555)
    randomShowPics()
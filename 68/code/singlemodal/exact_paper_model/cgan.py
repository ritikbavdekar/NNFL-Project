from __future__ import print_function, division
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,concatenate
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam, SGD
from utils import OneHot


#conditional GAN class
class conGAN():
    def __init__(self):

        self.prev_epoch=0
        self.img_shape = (28, 28, 1)

        self.numOfClasses = 10
        self.latentVectorSize = 100
        
        #SGD optimizer
        optimizer = SGD(lr=0.1, momentum=0.5, decay=1.00004)

        self.discriminator = self.buildDisc()
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])
        
        self.generator = self.buildGen()
        self.generator.compile(loss=['binary_crossentropy'],optimizer=optimizer)

        latentVec = Input(shape=(100,))
        condition = Input(shape=(1,))
        img = self.generator([latentVec,condition])
        self.discriminator.trainable = False
        valid = self.discriminator([img, condition])
        self.combined = Model([latentVec, condition], valid)
        self.combined.compile(loss=['binary_crossentropy'],optimizer=optimizer)

    def buildGen(self):
#generator

        first_input = Input(shape=(self.latentVectorSize,))
        first_dense = Dense(200,)(first_input)
        first_dense_lr = LeakyReLU()(first_dense)
        first_dense_do = Dropout(0.5)(first_dense_lr)
        second_input = Input(shape=(1,),dtype='int32')
        second_one_hot = Flatten()(OneHot(input_dim=10,input_length=1)(second_input))
        second_dense = Dense(1000,)(second_one_hot)
        second_dense_lr = LeakyReLU()(second_dense)
        second_dense_do = Dropout(0.5)(second_dense_lr)
        merge = concatenate([first_dense_do,second_dense_do])
        third_dense = Dense(1200,)(merge)
        third_dense_lr = LeakyReLU()(third_dense)
        third_fourth_do = Dropout(0.5)(third_dense_lr)
        fourth_dense = Dense(784,activation='sigmoid')(third_fourth_do)
        fourth_dense_reshape = Reshape(self.img_shape)(fourth_dense)

        model = Model(inputs=[first_input,second_input],outputs=fourth_dense_reshape)

        return model

    def buildDisc(self):
#discriminator
        first_input = Input(shape=self.img_shape)
        first_input_flatten = Flatten()(first_input)
        first_dense = Dense(240,)(first_input_flatten)
        first_dense_lr = LeakyReLU()(first_dense)
        first_dense_do = Dropout(0.5)(first_dense_lr)
        second_input = Input(shape=(1,),dtype='int32')
        second_one_hot = Flatten()(OneHot(input_dim=10,input_length=1)(second_input))
        second_dense = Dense(50,)(second_one_hot)
        second_dense_lr = LeakyReLU()(second_dense)
        second_dense_do = Dropout(0.5)(second_dense_lr)
        merge = concatenate([first_dense_do,second_dense_do])
        third_dense = Dense(240,)(merge)
        third_dense_lr = LeakyReLU()(third_dense)
        third_dense_do = Dropout(0.5)(third_dense_lr)
        fourth_dense = Dense(1,activation='sigmoid')(third_dense_do)
        model = Model(inputs=[first_input,second_input],outputs=fourth_dense)
        return model

    def train(self, epochs, batch_size=25, save_interval=10):                 #training function 
        
        (X_train, y_train), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        epoch_array=[]
        g_loss_array=[]
        d_loss_fake_array=[]
        d_loss_real_array=[]
        
        for epoch in range(self.prev_epoch,self.prev_epoch +epochs):
          
          epoch_array.append(epoch)
          
          idx = np.random.randint(0, X_train.shape[0], batch_size)
          imgs, labels = X_train[idx], y_train[idx]
          
          noise = np.random.normal(0, 1, (batch_size, 100))
          
          gen_imgs = self.generator.predict([noise, labels])
          
          valid = np.ones((batch_size, 1))
          fake = np.zeros((batch_size, 1))
          
          d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
          d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
          
          noise = np.random.normal(0, 1, (batch_size, 100))
          valid = np.ones((batch_size, 1))
          sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
          
          g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
          g_loss_array.append(g_loss)
          d_loss_real_array.append(d_loss_real[0])
          d_loss_fake_array.append(d_loss_fake[0])
          print ("%d Discriminator loss: %f Generator loss: %f" % (epoch, d_loss[0], g_loss))
          if epoch % save_interval == 0:
              self.save_imgs(epoch)
              self.save_model(epoch)
              self.loss_plot(epoch_array,g_loss_array,d_loss_real_array,d_loss_fake_array)
        
        self.loss_plot(epoch_array,g_loss_array,d_loss_real_array,d_loss_fake_array)
        self.save_model(epoch_array[-1])
        self.save_imgs(epoch_array[-1])

    def save_model(self,epoch):                                             #To save complete model after save_interval epochs 
      self.generator.save("/content/model/generator")
      self.discriminator.save("/content/model/discriminator")
      self.combined.save("/content/model/combined")
    def loss_plot(self,epoch_array,g_loss_array,d_loss_real_array,d_loss_fake_array):   #Plot the losses
      plt.plot(epoch_array, g_loss_array, label = "Generator Loss")
      plt.plot(epoch_array, d_loss_real_array, label = "Discriminator Loss for Real Image")
      plt.plot(epoch_array, d_loss_fake_array, label = "Discriminator Loss for Fake Image")
      plt.xlabel("number of epochs")
      plt.ylabel("Loss")
      plt.title("Loss Plots")
      plt.legend()
      plt.savefig("/content/lossplots/lossplot"+str(epoch_array[-1]))

      plt.close()

    def save_imgs(self, epoch):                # to save images after save_interval epochs
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.suptitle("CGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("%d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/content/images/%d.png" % epoch)
        plt.close()
        
    def pred(self,n=1000):                                      # Generates n samples using conditions 0,1,2 .. ,9 in succession. 
      r, c = 2, 5
      noise = np.random.normal(0, 1, (r * c, 100))
      sampled_labels = np.arange(0, 10).reshape(-1, 1)
      gen_imgs_array=[]
      for i in range(int(n//10)):
        gen_imgs=self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        for j in range(10):
          gen_imgs_array.append(gen_imgs[j])
      return np.array(gen_imgs_array)
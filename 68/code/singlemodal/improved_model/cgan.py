from __future__ import print_function, division
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


def loadmodel(epoch,path="/content/model/"):#Enter last epoch of previous run to continue training or to load model for testing
  cgan=CGAN()
  cgan.generator=keras.models.load_model(path+"generator")
  cgan.discriminator=keras.models.load_model(path+"discriminator")
  cgan.combined=keras.models.load_model(path+"combined")
  cgan.prev_epoch=1+epoch
  return cgan

class CGAN():
    def __init__(self):
        self.prev_epoch=0
        self.img_shape = (28,28,1)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        noise = Input(shape=(self.latent_dim,))

        label = Input(shape=(1,))
        img = self.generator([noise, label])
        self.discriminator.trainable = False

        valid = self.discriminator([img, label])


        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],optimizer=optimizer)
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.6))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.6))
        model.add(Dropout(0.5))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)


    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)
        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, y_train), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        epoch_array=[]
        g_loss_array=[]
        d_loss_fake_array=[]
        d_loss_real_array=[]

        for epoch in range(epochs):

            epoch_array.append(epoch)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            g_loss_array.append(g_loss)
            d_loss_real_array.append(d_loss_real[0])
            d_loss_fake_array.append(d_loss_fake[0])
            
            print ("%d Discriminator loss: %f Generator loss: %f" % (epoch, d_loss[0], g_loss))
            if epoch % sample_interval == 0:
                self.save_imgs(epoch)
                self.save_model(epoch)
                self.loss_plot(epoch_array,g_loss_array,d_loss_real_array,d_loss_fake_array)

    def save_model(self,epoch):
      # self.generator.save("/content/model/generator"+str(epoch))
      # self.discriminator.save("/content/model/discriminator"+str(epoch))
      # self.combined.save("/content/model/combined"+str(epoch))
      self.generator.save("/content/model/generator")
      self.discriminator.save("/content/model/discriminator")
      self.combined.save("/content/model/combined")

    def loss_plot(self,epoch_array,g_loss_array,d_loss_real_array,d_loss_fake_array):
      plt.plot(epoch_array, g_loss_array, label = "Generator Loss")
      plt.plot(epoch_array, d_loss_real_array, label = "Discriminator Loss for Real Image")
      plt.plot(epoch_array, d_loss_fake_array, label = "Discriminator Loss for Fake Image")
      plt.xlabel("number of epochs")
      plt.ylabel("Loss")
      plt.title("Loss Plots")
      plt.legend()
      plt.savefig("/content/lossplots/lossplot"+str(epoch_array[-1]))
      plt.close()
    
    def pred(self,n=1000):
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
    
    def gen_imgs_for_check(self):
      imgArr = self.pred(10)
      for i in range(10):
        fig = plt.plot()
        plt.imshow(imgArr[i,:,:,0],cmap="gray")
        plt.savefig("/content/predImages/%d.png"%(i+1))
        plt.close()        



    def save_imgs(self, epoch):
        r, c = 2,5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("%d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/content/images/%d.png" % epoch)
        plt.close()
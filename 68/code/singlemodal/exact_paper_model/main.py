from cgan import conGAN
from utils import loadmodel
from utils import callParzen


cgan = conGAN()
# TO USE THE MODEL FROM THE PREVIOUS RUN uncomment next line
# cgan = loadmodel(epoch=500)

cgan.train(100000,100,100)
parzen_value=callParzen(cgan,1000)

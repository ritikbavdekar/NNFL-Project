from cgan import conGAN
import keras
import numpy
import theano
from keras.layers import Lambda
from keras import backend as K


def loadmodel(epoch=0,path="/content/model/"):#Enter last epoch of previous run to continue training or to load model for testing
  cgan=conGAN()
  cgan.generator=keras.models.load_model(path+"generator")
  cgan.discriminator=keras.models.load_model(path+"discriminator")
  cgan.combined=keras.models.load_model(path+"combined")
  cgan.prev_epoch=1+epoch
  return cgan



def callParzen(cgan,n):

  T = theano.tensor
  val=cgan.pred(n)
  val=val.reshape(n,-1)
  def log_mean_exp(a):
      max_ = a.max(1)
      return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


  def make_lpdf(mu, sigma):
      x = T.matrix()
      mu = theano.shared(mu)
      a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma

      E = log_mean_exp(-0.5*(a**2).sum(2))

      Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))
      return theano.function([x], E - Z)


  class ParzenWindows(object):
      def __init__(self, samples, sigma):
          self._samples = samples
          self._sigma = sigma
          self.lpdf = make_lpdf(samples, sigma)

      def get_ll(self, x, batch_size=10):
          inds = range(x.shape[0])
          n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

          lls = []
          for i in range(n_batches):
              lls.extend(self.lpdf(x[inds[i::n_batches]]))

          return numpy.array(lls).mean()
  pw=ParzenWindows(val,0.3)
  ans=pw.get_ll(val)
  return ans

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
  if input_dim is None or input_length is None:
      raise TypeError("input_dim or input_length is not set")

  # Helper method (not inlined for clarity)
  def _one_hot(x, num_classes):
      return K.one_hot(K.cast(x, 'uint8'),
                        num_classes=num_classes)

  # Final layer representation as a Lambda layer
  return Lambda(_one_hot,
                arguments={'num_classes': input_dim},
                input_shape=(input_length,))



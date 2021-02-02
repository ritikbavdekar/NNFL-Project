from cgan import CGAN
import keras
import numpy
import theano
from keras.layers import Lambda

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

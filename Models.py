import tensorflow as tf
import random as rn
class Projection(tf.keras.Model):
  def __init__(self,d):
      super(Projection,self).__init__()
      self.P = tf.Variable(tf.random_normal(shape=(d,d)))
    
  def call(self, q):
    if len(q.shape)==1:
      q=tf.reshape(q,shape=(1,q.shape[0]))
    P=tf.matmul(self.P,tf.transpose(q))
    return tf.matmul(P,q,transpose_a=True,transpose_b=True)/(tf.norm(P)*tf.norm(q))
    
class Intersection(tf.keras.Model):
    def __init__(self,d,output_shape):
      super(Intersection,self).__init__()
      self.W = tf.Variable(tf.random_normal(shape=(d,output_shape)))#output_shape=1  
      seq=tf.keras.Sequential()
      dense=tf.keras.layers.Dense(output_shape,input_shape=(d,),activation=tf.nn.relu,
                                    use_bias=True,kernel_initializer='glorot_uniform',
                                    bias_initializer='glorot_uniform')
      dropout=tf.keras.layers.Dropout(0.2)
      seq.add(dense)
      seq.add(dropout)
      self.NN=seq
    
    def call(self, q):
        if len(q.shape)==1:
          q=tf.reshape(q,shape=(1,q.shape[0]))
        q=tf.reshape(q,(q.shape[0],q.shape[1]))
        B=tf.reduce_mean(self.NN(q),axis=0)
        B=tf.matmul(self.W,tf.reshape(B,shape=(B.shape[0],1)))
        B=tf.matmul(tf.transpose(B),tf.transpose(q))/tf.norm(B)*tf.norm(q)
        return tf.transpose(B)

    def evaluate(self,q):
      q=list(map(lambda x:tf.transpose(x),q))
      q=tf.convert_to_tensor(q)
      mean=tf.reduce_mean(self.NN(q),axis=0)
      output=tf.matmul(self.W,tf.transpose(mean))
      return output
        


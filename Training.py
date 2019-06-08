from Graph import *
from sklearn.model_selection import train_test_split
from Models import *
import pickle
tf.enable_eager_execution()
tfe = tf.contrib.eager
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

class Embbed_Projection(Projection):
  def __init__(self,d,input_shape):
      Projection.__init__(self,d)
      self.Z = tf.Variable(tf.random_normal(shape=(d,input_shape)))
      self.m=input_shape
    
  def call(self, x):
      x=tf.reshape(x,shape=(x.shape[0],1))
      Zu=tf.matmul(self.Z,x)/(self.m)#+0.000001*tf.matmul(self.Z,x)/(self.m)
      P=tf.matmul(self.P,Zu)
      #P=P+0.001*P
      ret=tf.matmul(P,Zu,transpose_a=True)/(tf.norm(P)*tf.norm(Zu))
      return ret#+0.0000001*ret

class Embbed_Intersection(Intersection):
    def __init__(self,d,output_shape,input_shape): 
      Intersection.__init__(self,d,output_shape)
      self.Z = tf.Variable(tf.random_normal(shape=(d,input_shape)))
      self.m=input_shape
      #self.NN=tf.Variable(tf.random_normal(shape=(d,output_shape)))
      #self.bias=tf.Variable(tf.random_normal(shape=(1,output_shape)))
    
    def call(self, x):
        #x=feature,embedings
        #q=tf.reshape(q,shape=(1,q.shape[0]))
        Zu=tf.matmul(self.Z,tf.transpose(x))/self.m
        Zu=tf.transpose(Zu)
        B=tf.reduce_mean(self.NN(Zu))#tf.nn.relu(tf.matmul(x,self.NN)+self.bias))
        #B Vector or scalar ?
        print(self.NN(Zu))
        print(B,'---',self.W.shape)
        B=self.W*B
        print(B.shape)
        return tf.matmul(tf.transpose(B),tf.transpose(Zu))/tf.norm(Zu)*tf.norm(B)#*Zu/tf.norm(Zu)


class Train():
  
  def __init__(self,Graph,query_type,embedding_space):
    global feed_Projection,Graph_,d
    feed_Projection=Graph.feed_Projection(query_type)    
    Graph_=Graph
    d=embedding_space
    
  def margin_loss(self,model,x_neg,x_pos):
      E1=model.call(x_neg)
      E=model.call(x_pos)
      cost=1-E + E1 + 5*model.P[rn.randint(0,d-1)][rn.randint(0,d-1)]**2 + 5*model.Z[rn.randint(0,d-1)][rn.randint(0,input_shape-1)]**2
      if 0>cost:
          return tf.convert_to_tensor(0)
      else:
          return cost
        
  def change(self,x):
          if x<0:
                  return tf.constant([0],dtype=tf.float32)
          else:
                  return x
                
  def func(self,tensor):
    return tf.map_fn(self.change,tensor,dtype=tf.float32)

  def mean_margin_loss(self,model,x_neg,x_pos):#mean margin losss
      E1=model.call(x_neg)
      E=model.call(x_pos)
      cost=1-E+E1
      ret=self.func(cost)
      return sum(ret)/len(ret)

  def grad(self,model,x_neg,x_pos,loss):
    with tf.GradientTape() as tape:
      loss_value=loss(model,x_neg,x_pos)
    return loss_value,tape.gradient(loss_value,model.trainable_variables)

  def Accuracy(self,model,neg,pos):
    return 'Pos_Acc = ',(sum(model(pos))/len(pos)).numpy(),'Neg_Acc=',(sum(model(neg))/len(pos)).numpy()

  def split_Data(self,feed):
      pos,neg=feed
      train_pos,val_pos=train_test_split(pos,test_size=0.1,shuffle=True)
      length_val_pos=len(val_pos)
      length_train_pos=len(train_pos)
      train_neg,val_neg=train_test_split(neg,test_size=0.1,shuffle=True)
      length_val_neg=len(val_neg)
      length_train_neg=len(train_neg)
      print('Number of Train_positive Samples = ',length_train_pos,' Number of Train_negative_Samples =',length_train_neg)
      print('Number of Val_positive Samples = ',length_val_pos,' Number of Val_negative_Samples =',length_val_neg)
      
      train_neg=tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_neg,dtype=tf.float32))
      train_pos=tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_pos,dtype=tf.float32))
      val_neg=tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(val_neg,dtype=tf.float32))
      val_pos=tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(val_pos,dtype=tf.float32))
      batch_size=min(64,length_train_pos,length_train_neg)
      batch_size_val=min(64,length_val_pos,length_val_neg)
      return batch_size,batch_size_val,len(pos[0]),[train_neg,train_pos,val_neg,val_pos]
    
  def train_Projection(self):
    global input_shape
    train_loss_results=[]
    val_loss_results=[]
    num_epochs=20000
    batch_size,batch_size_val,input_shape,data=self.split_Data(feed_Projection)
    train_neg,train_pos,val_neg,val_pos=data
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    model=Embbed_Projection(d,input_shape)
    for epoch in range(num_epochs):
        epoch_loss_avg=tfe.metrics.Mean()
        epoch_val_avg=tfe.metrics.Mean()
        epoch_val_PosAcc=tfe.ASYNC 
        neg_=next(iter(train_neg.shuffle(4).batch(batch_size,True)))
        pos_=next(iter(train_pos.shuffle(4).batch(batch_size,True)))
        for pos,neg in zip(pos_,neg_):
          loss_value, grads = self.grad(model, neg,pos,loss=self.margin_loss)
          #grads_and_vars = optimizer.minimize(batch_loss(model,neg[:1000],pos[:1000])
          #                                     ,model.trainable_variables)
          #print(grads_and_vars)
          epoch_loss_avg(loss_value.numpy())
          if loss_value.numpy()==0:
            continue
          optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
    
        val_neg_=next(iter(val_neg.shuffle(4).batch(batch_size_val,True)))
        val_pos_=next(iter(val_pos.shuffle(4).batch(batch_size_val,True)))
        for pos,neg in zip(val_pos_,val_neg):
            epoch_val_avg(self.margin_loss(model,neg,pos))
            #print('Accuracy=',Accuracy(model,neg,pos)) 
            #print('Val_Loss=',mean_margin_loss(model,neg,pos).numpy())
             
        train_loss_results.append(epoch_loss_avg.result())
        val_loss_results.append(epoch_val_avg.result())
        if epoch % 50 == 0: 
          print("Epoch {:03d}: Avg_Train_Loss: {:.3f} : Avg_Val_Loss:{:.3f} ,".format(epoch,epoch_loss_avg.result()
                                                                   ,epoch_val_avg.result()))
          print('Accuracy = ',self.Accuracy(model,neg,pos))
          if np.abs(train_loss_results[epoch]-val_loss_results[epoch])>0.9:
            print(False)
            #break
    self.Projection=model

  def train_Intersection(self):
    train_loss_results=[]
    val_loss_results=[]
    num_epochs=4000
    batch_size,batch_size_val,input_shape,data=self.split_Data(Graph_.feed_Intersection(self.Projection.P,self.Projection.Z))
    train_neg,train_pos,val_neg,val_pos=data
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    model=Intersection(d,output_shape=100)
    for epoch in range(num_epochs):
      epoch_loss_avg=tfe.metrics.Mean()
      epoch_val_avg=tfe.metrics.Mean()
      neg_=next(iter(train_neg.shuffle(4).batch(batch_size,True)))
      pos_=next(iter(train_pos.shuffle(4).batch(batch_size,True)))        
      loss_value, grads = self.grad(model, neg_,pos_,loss=self.mean_margin_loss)
      #grads_and_vars = optimizer.minimize(batch_loss(model,neg[:1000],pos[:1000])
      #                                     ,model.trainable_variables)
      #print(grads_and_vars)
      epoch_loss_avg(loss_value.numpy())
      if loss_value.numpy()==0:
        continue 
      
      optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
      val_neg_=next(iter(val_neg.shuffle(4).batch(batch_size_val,True)))
      val_pos_=next(iter(val_pos.shuffle(4).batch(batch_size_val,True)))
      epoch_val_avg(self.mean_margin_loss(model,val_neg_,val_pos_))#Accuracy(model,val_neg,val_pos))
      
      #print('Val_Loss=',mean_margin_loss(model,val_neg_,val_pos_).numpy())
      train_loss_results.append(epoch_loss_avg.result())
      val_loss_results.append(epoch_val_avg.result())
      #print(epoch)
      if epoch % 50 == 0:
        print("Epoch {:03d}: Avg_Train_Loss: {:.3f} : Avg_Val_Loss:{:.3f} ".format(epoch,epoch_loss_avg.result()
                                                                   ,epoch_val_avg.result()))
        print('Accuracy=',self.Accuracy(model,val_neg_,val_pos_))
        #if np.abs(train_loss_results[epoch]-val_loss_results[epoch])>0.2:
        #  print(False)
        #  break
    self.Intersection=model
    
if __name__=='__main__':
    #fold='bio_data'
    fold='GrankAIDataSet'
    G=KGraph(fold)
    G.trim_Graph(percentage=0.1)
    
    Train_=Train(G,query_type=2,embedding_space=128)
    Train_.train_Projection()
    Train_.train_Intersection()

    em=G.generate_Embbedings(Train_.Projection,Train_.Intersection)
    em_query=G.generate_Testing_Embbedings(Train_.Projection,Train_.Intersection,query_type=2)
    with open('TrainedEmbe.pkl','wb') as f:
        pickle.dump(em,f)
    with open('QueryEmbe.pkl','wb') as f:
        pickle.dump(em_query,f)

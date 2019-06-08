import random as rn
import numpy as np
import math
from multiprocessing import Pool, Manager
from functools import reduce
from Graph import *
from Visualization import *
class LSH():
    def __init__(self,data,bucket_size,radius):
        global n,d,r,c,k,m,band,size_of_buckets
        n=len(data)
        d=len(data[0])
        r=radius#d/2
        c=3/2
        k=int(math.log(n,2)/2)
        m=1
        size_of_buckets=bucket_size
        band=1
        self.data=data
    def reduce_(self,x,y):
            s=[]
            for i in zip(x,y):
                    s.append(i[0]+i[1])
            return s
    def H(self):
            b=rn.randint(0,size_of_buckets)
            u=tf.random_normal(shape=(1,d))
            return [u,b]
    def create_Signatures(self):
        Sign_Data=[] 
        H_Family=[self.H() for i in range(k)]
        for i in range(m):
            #H_Family=rn.sample(H_Family,k)
            Hashed_Row=[]
            for item in self.data:
                H_Family=rn.sample(H_Family,k)
                Hashed_Row.append([tf.ceil((tf.matmul(h[0],item)+h[1])/size_of_buckets).numpy()[0][0] for h in H_Family])
            Sign_Data.append(Hashed_Row)
            
        if band==m:
            return Sign_Data
        else:
            bands=[]
            size=m//band
            for i in range(0,len(Sign_Data),size):
                reduced_Data=reduce(lambda x,y:self.reduce_(x,y),Sign_Data[i:i+size])
                bands.append(reduced_Data)
            return bands

    def Hash_To_Buckets(self):
        Buckets=dict()
        M=self.create_Signatures()
        for row in M:
            for i in range(len(row)):
                keys=row[i]
                for key in keys:
                    if key not in Buckets.keys():
                        Buckets[key]=[self.data[i]]
                    else:
                        Buckets[key]+=[self.data[i]]
        return Buckets

    def NN_Bucket(self,query,collisions):
        Buckets=self.Hash_To_Buckets()
        NN={}
        for col in range(collisions):
            H_Family=[self.H() for i in range(k)]
            query_signs=[tf.ceil((tf.matmul(h[0],query)+h[1])/size_of_buckets).numpy()[0][0] for h in H_Family]
            for sign in query_signs:
                print(sign)
                if sign in Buckets.keys() and sign not in NN.keys():
                    NN[sign]=Buckets[sign]

        det=[]
        for nn in list(NN.values()):
            for item in nn:
                det.append([item,tf.norm(item-query)])
        return det
    def Nearest_Neighbors(self,query):
        Tables=self.Hash_To_Buckets()
        all_NN=[]
        collision=0
        H_Family=[self.H() for i in range(k)]
        query_signs=[tf.ceil((tf.matmul(h[0],query)+h[1])/size_of_buckets).numpy()[0][0] for h in H_Family]
        for sign,bucket in zip(query_signs,Tables):
            if sign in bucket.keys():
                NN=[]
                neigs=bucket[sign]
                for neigh in neighs:
                    dist=np.norm(np.abs(np.array(query)-np.array(neigh)))
                    if dist<r:
                        NN.append(neigh)
                if NN==[]:
                    collision+=1
                else:
                    all_NN.append(NN)
            if collision==k:
                break
        return all_NN
    
if __name__=='__main__':
    #Inference using LSH or Linear Search
    with open('TrainedEmbe.pkl','rb') as f:
        em_trained=pickle.load(f)
    with open('QueryEmbe.pkl','rb') as f:
        em_query=pickle.load(f)
    
    embedings=list(em_trained.values())
    keys=list(em_trained.keys())
    query=[]
    for x in em_query:
        print(x[0],'---',x[1])
        query.append([x[1],x[0]])
    det=[]
    for x in embedings:
        det.append([x,tf.norm(x-query[0][0])])
          
    det_s=sorted(det,key=lambda x:x[1])  
    det_ss=list(map(lambda x:keys[embedings.index(x[0])],det_s))
    print('Keys of Nearest embedings --> ',det_ss[:len(query[0][-1][-1])],' \nGround truth --->',query[0][-1][-1])
                
    Visualize_querytype2(query[0][1][:-1]+[(query[0][1][0][1],'?')],det_ss[:10],query[0][-1][-1])

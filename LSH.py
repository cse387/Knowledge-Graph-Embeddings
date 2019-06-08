import random as rn
import numpy as np
import math
from multiprocessing import Pool, Manager
from functools import reduce
n=2**10
d=2**7
r=d/2
c=3/2
k=int(math.log(n,2)/2)
m=100#best 100
band=100
data=[[rn.randint(0,1) for i in range(d)] for j in range(n)]
query=data[0]

def linear_search():
    NN=[]
    FP=[]
    ANN=[]
    for item in data:
        dist=sum(np.abs(np.array(query)-np.array(item)))
        if dist<r:
                NN.append(item)
        elif r<=dist<=c*r:
                ANN.append(item)
        elif dist>=c*r:
                FP.append(item)
    return [NN,ANN,FP]

result=linear_search()
print('Linear results:\nNumber of NN with dist < r: ',len(result[0]),
      '\nNumber of NN with dist between [r,cr]: ',len(result[1]),
      '\nFalse Positives with dist > cr: ',len(result[2]))

def bin_to_dec(Bin):
    s=0
    for i in range(len(Bin)):
        if Bin[i]!=0:
            s+=2**i
    return s%5#best5
 
def reduce_(x,y):
	s=[]
	for i in zip(x,y):
		s.append(i[0]+i[1])
	return s
    
def create_Signatures():
    Sign_Data=[]
    def H(x,i):
        return x[i]
    H_Family=[i for i in range(k)]
    for i in range(m):
        H_Family=rn.sample(H_Family,k)
        Hashed_Row=[]
        for item in data:
            #H_Family=rn.sample(H_Family,k)
            Hashed_Row.append([H(item,j) for j in H_Family])
        Sign_Data.append(Hashed_Row)
        
    if band==m:
        return Sign_Data
    else:
        bands=[]
        size=m//band
        for i in range(0,len(Sign_Data),size):
            reduced_Data=reduce(lambda x,y:reduce_(x,y),Sign_Data[i:i+size])
            bands.append(reduced_Data)
        return bands

def Hash_To_Buckets():
    Bucket=dict()
    M=create_Signatures()
    for row in M:
        for i in range(len(row)):
            key=bin_to_dec(row[i])
            if key not in Bucket.keys():
                Bucket[key]=(data[i],)
            elif data[i] not in Bucket[key]:
                Bucket[key]+=(data[i],)
    return Bucket,M

Table,M=Hash_To_Buckets()
query_keys=list(map(lambda x:x[0],M))
s=[]
for i in query_keys:
    if i not in s:
        s.append(i)
query_keys=s
#random_query_key=query_keys[rn.randint(0,len(s)-1)]

def Compare_Bucket(key,q,r,cr):
    NN=[]
    FP=[]
    ANN=[]
    for item in Table[bin_to_dec(key)]:
        dist=sum(np.abs(np.array(q)-np.array(item)))
        if dist<r and item not in NN:
            NN.append(item)
        elif r<=dist<=cr and item not in ANN:
            ANN.append(item)
        elif dist>cr and item not in FP:
            FP.append(item)
    return [NN,ANN,FP]
Max=[]
for key in  query_keys:
    result=Compare_Bucket(key,query,r,c*r)
    Max.append((len(result[0]),len(result[1]),len(result[2])))
print(max(Max))
print('LSH results:\nNumber of NN with dist < r: ',len(result[0]),
      '\nNumber of NN with dist between [r,cr]: ',len(result[1]),
      '\nFalse Positives with dist > cr: ',len(result[2]))
            
        

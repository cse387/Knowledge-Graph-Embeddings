import tensorflow as tf
import glob
from multiprocessing import Pool, Manager ,freeze_support
from multiprocessing.pool import ThreadPool
import random as rn
import numpy as np
import copy
from Training import *
    
class KGraph():
        
    def __init__(self,fold):
        data=self.read_files(fold)
        Sets=self.get_neighs(data)
        
        global number_Anc
        global number_Var
        
        Anc={}
        Var={}
        Target={}
        keys=list(Sets[1].keys())
        for var in keys:
                if len(Sets[1][var][0])>1:
                        for anc in Sets[1][var][0]:
                                if anc not in Anc.keys():
                                        Anc[anc]=[var]####buggg
                                        del Sets[0][anc]
                                if len(Anc.keys())>4:###
                                    break
                        if var not in Var.keys():
                                Var[var]=Sets[1][var]             
                if len(Var.keys())>20:
                    break
        
        number_Anc=len(list(Sets[0].keys()))
        number_Var=len(list(Sets[1].keys()))
        self.Sets=copy.deepcopy(Sets)
        self.Test_Sets=[Anc,Var]
        
    def read_files(self,fold):
        files=glob.glob(fold+'/*')
        data=open(files[0],'r')
        data=data.read().strip().split('\n')
        #data=list(map(lambda x:x.split('\t'),data))
        data=list(map(lambda x:x.split('\t')[1:4],data[1:200000]))
        def swap(x):
            temp=x[1]
            x[1]=x[2]
            x[2]=temp
            return x
        #data=list(map(lambda x:swap(x),data))
        return data

    def help_(self,data,Anc,Var1,Target):
        for x in data:
            if x[0] not in Anc.keys():
                Anc[x[0]]=[x[1]]
            else:
                if x[1] not in Anc[x[0]]:
                    Anc[x[0]]+=[x[1]]
            if x[1] not in Var1.keys():
                Var1[x[1]]=[[x[0]],[x[2]]]
            else:
                if x[0] not in Var1[x[1]][0]:
                    Var1[x[1]][0]+=[x[0]]
                if x[2] not in Var1[x[1]][1]:
                    Var1[x[1]][1]+=[x[2]]
            if x[2] not in Target.keys():
                Target[x[2]]=[x[1]]
            else:
                if x[1] not in Target[x[2]]:
                    Target[x[2]]+=[x[1]]
                    
    def get_Features(self,Set,size):
        Set=copy.deepcopy(Set)
        embedding=[0]*size
        Emb={}
        for x in Set.keys():
            feature=[0]*size
            index=hash(x)%size
            while(1):
                index=(index+1)%size
                if embedding[index]==0:
                    embedding[index]=1
                    feature[index]=1
                    Set[x]+=[feature]
                    break
        return Set

    def get_number_of_Edges(self):
        Sets=self.Sets
        number=0
        if len(Sets)>=3:
            for i,s in enumerate(Sets):
                if i==len(Sets)-1:
                    break 
                if i==0:
                    for key in s.keys():
                        number+=len(s[key])
                else:
                    for key in s.keys():
                        number+=len(s[key][1])
            return number
        else:
            for i,s in enumerate(Sets):
                if i==len(Sets)-1:
                    break
                for key in s.keys():
                    number+=len(s[key][0])
            return number
        
    def get_number_of_Nodes(self):
            Sets=self.Sets
            return sum([len(list(Sets[0].keys())),len(list(Sets[1].keys())),len(list(Sets[2].keys()))])
        
    def trim_Graph(self,percentage):
        Sets=self.Sets
        Edge_number=self.get_number_of_Edges()
        Deleted_all=[]
        count=0
        for i,s in enumerate(Sets):
            if i==len(Sets)-1:
                break
            elif i==0:
                Deleted=[]
                for key in s.keys():
                    length=len(s[key])
                    if length>1:
                        prob=[rn.random() for i in range(length)]
                        prob=list(map(lambda x:prob.index(x),filter(lambda x:x<percentage,prob)))
                        if prob==[]:
                            continue
                        deleted_values=[]
                        for index in prob:
                            if len(Sets[i+1][s[key][index]][0])>=2:
                                deleted_values.append(s[key][index])
                                s[key][index]=None
                        s[key]=list(filter(lambda x:x!=None,s[key]))
                        Deleted.append([key,deleted_values])
                Deleted_all.append(Deleted)
            else:
                Deleted=[]
                for key in s.keys():
                    length=len(s[key][1])
                    if length>1:
                        prob=[rn.random() for j in range(length)]
                        prob=list(map(lambda x:prob.index(x),filter(lambda x:x<percentage,prob)))
                        if prob==[]:
                            continue
                        deleted_values=[]
                        for index in prob:
                            if len(Sets[i+1][s[key][1][index]])>=2:
                                deleted_values.append(s[key][1][index])
                                s[key][1][index]=None
                        s[key][1]=list(filter(lambda x:x!=None,s[key][1]))
                        Deleted.append([key,deleted_values])
                Deleted_all.append(Deleted)
                
        for i in range(len(Deleted_all)):
            s=Sets[i+1]
            Deleted=Deleted_all[i]
            if Deleted!=[]:
                for x in Deleted:
                    for key in x[1]:
                        if key==[None]:
                            continue
                        elif key in s.keys():
                            if i==len(Sets)-2:
                                if len(s[key])==1:
                                    #index=s[key][0].index(x[0])
                                    #del s[key][0][index]
                                    s[key]=[None]
                                else:
                                    index=s[key].index(x[0])
                                    del s[key][index]
                            else:
                                if len(s[key][0])==1:
                                    #index=s[key][0].index(x[0])
                                    #del s[key][0][index]
                                    s[key][0]=[None]
                                else:
                                    index=s[key][0].index(x[0])
                                    del s[key][0][index]

    def sample_Index(self,weights):
        index=np.random.choice(len(weights),1,p=weights)[0]
        return index

    def sample_TargetNode(self,set_):
        keys=list(set_.keys())
        values=list(set_.values())
        weights=list(filter(lambda x:None not in x,values))
        weights=list(map(lambda x:len(x),weights))
        weights=list(map(lambda x:x/sum(weights),weights))
        index=0
        while(1):
            index=self.sample_Index(weights)
            if set_[keys[index]][0]!=None:
                break
        return set_[keys[index]]

    def sample_VariableNode(self,set_,Var_Keys,query_type):
        k=0
        index=0
        weights=[set_[key][0] for key in Var_Keys]
        weights=list(filter(lambda x:None not in x,weights))
        weights=list(map(lambda x:len(x),weights))
        weights=list(map(lambda x:x/sum(weights),weights))
        if len(weights)==0:
            print(weights)
            import sys
            sys.exit()
            
        while(k!=len(Var_Keys)):
            index=self.sample_Index(weights)
            if len(set_[Var_Keys[index]][0])>=query_type:
                break
            k+=1
        if k==len(Var_Keys):
            return False
        else:
            return set_[Var_Keys[index]][0]

    def sample_AnchorNode(self,Anc,query_type):
        if len(Anc)==query_type:
            return Anc
        else:
            sample=[]
            while(len(sample)<query_type):
                index=rn.randint(0,len(Anc)-1)
                if Anc[index] not in sample:
                    sample.append(Anc[index])
            return sample
        
    def rejection_Sampling(self,query_type):
        Pos_Keys=[]
        Sets=self.Sets
        i=len(Sets)-1
        iter_=0
        while(len(Pos_Keys)<=len(Sets[0].keys())//2 and iter_!=1000):
            Var_Keys=self.sample_TargetNode(Sets[i])
            Anc_Keys=self.sample_VariableNode(Sets[i-1],Var_Keys,query_type)
            if Anc_Keys!=False:
                Anc_Keys=self.sample_AnchorNode(Anc_Keys,query_type)
                len(Anc_Keys)
                for key in Anc_Keys:
                    if key not in Pos_Keys:
                        Pos_Keys.append(key)
            iter_+=1
        return Pos_Keys
        
    def feed_Projection(self,query_type):
        Anc=self.get_Features(self.Sets[0],number_Anc)
        Var=self.get_Features(self.Sets[1],number_Var)
        Pos_Keys=self.rejection_Sampling(query_type)
        Neg_Samples=[]
        Pos_Samples=[]
        Paths=dict()
        for key in Anc.keys():
            for var in Anc[key][:-1]:
                if var in Var.keys():
                    feature_emb=Anc[key][-1]+Var[var][-1]
                    if key in Pos_Keys:
                        Pos_Samples.append(feature_emb)
                    else:
                        Neg_Samples.append(feature_emb)
                    if (key,var) not in Paths.keys():
                        Paths[(key,var)]=tf.convert_to_tensor(feature_emb,dtype=tf.float32)
                    else:
                        Paths[(key,var)]+=tf.convert_to_tensor(feature_emb,dtype=tf.float32)
        self.Paths=Paths
        self.Pos_Keys=Pos_Keys
        return Pos_Samples,Neg_Samples
        
    def feed_Intersection(self,P,Z):
        Pos_Samples=[]
        Neg_Samples=[]
        Intersection=dict()
        NonIntersection=dict()
        for key in self.Paths.keys():
            feature=tf.reshape(self.Paths[key],shape=(self.Paths[key].shape[0],1))
            projection=tf.matmul(P,tf.matmul(Z,feature))
            if len(self.Sets[1][key[1]][0])>1:                    
                if key[0] in self.Pos_Keys:
                    Pos_Samples.append(projection)
                else:
                    Neg_Samples.append(projection)
                if key[1] not in Intersection.keys():
                    Intersection[key[1]]=[projection]
                else:
                    Intersection[key[1]]+=[projection]#Apply the interaction operation
            else:
                if key[1] not in NonIntersection.keys():
                    NonIntersection[key[1]]=[projection]
                else:
                    NonIntersection[key[1]]+=[projection]
        self.Intersection=Intersection
        self.NonIntersection=NonIntersection
        return Pos_Samples,Neg_Samples        

    def collect_Embbedings(self,proj,inter):
        Intersection=copy.deepcopy(self.Intersection)
        NonIntersection=self.NonIntersection
        for key in Intersection.keys():
            Intersection[key]=tf.matmul(proj.P,inter.evaluate(Intersection[key]))#inter.evaluate(Intersection[key])
        for key in NonIntersection.keys():
            NonIntersection[key]=NonIntersection[key][0]#tf.matmul(proj.P,NonIntersection[key][0])#Apply Projections?

        Gen_Emb=copy.deepcopy(self.Sets[2])
        for key in Gen_Emb.keys():
            embeddings=[]
            Gen_Emb[key]=list(filter( lambda x:x!=None,Gen_Emb[key]))
            for neig in Gen_Emb[key]:
                if neig in Intersection.keys():
                    embeddings+=[Intersection[neig]]
                elif neig in NonIntersection.keys():
                    embeddings+=[NonIntersection[neig]]
            Gen_Emb[key]=embeddings
        return Gen_Emb
    
    def generate_Embbedings(self,proj,inter):
        em=self.collect_Embbedings(proj,inter)
        for key in em.keys():
            if len(em[key])>1:
                em[key]=tf.matmul(proj.P,inter.evaluate(em[key]))
        ret={}
        for k in em.keys():
            if k not in ret.keys():
                if len(em[k])==1:
                        ret[k]=em[k][0]
                elif em[k]==[]:
                        continue
                else:
                        ret[k]=em[k]
        return ret

    def generate_Testing_Embbedings(self,proj,inter,query_type):
            #extract a query from an unseen stucture
            Anc=copy.deepcopy(self.Test_Sets[0])
            Var=copy.deepcopy(self.Test_Sets[1])
            Query_Var={}
            for k in Anc.keys():
                    for var in Anc[k]:
                            if var not in Query_Var.keys():
                                    Query_Var[var]=1
                            else:
                                    Query_Var[var]+=1
            query=[]
            for k in Query_Var.keys():
                    temp=[]
                    if Query_Var[k]>=query_type:
                            ancs=Var[k][0]
                            target=Var[k][1]
                            while(len(temp)<query_type):
                                index_=rn.randint(0,len(ancs)-1)
                                if ancs[index_] in Anc.keys():
                                    temp.append((ancs[index_],k))
                            query.append([ancs for ancs in temp]+[target])
            embedings=copy.deepcopy(query)
            query=list(map(lambda x:x[:-1],query))
            Anc=self.get_Features(Anc,number_Anc)
            Var=self.get_Features(Var,number_Var)
            for i in range(len(query)):
                    for j in range(len(query[i])):
                            feature=Anc[query[i][j][0]][-1]+Var[query[i][j][1]][-1]
                            shape=(len(feature),1)
                            feature=tf.convert_to_tensor(feature,dtype=tf.float32)
                            feature=tf.reshape(feature,shape)
                            feature=tf.matmul(proj.P,tf.matmul(proj.Z,feature))
                            query[i][j]=feature
                                
            for i in range(len(query)):
                    if len(query[i])>1:
                        query[i]=tf.matmul(proj.P,inter.evaluate(query[i]))#inter.evaluate(query[i])
                
            return zip(embedings,query)
        
    def get_neighs(self,data):
        Anc={}#Manager().dict()
        Var1={}
        Target={}
        p=Pool(8)
        p.apply_async(self.help_(data,Anc,Var1,Target))
        p.close()
        p.join()
        #self.help_(data,Anc,Var1,Target)
        return Anc,Var1,Target

if __name__=='__main__':
    freeze_support()
    fold='bio_data'
    #fold='GrankAIDataSet'
    G=KGraph(fold)
    #import sys
    #sys.exit()
    print('trimming')
    G.trim_Graph(percentage=0.1)
    print('Feeding-Projection')
    pos,neg=G.feed_Projection(query_type=2)
    proj=Embbed_Projection(128,1306)
    print('Feeding-Intersection')
    pos,neg=G.feed_Intersection(proj.P,proj.Z)
    inter=Intersection(128,100) 
    em=G.generate_Embbedings(proj,inter)
    em_t=G.generate_Testing_Embbedings(proj,inter,query_type=2)
    #Inference using LSH or K-means

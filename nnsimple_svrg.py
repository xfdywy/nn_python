# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 21:57:55 2016

@author: v-yuewng
"""



import numpy as np
import os 


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def readtraindata():
    traindata =  np.array(unpickle('data_batch_1')['data'])
    trainlabel = np.array(unpickle('data_batch_1')['labels'])
    for ii in range(2,6):
        traindata = np.vstack((traindata, np.array(unpickle('data_batch_'+str(ii))['data'])))
        trainlabel = np.hstack((trainlabel,np.array(unpickle('data_batch_'+str(ii))['labels'])))
    return traindata,trainlabel

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    
    return (1.0/(1+np.exp(-x)))

def softmax(x,deriv = False):
        
    return np.exp(x)/np.sum(np.exp(x))

def softmaxloss(x,y,deriv = False):
    return -1*np.log(x[0,y])
    
    

def ininet():
    # randomly initialize our weights with mean 0
    np.random.seed(1)
    syn0 = 2*np.random.random((numfeature,numconvolusionlayer)) - 1
    syn1 = 2*np.random.random((numconvolusionlayer,numoutputlayer)) - 1
    b0 = 2*np.random.random((1,numconvolusionlayer)) - 1
    b1 = 2*np.random.random((1,numoutputlayer)) - 1
    return syn0,syn1,b0,b1


def forwardstep(data,syn0,syn1,b0,b1):
    ndata = (data.shape[0])
    nh = syn0.shape[1]
    no = syn1.shape[1]
    l1 = np.zeros((ndata,nh))
    l2 = np.zeros((ndata,no))
    for ii in range(data.shape[0]):
        l0 = data[ii,:]
        l1[ii,:] = nonlin(np.dot(l0,syn0)+b0)
        l2[ii,:] = softmax(np.dot(l1[ii,:],syn1)+b1)
    return l1,l2
        
    

def datatest(data,label,syn0,syn1,b0,b1):
    l2_error= np.zeros((1,data.shape[0]))
    for ii in range(data.shape[0]):
        l0 = data[ii,:]
        l1 = nonlin(np.dot(l0,syn0)+b0)
        l2 = softmax(np.dot(l1,syn1)+b1)
        l2_error[0,ii] = softmaxloss(l2,int(label[ii]))
    return(l2_error)


def diffho(label,l1,l2):
    resw = np.zeros((l1.shape[0],l1.shape[1],l2.shape[1]))
    resb = np.zeros((l1.shape[0],l2.shape[1]))
    for ii in range(l1.shape[0]):
        y = l1[ii,:]
        delta = l2[ii,:]
        delta[label[ii]]-=1 
        
        delta.shape = (1,10)   
        y.shape = (100,1)
        
        resw[ii,:,:] = np.dot(y,delta)
        resb[ii,:] = delta
        
    return np.mean(resw,0),np.mean(resb,0)
    
def diffih(label,l0,l1,syn1):
    resw = np.zeros((l0.shape[1],l1.shape[1]))
    resb = np.zeros((1,l1.shape[1]))
    
    for ii in range(l0.shape[0]):
        y=l1[ii,:]
        first = y*(1-y)
        first.shape = (100,1)
        
        delta = l2[ii,:]
        delta[label[ii]]-=1        
        delta.shape = (10,1)      
        
        second = syn1.dot(delta)
        deltaj =  first*second
        deltaj.shape = (1,100)
        
        x = l0[ii,:]
        x.shape = ((3072,1))
        resw += np.dot(x,deltaj)
        resb += np.transpose(first*second)
        
    
    return resw/(l0.shape[0]+0.0),resb/(l0.shape[0]+0.0)


    


    
os.chdir(r'D:\project\mynn\cifar-10-batches-py')

traindata,trainlabel = readtraindata()


testdata = np.array(unpickle('test_batch')['data'])
testlabel = np.array(unpickle('test_batch')['labels'])

numfeature = traindata.shape[1]
numalldata = traindata.shape[0]

numinputlayer = numfeature
numconvolusionlayer = 100
numoutputlayer = 10

inneriter = 10000



outputfile = open('nn_svrg.csv','w')
output = 'trainloss'+','+'testloss'+','+'passeddata'+'\n'
outputfile.write(output)

syn0,syn1,b0,b1 = ininet()

#traindata = traindata[0:50000,:]
#trainlabel = trainlabel[0:50000]

for j in xrange(80000):
    
        
    
    l0 = traindata
    l1,l2 = forwardstep(l0,syn0,syn1,b0,b1)
 
    
    if(j==0):
        trainerror = datatest(traindata,trainlabel,syn0,syn1,b0,b1)
        testerror = datatest(testdata,testlabel,syn0,syn1,b0,b1)
        output = str(trainerror.mean())+','+ str(testerror.mean())+','+str((j)*l0.shape[0])+'\n'
        outputfile.write(output)
        print(output)
        

    deltawho_out,deltabho_out = diffho(trainlabel,l1,l2)
    deltawih_out,deltabih_out = diffih(trainlabel,l0,l1,syn1)  
    
    syn0_in,syn1_in,b0_in,b1_in = syn0,syn1,b0,b1
    
    for jj in range(inneriter):
        minibatchindex = np.random.randint(0,traindata.shape[0],1)
        l0 = traindata[minibatchindex,:]   
        label = trainlabel[minibatchindex]
        
        l1_new,l2_new = forwardstep(l0,syn0_in,syn1_in,b0_in,b1_in)        
        
        deltawho_innew,deltabho_innew = diffho(label,l1_new,l2_new)
        deltawih_innew,deltabih_innew = diffih(label,l0,l1_new,syn1_in)  
        
        
        
        l1_old,l2_old = forwardstep(l0,syn0,syn1,b0,b1) 
        
        
        
        deltawho_inold,deltabho_inold = diffho(label,l1_old,l2_old)
        deltawih_inold,deltabih_inold = diffih(label,l0,l1_old,syn1)  
        
        #print deltawho_innew.max()   
        #print deltawih_innew.max()
        
        syn1_in -= 0.00003*(deltawho_innew - deltawho_inold + deltawho_out)
        b1_in -= 0.00003*(deltabho_innew - deltabho_inold + deltabho_out)
        syn0_in -= 0.00003*(deltawih_innew - deltawih_inold + deltawih_out)
        b0_in -= 0.00003*(deltabih_innew - deltabih_inold + deltabih_out)
#        
#        print deltawho_inold.max(),deltawho_innew.max(),deltawho_out.max()
#        syn1_in -= 0.01*(deltawho_innew)
#        b1_in -= 0.01*(deltabho_innew )
#        syn0_in -= 0.01*(deltawih_innew )
#        b0_in -= 0.01*(deltabih_innew )        
#        
        
        if(jj% 10000==0):
            trainerror = datatest(traindata,trainlabel,syn0_in,syn1_in,b0_in,b1_in)
            testerror = datatest(testdata,testlabel,syn0_in,syn1_in,b0_in,b1_in)
            output = str(trainerror.mean())+','+ str(testerror.mean())+','+str(traindata.shape[0]+(j)*(traindata.shape[0]+inneriter)+jj)+'\n'
            outputfile.write(output)
            print(output)
            
        
    syn0,syn1,b0,b1 = syn0_in,syn1_in,b0_in,b1_in
    
    
outputfile.close()  

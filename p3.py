# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:37:26 2020

@author: yinz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class training():
    def __init__(self,inputs,labels,midlayer_size,output_size,learning_rate,iteration,):
        #np.random.seed(0)

        self.inputs = inputs 
        self.labels = labels 
        self.weight_hide = np.random.randn(inputs.shape[1], midlayer_size) 
        self.weight_out = np.random.randn(midlayer_size,output_size)

        self.learning_rate = learning_rate
        self.threshold_hide = np.random.rand(midlayer_size)
        self.threshold_out = np.random.rand(output_size)
        min_val = -2.4/inputs.shape[1]
        max_val = 2.4/inputs.shape[1]
        
        self.threshold_hide = min_val + (self.threshold_hide *(max_val - min_val))
        self.threshold_out = min_val + (self.threshold_out *(max_val - min_val))

        self.iteration = iteration

    def sigmod(self,x):
        return 1/(1+np.exp(-x))
        
    def sigmoid_prime(self,x):
        return (x)*(1.0-x)

    def feedforward(self):

        self.midlayer = self.sigmod(np.dot(self.inputs,self.weight_hide) - self.threshold_hide)
       # print ("midlayer",self.midlayer.shape)
        self.outputlayer = self.sigmod(np.dot(self.midlayer,self.weight_out )- self.threshold_out) 
       # print ("outputlayer",self.outputlayer.shape)
  
        
    def weight(self):
        
        error_out = (self.labels - self.outputlayer) * self.sigmoid_prime(self.outputlayer) 
       # print ("error_out",error_out.shape)
        
        gradient_descent_output = np.dot(self.midlayer.T,error_out) 
       # print ("gradient_descent_output",gradient_descent_output.shape)
        delta_weight_out = np.dot(self.learning_rate,gradient_descent_output) 
       # print ("delta_weight_out",delta_weight_out.shape)
        #delta_threashold_out = np.dot(self.learning_rate,np.dot(self.threshold_out.T,error_out)) 
       # print ("delta_threashold_out",delta_threashold_out.shape)
        
        error_hide = np.dot(error_out,self.weight_out.T) * (self.sigmoid_prime(self.midlayer))
       # error_hide = np.dot(self.sigmoid_prime(self.midlayer),np.dot(self.weight_out,error_out)) 
      #  print ("error_hide",error_hide.shape)
        gradient_descent_hide = np.dot(self.inputs.T,error_hide) 
     
       # print ("gradient_descent_hide",gradient_descent_hide.shape)
        delta_weight_hide = np.dot(self.learning_rate,gradient_descent_hide)
       # print ("delta_weight_hide",delta_weight_hide.shape)
        
        #delta_threashold_hide =  np.dot(self.learning_rate,np.dot(self.threshold_hide,error_hide))

        self.weight_out = self.weight_out + delta_weight_out
        #self.threshold_out = self.threshold_out + delta_threashold_out
        
        
        self.weight_hide = self.weight_hide + delta_weight_hide
        #self.threshold_hide=self.threshold_hide + delta_threashold_hide
         
    def evaulation(self):
        accuracy = np.argmax(self.outputlayer,1) == np.argmax(self.labels,1)
        accuracy = str(int(np.count_nonzero(accuracy)/10*100))+"%"
        return (np.sum(np.power((self.labels - self.outputlayer),2))),accuracy
        
      
    def test_result(self,testData):

        midlayer = self.sigmod(np.dot(testData,self.weight_hide) - self.threshold_hide)
       # print ("midlayer",self.midlayer.shape)
        result = self.sigmod(np.dot(midlayer,self.weight_out )- self.threshold_out) 
        return (result)
       # print ("outputlayer",self.outputlayer.shape)
        
    def main(self,testData,testLabel):
        results = []
        for i in range (self.iteration):
            self.feedforward()
            self.weight()
            result = self.evaulation()
            #if result[1] == "100%":
           #     break
            results.append(result)

            
        sum_square_error,accuracy = zip(*results)
        plt.plot(sum_square_error)
        plt.xlabel("Epochs")
        plt.ylabel("Sum Square Error")
        plt.show()
        
        
        correct = 0;

        for i in range (np.size(testLabel)):
            testint_result = self.test_result(testData[i])
            if (np.argmax(testint_result)) == testLabel[i]:
                correct += 1
                
        print ("The accruacy of the ANN is %f", correct/np.size(testLabel) * 100 )
        
    
        
class data_set():
    
    
    def data(self):
        d1=[3,7,8,11,13,18,23,28,33,38,43]
        d2=[2,3,4,6,10,15,20,24,28,32,36,41,42,43,44,45]
        d3=[2,3,4,6,10,15,20,24,30,35,36,40,42,43,44]
        d4=[4,8,9,13,14,18,19,22,24,26,29,31,32,33,34,35,39,44]
        d5=[1,2,3,4,5,6,11,16,17,18,19,21,25,30,35,36,40,42,43,44]
        d6=[2,3,4,6,10,11,16,21,22,23,24,26,30,31,35,36,40,42,43,44]
        d7=[1,2,3,4,5,10,14,19,23,28,33,38,43]
        d8=[2,3,4,6,10,11,15,16,20,22,23,24,26,30,31,36,36,40,42,43,44]
        d9=[2,3,4,6,10,11,15,16,20,22,23,24,25,30,35,36,40,42,43,44]
        d0=[2,3,4,6,10,11,16,21,26,31,36,42,43,44,15,20,25,30,35,40]
        
        data_list=[d0,d1,d2,d3,d4,d5,d6,d7,d8,d9]
        d_shape = np.zeros(45)
        data_set=[]
        for j in data_list:
            
            d_shape = np.zeros(45)
            
            for i in j:
                
                d_shape[i-1] = 1
           # d_shape = d_shape.reshape((-1,1))
            data_set.append(d_shape)
        data_set = np.asanyarray(data_set) 
        #data_set = data_set.T
        label_set = np.zeros([10,10])
   
     
        
        for i in range(10):
            label_set[i][i]=1
           # label = np.zeros(10)
         #   label[i]=1
         #   label = label.reshape(-1,1)
         #   label_set.append(label)
            
        #label_set = np.asanyarray(label_set)
        return data_set,label_set
        

class test_set():
    
    
    def data(self):

        
        d1=[3,7,8,11,13,18,23,28,33,38,43]
        d2=[4,7,8,11,13,18,23,28,33,38,43]
        d3=[3,7,8,11,13,23,28,33,38,43]
        d4=[2,3,4,6,10,15,20,24,28,32,36,41,42,43,44,45]
        d5=[2,3,4,6,10,15,24,28,32,36,41,42,43,44,45]
        d6=[2,3,4,6,10,15,20,24,28,32,36,42,43,44,45]
        d7=[2,3,4,6,10,15,20,24,30,35,36,40,42,43,44]
        d8=[2,3,4,6,15,20,24,30,35,36,40,42,43,44]
        d9=[2,3,4,6,10,15,20,24,30,35,38,40,42,43,44]
        d10=[4,8,9,13,14,18,19,22,24,26,29,31,32,33,34,35,39,44]
        d11=[4,8,9,13,14,18,22,24,26,29,31,32,33,34,35,39,44]
        d12=[4,8,9,13,14,18,19,22,24,26,29,30,32,33,34,35,39,44]
        d13=[1,2,3,4,5,6,11,16,18,19,21,25,30,35,36,40,42,43,44]
        d14=[1,2,3,4,5,6,11,16,17,18,19,21,25,30,35,36,40,42,43,44]
        d15=[1,2,3,4,5,6,11,16,17,18,19,21,25,30,35,39,40,42,43,44]
        d16=[2,3,4,6,10,11,16,21,22,23,24,26,30,31,35,36,40,42,43,44]
        d17=[2,3,4,6,10,16,21,22,23,24,26,30,31,35,36,40,42,43,44]
        d18=[2,3,4,6,10,11,16,21,22,23,24,26,30,31,35,39,40,42,43,44]
        d19=[1,2,3,4,5,10,14,19,23,28,33,38,43]
        d20=[1,2,3,4,5,10,14,19,23,28,33,43]
        d21=[1,2,4,5,10,14,19,23,28,33,38,43]
        d22=[2,3,4,6,10,11,15,16,20,22,23,24,26,30,31,36,36,40,42,43,44]
        d23=[2,3,4,10,11,15,16,20,22,23,24,26,30,31,36,36,40,42,43,44]
        d24=[2,3,4,6,10,11,15,16,20,22,23,25,26,30,31,36,36,40,42,43,44]
        d25=[2,3,4,6,10,11,15,16,20,22,23,24,25,30,35,36,40,42,43,44]
        d26=[2,3,4,6,10,11,15,16,20,22,23,25,30,35,36,40,42,43,44]
        d27=[2,3,4,6,9,11,15,16,20,22,23,24,25,30,35,36,40,42,43,44]
        d28=[2,3,4,6,10,11,16,21,26,31,36,42,43,44,15,20,25,30,35,40]
        d29=[2,3,4,6,10,16,21,26,31,36,42,43,44,15,20,25,30,35,40]
        d30=[2,3,4,6,10,11,16,21,26,31,36,42,43,44,17,20,25,30,35,40]
        

        test_set = []
        test_label_set=[]
        label = 1
       
        
        for i in range(30):
            temp_data = vars()["d"+str(i+1)]
            d_shape = np.zeros(45)
            for j in temp_data:
                d_shape[j-1] = 1

            test_set.append(d_shape)
                    
            test_label_set.append(label)
            if (i+1)%3 == 0:
                label += 1
        return test_set,test_label_set
    
get_data = data_set()
training_data,label_data = get_data.data()
testing = test_set()
testing_data,testing_label_data = testing.data()


output_test = [0,1,0,0,0,0,0,0,0,0]
#input_test = input_test.reshape(-1,1)
output_test = np.asanyarray(output_test)#.reshape(-1,1)
trainings = training(training_data,label_data,5,10,1.5,10000)
trainings.main(testing_data,testing_label_data)


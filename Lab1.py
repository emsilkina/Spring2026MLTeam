from math import log
import numpy as np
import matplotlib.pyplot as plt

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


#Some useful conversion functions
def ListtoVector(new_list):
    length = len(new_list)
    vec = np.arange(length)
    for i,v in enumerate(new_list):
        vec[i] = v
    return vec.reshape(length, 1)

def VectortoList(new_vec):
    length = (new_vec.size)
    reshaped = new_vec.reshape(1, length)
    new_list = list()
    for v in reshaped[0]:
        new_list.append(v)
    return new_list

#A function that setups your initial network including randomized weights, input is the form of a list of the number of neurons in each layer. e.g [784, x, y, 10] where x and y are the number of neurons in your hiddenn layers
def architecture(new_list):
    weights = list()
    biases = list()
    weights.append(None)
    biases.append(None)
    network_length = len(new_list)
    for c in range(network_length-1):
        weight_matrix = 2 * np.random.rand(new_list[c+1], new_list[c]) - 1
        bias_matrix = 2 * np.random.rand(new_list[c+1],1) - 1
        weights.append(weight_matrix)
        biases.append(bias_matrix)
    return weights, biases

#take in the CSVs and vectorize the output, would reccomend experimenting with to see exactly what happens
def read_file(file_name):
    toReturn = list()
    with open(file_name) as f:
        for line in f:
            image = line[0:len(line)-1].split(",")
            output = image.pop(0)
            in_vec = ListtoVector(image)
            out_vec = list()
            for c in range(10):
                if c == int(output):
                    out_vec.append(1)
                else:
                    out_vec.append(0)
            out_vec = ListtoVector(out_vec)
            toAppend = (in_vec,out_vec)
            toReturn.append(toAppend)
    return toReturn

#TODO A feed forward of the network where A_vec is the activation function, weights is a list of all the weight matrices, biases is a list of all the bias vectors, and inp is the input, return the output as a vector
def p_net(A_vec, weights, biases, inp):
    a=inp
    activationLayers=[a]
    i=0
    while i<len(weights):
        z=np.dot(weights[i],a)+biases[i]
        a=A_vec(z)
        activationLayers.append(a)
        i+=1
    return activationLayers

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    learningRate=0.1
    mse=0
    correct=0

    for x,y in training:
        activations=p_net(sigmoid, weights, biases, x)
        predicted=activations[-1]
        actual=x
        delta=sigmoidPrime(activations[-1])*(predicted-actual)

        s=predicted-actual
        error = predicted - y
        s += np.mean(error**2)
        mse+=s

        predictedNum = np.argMax(activations[-1])
        expectedNum = np.argMax(training[-1])

        if predictedNum==expectedNum:
            correct+=1

        i=len(weights)-1
        while i>0:
            newDelta=np.dot(weights[i].T, delta)*sigmoidPrime(activations[i-1])
            weights[i]=weights[i]-learningRate*np.dot(delta,activations[i-1].T)
            biases[i]=biases[i]-learningRate*delta
            delta=newDelta
            i-=1

    mse = s/len(weights)
    accuracy=correct/len(training)

    return weights, biases, mse, accuracy

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
def model(numEpochs, testFile, trainFile):
    testinData=read_file(testFile)
    trainingData=read_file(trainFile)
    # for i in range(0, numEpochs):








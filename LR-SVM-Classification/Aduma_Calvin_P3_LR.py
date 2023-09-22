# 
#First line: m and n, tab separated
#Each line after that has two real numbers representing the
#results of the two tests, followed by a 1.0 if the capacitor
#passed QC anda 0.0 if it failed QC—tab separated.
#

# training set 85 examples
# test set 33 examples

import matplotlib.pyplot as plt 
import numpy as np

def readFile(filename):
    myFile = open(filename, "r")
    line = myFile.readline()
    line = line.split("\t")
    sampleSize = int(line[0])
    numOfFeatures = int(line[1])
    test1Data = []
    test2Data = []
    passORfail = []
    line = myFile.readline()
    while line:
        line = line.split("\t")
        test1Data.append(float(line[0].strip()))
        test2Data.append(float(line[1].strip()))
        passORfail.append(float(line[2].strip()))
        line = myFile.readline()

    myFile.close()

    return test1Data, test2Data, passORfail, sampleSize

def createFeatures(t1, t2):
    capacitors = {}
    for capacitor_num in range(len(t1)):
        tempList = []
        tempList.append(1)
        tempList.append(t1[capacitor_num]) # x1
        tempList.append(t1[capacitor_num]**2) # x2
        tempList.append(t2[capacitor_num]) # x3
        tempList.append(t1[capacitor_num]*t2[capacitor_num]) # x4
        tempList.append(t1[capacitor_num]*(t2[capacitor_num]**2)) # x5
        tempList.append(t2[capacitor_num]**2) # x6
        tempList.append((t1[capacitor_num]**2)*t2[capacitor_num]) # x7
        tempList.append((t1[capacitor_num]**2)*(t2[capacitor_num]**2)) # x8
        capacitors[capacitor_num] = tempList
    return capacitors

def calculateCost(HW, y):
    if y == 1:
        A = np.log(HW)
        return -A
    elif y == 0:
        A = np.log(1-HW)
        return -A

def calculateHW(w, x):
    # calculate wTx
    z = np.dot(w,x)
    u = np.exp(-z) # scalar exp(-z)
    return 1 / (1+u) # scalar

def calculateDerivativeJW(capacitors, w, num_features, QC, a, direction, iteration):
    tempList = []
    # creates list of features in a 1 x NumFeature Array+1
    tempList.append(capacitors[iteration][:num_features])
    x = np.array(tempList)
    x = x.T
    
    # calcualtes HW
    HW = calculateHW(w, x) # scalar

    # calculate alpha part
    B = (HW-QC[iteration])*x
    #B = np.dot((HW-QC[iteration]),x)
    C = np.dot(a,B)

    # calculate new w
    if direction == 0:
        D = np.multiply(C,-1)
        D = D.T
        return w + D, HW # array, scalar
    else:
        return w + C, HW

def gradientDescent(capacitors, QC):
    bestWeight = [float("inf"),float("inf"),0] # holds best weight with best J(w)
    learning_rate = 0.2
    tempIteration = []
    tempJ = []
    iterationVsJ = [] # [[iteration #],[J value]]
    for num_features in range(2,9):
        direction = 0
        initial_w = 2
        weight_array = np.array([[initial_w]*(num_features+1)]) # 1xNumFeature+1 array of weights
        prevJ = float("inf")
        for iteration in range(1001):
            iteration_num = iteration%35
            # w is array
            w, HW = calculateDerivativeJW(capacitors, weight_array, num_features+1, QC, learning_rate, direction, iteration_num)

            # calculates cost
            J = calculateCost(HW, QC[iteration_num])

            # record best weight
            if J < bestWeight[1]:
                bestWeight[0] = w
                bestWeight[1] = J[0][0]
                bestWeight[2] = np.shape(w)[1]-1

            # records iterations vs J for plotting. Only occurs for num_features = 8
            # because after a repetition of tries, it was found that 8 features yielded the lowest J value
            if num_features == 2:
                tempIteration.append(iteration)
                tempJ.append(J[0][0])

            # deterimines direction of gradient
            if J - prevJ < 0:
                prevJ = J
            else:
                prevJ = J
                if direction == 0:
                    direction == 1
                else:
                    direction == 0

    iterationVsJ.append(tempIteration)
    iterationVsJ.append(tempJ)
    return iterationVsJ, bestWeight

def plotIvsJData(IvsJData):
    plt.figure(1)
    plt.plot(IvsJData[0], IvsJData[1], color='orange', marker='o')
    plt.title('Iterations vs J(w) of 8 Features')
    plt.xlabel('Number of Iterations')
    plt.ylabel('J(w)')
    plt.show()

def createDBdata(num_features, x, w):
    y = []
    if num_features == 2:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2)))
    elif num_features == 3:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num))
    elif num_features == 4:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num)+(w[4]*(num**2)))
    elif num_features == 5:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num)+(w[4]*(num**2))+(w[5]*(num**3)))
    elif num_features == 6:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num)+(w[4]*(num**2))+(w[5]*(num**3))+(w[6]*(num**2)))
    elif num_features == 7:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num)+(w[4]*(num**2))+(w[5]*(num**3))+(w[6]*(num**2))+(w[7]*(num**3)))
    elif num_features == 8:
        for num in x:
            y.append(w[0]+(w[1]*num)+(w[2]*(num**2))+(w[3]*num)+(w[4]*(num**2))+(w[5]*(num**3))+(w[6]*(num**2))+(w[7]*(num**3))+(w[8]*(num**4)))

    return y

def decisionBoundary(t1,t2, bestWeight, QC):
    plt.figure(2)
    idx = 0
    plt.xticks(np.arange(-0.75,1.25,0.25))
    plt.yticks(np.arange(-0.75,1.25,0.25))
    for t1data, t2data in zip(t1,t2):
        if QC[idx] == 1:    
            plt.plot(t1data,t2data,'go')
        else:
            plt.plot(t1data,t2data,'rx')
        idx += 1

    x = np.arange(-0.75,1.00,0.25)
    if bestWeight[2] == 2:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 3:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 4:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 5:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 6:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 7:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])
    elif bestWeight[2] == 8:
        y = createDBdata(bestWeight[2], x, bestWeight[0][0])

    plt.plot(x,y,color='blue',lw=1)
    plt.show()

def getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN):
    for capacitor_num in range(len(capacitors)):
        x = capacitors[capacitor_num][:num_features+1]
        x = np.array([x])
        x = x.T
        HW = calculateHW(w,x)
        TP, FP, TN, FN = getCMdata(HW, QC, TP, FP, TN, FN, capacitor_num)
    return TP, FP, TN, FN, HW[0]

def getCMdata(HW, QC, TP, FP, TN, FN, capacitor_num):
    if HW >= 0.5:
        if QC[capacitor_num] == 0:
            return TP, FP + 1, TN, FN
        elif QC[capacitor_num] == 1:
            return TP + 1, FP, TN, FN
    elif HW < 0.5:
        if QC[capacitor_num] == 0:
            return TP, FP, TN + 1, FN
        elif QC[capacitor_num] == 1:
            return TP, FP, TN, FN + 1

def confusionMatrix(QC, capacitors, w):
    TP, FP, TN, FN = 0, 0, 0, 0
    num_features = len(w[0]) - 1
    if num_features == 2:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 3:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 4:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 5:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 6:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 7:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)
    elif num_features == 8:
        TP, FP, TN, FN, HW = getTestHW(capacitors, num_features, QC, w, TP, FP, TN, FN)

    return TN, TP, FN, FP, HW

def metric(TN, TP, FN, FP):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2 * (1 / ((1/precision)+(1/recall)))
    return accuracy, precision, recall, f1


trainingFilename = input("Enter test file name: ")

# reads file and inputs features into the 2 lists: feature1 and feature2
#RETURNS [test 1 data], [test 2 data], [pass or fail QC], size
training1Data, training2Data, trainingQC, trainingSampleSize = readFile(trainingFilename)

# manipulate test1Data and test2Data to create new features
# RETURNS { key=capacitor number : value=[list of features] }
trainingCapacitors = createFeatures(training1Data, training2Data) # {0: [list of features]}

# calculates the gradient descent for the different features
# returns [ [iterations], [J] ], [ weights, J, num of features]
IvsJData, bestWeight = gradientDescent(trainingCapacitors, trainingQC)

# create plot of iteration vs J
plotIvsJData(IvsJData)

# plots decision boundary based on best weights on original graph of data
decisionBoundary(training1Data, training2Data, bestWeight, trainingQC)

# TEST Data
testFilename = input("Enter training file name: ")
test1Data, test2Data, testQC, testSampleSize = readFile(testFilename)
testCapacitors = createFeatures(test1Data, test2Data)
TN, TP, FN, FP, testHW = confusionMatrix(testQC, testCapacitors, bestWeight[0])
testJ = calculateCost(testHW, testQC[0])
accuracy, precision, recall, f1 = metric(TN, TP, FN, FP)
print("Initial Training J: {0}, Final Training J: {1}".format(IvsJData[1][0],bestWeight[1]))
print("J Test: {0}".format(testJ))
print("FP: {0}, FN: {1}, TP: {2}, TN: {3}".format(FP, FN, TP, TN))
print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(accuracy, precision, recall, f1))
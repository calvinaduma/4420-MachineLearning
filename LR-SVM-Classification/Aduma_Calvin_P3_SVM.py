#
#First line: m and n, tab separated
#Each line after that has two real numbers representing the
#results of the two tests, followed by a 1.0 if the capacitor
#passed QC anda 0.0 if it failed QC—tab separated.
#

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
    capacitors = []
    for capacitor_num in range(len(t1)):
        tempList = []
        tempList.append(1) # x0
        tempList.append(t1[capacitor_num]) # x1
        tempList.append(t1[capacitor_num]**2) # x2
        tempList.append(t2[capacitor_num]) # x3
        tempList.append(t1[capacitor_num]*t2[capacitor_num]) # x4
        tempList.append(t1[capacitor_num]*(t2[capacitor_num]**2)) # x5
        tempList.append(t2[capacitor_num]**2) # x6
        tempList.append((t1[capacitor_num]**2)*t2[capacitor_num]) # x7
        tempList.append((t1[capacitor_num]**2)*(t2[capacitor_num]**2)) # x8
        capacitors.append(tempList)
    return capacitors

def plotMarginHyperplane(t1,t2,QC,model):
    # creates graph of data
    idx = 0
    plt.figure()
    plt.xticks(np.arange(-0.75,1.01,0.25))
    plt.yticks(np.arange(-0.75,1.01,0.25))
    for t1data, t2data in zip(t1,t2):
        if QC[idx] == 1:    
            plt.plot(t1data,t2data,'go')
        else:
            plt.plot(t1data,t2data,'rx')
        idx += 1

    # plots points onto chart
    idx = 0
    for t1data, t2data in zip(t1,t2):
        if QC[idx] == 1:    
            plt.plot(t1data,t2data,'go')
        else:
            plt.plot(t1data,t2data,'rx')
        idx += 1

    # creates hyperplane
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.arange(-0.75,1.01,0.25)
    yy = a * xx + 0.065

    # plots the parallels to the separating hyperplane that pass through the
    # support vectors
    b = model.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = model.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the hyperplane, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'b-')
    plt.plot(xx, yy_down, 'b--')
    plt.plot(xx, yy_up, 'b--')

    plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=80, facecolors='none')
    plt.show()
    return w

def trainAlgorithm(capacitor,QC):
    svc = SVC(kernel='linear')
    model = svc.fit(capacitor,QC)
    return model

def makePredictions(capacitor,model):
    predicted_list = model.predict(capacitor)
    return predicted_list


#trainingFilename = input("Enter test file name: ")
# reads file and inputs features into the 2 lists: feature1 and feature2
#RETURNS [test 1 data], [test 2 data], [pass or fail QC], size
training1Data, training2Data, trainingQC, trainingSampleSize = readFile("Aduma_Calvin_P3train.txt")

trainingCapacitors = createFeatures(training1Data, training2Data) # {0: [list of features]}

model = trainAlgorithm(trainingCapacitors, trainingQC)

weight = plotMarginHyperplane(training1Data,training2Data,trainingQC, model)

# TEST data
#testFilename = input("Enter training file name: ")
test1Data, test2Data, testQC, testSampleSize = readFile("Aduma_Calvin_P3test.txt")
testCapacitors = createFeatures(test1Data, test2Data)
y_pred = makePredictions(testCapacitors,model)

# Predictions
print(confusion_matrix(testQC,y_pred))
print(classification_report(testQC,y_pred))
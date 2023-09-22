# first line: m lines of data and n number of features

import matplotlib.pyplot as plt 
import numpy as np
import random

def readFile(filename):
    myFile = open(filename, "r")
    line = myFile.readline()
    line = line.split("\t")
    numOfLines = int(line[0])
    numOfFeatures = int(line[1])
    timeList = []
    yearList = []
    line = myFile.readline()
    while line:
        line = line.split("\t")
        yearList.append([int(line[0].strip())])
        timeList.append(float(line[1].strip()))
        line = myFile.readline()
    myFile.close()

    combinedList = list((zip(yearList,timeList))) 
    random.shuffle(combinedList)
    yearList, timeList = zip(*combinedList)

    return timeList, yearList

def createFolds(timeList,yearList,folds,power):
    tempTime = []
    tempYear = []
    finalFolds = []
    for i in range(len(timeList)):
        tempTime.append(timeList[i]) # [time]
        tempYear.append(yearList[i][power]) # [linear_year]
        if (i+1)%folds == 0 or (i+1) >= len(timeList):
            finalFolds.append([tempTime,tempYear])
            tempTime = []
            tempYear = []
    return finalFolds # [ [[time_1], [year_1]],[[time_2], [year_2]],...,[[time_n], [year_n]] ] of years

def linearModel(timeList, yearList, folds, J_data, table):
    fold_data = createFolds(timeList, yearList, folds, 0)
    # calculates J
    for i in range(folds-1): # i = test
        # i=folds; test_data = race_data[i+i*(folds-1):(i*folds)+(folds-1)]
        x_test_data = fold_data[i][1] # years
        y_test_data = fold_data[i][0] # time
        x_training_data = []
        y_training_data = []
        for j in range(folds-1):
            if j == i:
                continue
            x_training_data += fold_data[j][1]
            y_training_data += fold_data[j][0]

        ones_training = np.ones(len(x_training_data), dtype=int)
        ones_test = np.ones(len(x_test_data), dtype=int)
        x_training = np.array([ones_training,np.array(x_training_data)]) # years- training
        x_test = np.array([ones_test,np.array(x_test_data)]) # years- test
        x_training = x_training.T
        x_test = x_test.T
        y_training = np.array([y_training_data])
        y_training = y_training.T
        y_test = np.array([y_test_data])
        y_test = y_test.T
        A_test = np.linalg.pinv(np.dot(x_test.T, x_test))
        A_training = np.linalg.pinv(np.dot(x_training.T, x_training))
        B_test = np.dot(x_test.T, y_test)
        B_training = np.dot(x_training.T, y_training)
        w_test = np.dot(A_test, B_test)
        w_training = np.dot(A_training, B_training)
        C_test = np.dot(x_test,w_test) - y_test
        C_training = np.dot(x_training,w_training) - y_training
        J_test = (1/len(x_test_data))*np.dot(C_test.T,C_test)
        J_training = (1/len(x_training_data))*np.dot(C_training.T,C_training)
        J_data[i] = [[J_training[0][0],J_test[0][0]]]
        table_data[(i*2)+1,1] = J_training[0,0]
        table_data[(i*2)+2,1] = J_test[0,0]       

def quadraticModel(timeList, yearList, folds, J_data, table):
    fold_data = createFolds(timeList, yearList, folds,1) 
    for i in range(folds-1):
        # i=folds; test_data = race_data[i+i*(folds-1):(i*folds)+(folds-1)]
        x_test_data = fold_data[i][1] # years
        y_test_data = fold_data[i][0] # time
        x_training_data = []
        y_training_data = []
        for j in range(folds-1):
            if j == i:
                continue
            x_training_data += fold_data[j][1]
            y_training_data += fold_data[j][0]

        ones_training = np.ones(len(x_training_data), dtype=int)
        ones_test = np.ones(len(x_test_data), dtype=int)
        x_training = np.array([ones_training,np.array(x_training_data)]) # years- training
        x_test = np.array([ones_test,np.array(x_test_data)]) # years- test
        x_training = x_training.T
        x_test = x_test.T
        y_training = np.array([y_training_data])
        y_training = y_training.T
        y_test = np.array([y_test_data])
        y_test = y_test.T
        A_test = np.linalg.pinv(np.dot(x_test.T, x_test))
        A_training = np.linalg.pinv(np.dot(x_training.T, x_training))
        B_test = np.dot(x_test.T, y_test)
        B_training = np.dot(x_training.T, y_training)
        w_test = np.dot(A_test, B_test)
        w_training = np.dot(A_training, B_training)
        C_test = np.dot(x_test,w_test)-y_test
        C_training = np.dot(x_training,w_training)-y_training
        J_test = (1/len(x_test_data))*np.dot(C_test.T,C_test)
        J_training = (1/len(x_training_data))*np.dot(C_training.T,C_training)
        tmpList = J_data[i]
        tmpList.append([J_training[0][0],J_test[0][0]]) #[0][0] because data stored in tuple of list
        J_data[i] = tmpList
        table_data[(i*2)+1,2] = J_training[0,0]
        table_data[(i*2)+2,2] = J_test[0,0]  

def cubicModel(timeList, yearList, folds, J_data, table):
    fold_data = createFolds(timeList, yearList, folds, 2) # [ [[time_1],[year_1]],[[time_2],[year_2]],...,[[time_n],[year_n]] ] of years
    # calculates J
    for i in range(folds-1):
        # i=folds; test_data = race_data[i+i*(folds-1):(i*folds)+(folds-1)]
        x_test_data = fold_data[i][1] # years
        y_test_data = fold_data[i][0] # time
        x_training_data = []
        y_training_data = []
        for j in range(folds-1):
            if j == i:
                continue
            x_training_data += fold_data[j][1]
            y_training_data += fold_data[j][0]

        ones_training = np.ones(len(x_training_data), dtype=int)
        ones_test = np.ones(len(x_test_data), dtype=int)
        x_training = np.array([ones_training,np.array(x_training_data)]) # years- training
        x_test = np.array([ones_test,np.array(x_test_data)]) # years- test
        x_training = x_training.T
        x_test = x_test.T
        y_training = np.array([y_training_data])
        y_training = y_training.T
        y_test = np.array([y_test_data])
        y_test = y_test.T
        A_test = np.linalg.pinv(np.dot(x_test.T, x_test))
        A_training = np.linalg.pinv(np.dot(x_training.T, x_training))
        B_test = np.dot(x_test.T, y_test)
        B_training = np.dot(x_test.T, y_test)
        w_test = np.dot(A_test, B_test)
        w_training = np.dot(A_training, B_training)
        C_test = np.dot(x_test,w_test)-y_test
        C_training = np.dot(x_training,w_training)-y_training
        J_test = (1/len(x_test_data))*np.dot(C_test.T,C_test)
        J_training = (1/len(x_training_data))*np.dot(C_training.T,C_training)
        tmpList = J_data[i]
        tmpList.append([J_training[0][0],J_test[0][0]]) #[0][0] because data stored in tuple of list
        J_data[i] = tmpList
        table_data[(i*2)+1,3] = J_training[0,0]
        table_data[(i*2)+2,3] = J_test[0,0]  

def getMean(J_data,folds,m,table_data):
    J_linear_means = [] # [train, test]
    J_quadratic_means = []
    J_cubic_means = []
    training_mean_linear = 0
    training_mean_quadratic = 0
    training_mean_cubic = 0
    test_mean_linear = 0
    test_mean_quadratic = 0
    test_mean_cubic = 0
    for i in range(folds-1):
        training_mean_linear += J_data[i][0][0]
        training_mean_quadratic += J_data[i][1][0]
        training_mean_cubic += J_data[i][2][0]
        test_mean_linear += J_data[i][0][1]
        test_mean_quadratic += J_data[i][1][1]
        test_mean_cubic += J_data[i][2][1]
           
    J_linear_means = [training_mean_linear/(m-folds),test_mean_linear/folds]
    J_quadratic_means = [training_mean_quadratic/(m-folds),test_mean_quadratic/folds]
    J_cubic_means = [training_mean_cubic/(m-folds),test_mean_cubic/folds]
    table_data[9,1] = J_linear_means[0]
    table_data[10,1] = J_linear_means[1]
    table_data[9,2] = J_quadratic_means[0]
    table_data[10,2] = J_quadratic_means[1]
    table_data[9,3] = J_cubic_means[0]
    table_data[10,3] = J_cubic_means[1]
    return J_linear_means, J_quadratic_means, J_cubic_means

# [train,test]
def plotMeans(linear, quadratic, cubic):
    training_data = [linear[0],quadratic[0],cubic[0]]
    test_data = [linear[1],quadratic[1],cubic[1]]
    plt.scatter([1,2,3], training_data, color = 'blue', marker = '+', label = 'Training J')
    plt.scatter([1,2,3], test_data, color = 'orange', marker = '+', label = 'Testing J')
    xAxis = [0,1,2,3,4]
    xRange = range(len(xAxis))
    plt.xticks(xRange, xAxis)
    plt.ylim(0,30)
    plt.title('Olympic Record Times J Values for Polynomial Degrees')
    plt.legend()
    plt.xlabel('Highest Polynomial Degree')
    plt.ylabel('Squared Error Cost Function (J)')
    plt.plot([1,2,3],training_data,color='blue',lw=1)
    plt.plot([1,2,3],test_data,color='orange',lw=1)
    plt.show()

def bestPolynomial(linear,quadratic,cubic,year,timeList,yearList):
    linearDiff = abs(linear[0]-linear[1])
    quadraticDiff = abs(quadratic[0]-quadratic[1])
    cubicDiff = abs(cubic[0]-cubic[1])
    if min(linearDiff,quadraticDiff,cubicDiff) == linearDiff:
        weights,predicted_time = bestPolynomialModel(1,timeList,yearList,year)
    elif min(linearDiff,quadraticDiff,cubicDiff) == quadraticDiff:
        weights,predicted_time = bestPolynomialModel(2,timeList,yearList,year)
    elif min(linearDiff,quadraticDiff,cubicDiff) == cubicDiff:
        weights,predicted_time = bestPolynomialModel(3,timeList,yearList,year)
    return weights,predicted_time

def bestPolynomialModel(num,timeList,yearList,year):
    if num == 1:
        weight = weight_solver(timeList,yearList,num)
        predicted_time = weight[0][0]+(weight[1][0]*year)
    elif num == 2:
        weight = weight_solver(timeList,yearList,num)
        predicted_time = weight[0][0]+(weight[1][0]*year)+(weight[2][0]*(year**num))
    elif num == 3:
        weight = weight_solver(timeList,yearList,num)
        predicted_time = weight[0][0]+(weight[1][0]*year)+(weight[2][0]*(year**(num-1)))+(weight[3][0]*(year**num))
    return weight, predicted_time

def weight_solver(timeList,yearList,num):
    x_data = []
    tmp1 = []
    for years in yearList:
        tmp1.append(years[0])
    x_data.append(tmp1)
    if num == 2:
        tmp1 = []
        for years in yearList:
            tmp1.append(years[num-1])
        x_data.append(tmp1)
    elif num == 3:
        tmp1 = []
        tmp2 = []
        for years in yearList:
            tmp1.append(years[num-2])
            tmp2.append(years[num-1])
        x_data.append(tmp1)
        x_data.append(tmp2)
    
    ones_x = [[1]*len(x_data[0])]
    for i in range(len(x_data)):
        ones_x.append(x_data[i])
    x = np.array(ones_x)
    y = np.array([timeList])
    x = x.T
    y = y.T
    A = np.linalg.pinv(np.dot(x.T,x))
    B = np.dot(x.T,y)
    w = np.dot(A,B)
    return w

#global variables
J_data = {} # [model_number: [[J_training_linear,J_test_linear],...,[J_training_cubic,J_test_cubic]]
table_data = []
filename = 'W100MTimes.txt'   
number_of_folds = 5
columnHeader = ['                 ','      Linear     ','    Quadratic    ','       Cubic     ']
training_mean = ['Mean for Training','          ','          ', '          ']
testing_mean = [' Mean for Testing ','          ','          ', '          ']
table_data.append(columnHeader)
table_data.append(['             234 ','          ','          ', '          '])
table_data.append(['               1 ','          ','          ', '          '])
table_data.append(['             134 ','          ','          ', '          '])
table_data.append(['               2 ','          ','          ', '          '])
table_data.append(['             124 ','          ','          ', '          '])
table_data.append(['               3 ','          ','          ', '          '])
table_data.append(['             123 ','          ','          ', '          '])
table_data.append(['               4 ','          ','          ', '          '])
table_data.append(training_mean)
table_data.append(testing_mean)
table_data = np.array(table_data)

year = int(input("Enter the year: "))
year = year - 1900

try:
    timeList, yearList = readFile(filename) # [time] [year]
except:
    print("Failed to read file")

#squares and cubes x values
for currentYear in yearList:
    squared = currentYear[0] ** 2
    cubed = currentYear[0] ** 3
    currentYear.append(squared)
    currentYear.append(cubed)

# places data into J_data as [model_number/fold_number: [[J_training_linear,J_test_linear],...,[J_training_cubic,J_test_cubic]]
# prints table
linearModel(timeList, yearList, number_of_folds, J_data, table_data)
quadraticModel(timeList, yearList, number_of_folds, J_data, table_data) 
cubicModel(timeList, yearList, number_of_folds, J_data, table_data) 

# gets mean of data as [train,test] and prints table of J values
linear_mean, quadratic_mean, cubic_mean = getMean(J_data,number_of_folds, len(yearList), table_data)
print(table_data)

# plots means
plotMeans(linear_mean, quadratic_mean, cubic_mean)

#Time Prediction and gets weights
weights,predicted_time = bestPolynomial(linear_mean, quadratic_mean, cubic_mean,year,timeList,yearList)
print("Weights: ", weights)
print("For the year {0}, the predicted time is: {1}".format(year+1900,predicted_time))

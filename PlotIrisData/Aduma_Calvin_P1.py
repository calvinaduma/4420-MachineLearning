import matplotlib.pyplot as plt 
import numpy as np

def readFile(filename, xAxis, yAxis):
    myFile = open(filename, "r")
    line = myFile.readline()
    line = line.split("\t")
    numOfLines = int(line[0])
    numOfFeatures = int(line[1])
    maxX, maxY = 0, 0
    minX, minY = float("inf"), float("inf")
    data = {}
    line = myFile.readline()
    while line:
        line = line.split("\t")
        # [ plant name, [feature 1], [feature 2] ] 
        plantName = line[numOfFeatures].strip()
        xAxisVal = float(line[xAxis])
        yAxisVal = float(line[yAxis])

        if plantName not in data:
            listArray = [[xAxisVal], [yAxisVal]]
            data[plantName] = listArray
        else:
            foundList = data[plantName]
            foundList[0].append(xAxisVal)
            foundList[1].append(yAxisVal)
            data[plantName] = foundList

        if xAxisVal > maxX:
            maxX = xAxisVal
        if xAxisVal < minX:
            minX = xAxisVal
        if yAxisVal > maxY:
            maxY = yAxisVal
        if yAxisVal < minY:
            minY = yAxisVal
        line = myFile.readline()

    myFile.close()
    return data, maxX, minX, maxY, minY


featureCodes = ["Sepal Length",'Sepal Width','Petal Length','Petal Width']
redo = "y"
filename = input("Enter the name of your data file: ")
print("\nYou can do a plot of any two features of the Iris Data set")
while redo == "y" or redo == "Y":
    print("Feature Codes:\n\t0 = sepal length\n\t1 = sepal width\n\t2 = petal length\n\t3 = petal width\n")
    xAxis = int(input("Enter the feature code for the horizontal axis: "))
    while xAxis < 0 or xAxis > 3:
        xAxis = int(input("Enter a number between 0 and 3 for feature code for the horizontal axis: "))
    yAxis = int(input("Enter the feature code for the vertical axis: "))
    while yAxis < 0 or yAxis > 3 or yAxis == xAxis:
        yAxis = int(input("Enter a number between 0 and 3 that is not {0}, for feature code for the vertical axis: ".format(xAxis)))
    print()

    # PLOT
    try:
        dataDictionary, maxX, minX, maxY, minY = readFile(filename, xAxis, yAxis) #reads data and stores in list of list
        for plant in dataDictionary:
            plantList = dataDictionary[plant]
            if plant == "setosa":
                plt.scatter(plantList[0], plantList[1], color = 'red', marker = '.', label = plant)
            elif plant == "versicolor":
                plt.scatter(plantList[0], plantList[1], color = 'blue', marker = '*', label = plant)
            elif plant == "virginica":
                plt.scatter(plantList[0], plantList[1], color = 'green', marker = '+', label = plant)

        xInterval = np.arange(minX, maxX, 0.5)
        yInterval = np.arange(minY, maxY, 0.5)
        plt.legend()
        plt.title('Iris Flower Plot: {0} vs {1}'.format(featureCodes[xAxis],featureCodes[yAxis]))
        plt.xlabel(featureCodes[xAxis])
        plt.ylabel(featureCodes[yAxis])  
        plt.show()
    except:
        print ("failed to read file")

    redo = input("Would you like to do another plot? (y/n): ")
    print()


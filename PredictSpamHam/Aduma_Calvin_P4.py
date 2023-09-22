# 1=SPAM 0=HAM
# Output:
    # number of Spam and Ham emails are in Test set
    # number of FP, TP, FN, TN that spam filter predicted
    # accuracy, preision, recall, F1 of spam filter

# Turn-in:
    # intro of model
    # Confusion matrix
    # accuracy, precision, recall, F1
    # screenshot of python console
    # copy of code

import math

def readTrainFile(trainFilename,stopFilename):
    spam = 0
    ham = 0
    spamhamDictionary = {}
    trainFile = open(trainFilename,"r")
    stopFile = open(stopFilename,"r")
    stopWords = []
    line = stopFile.readline()
    # creates list for stop words
    while line:
        stopWordsFunc(line,stopWords)
        try:
            line.readline()
        except:
            break

    line = trainFile.readline()
    # creates dictionary of tally of words in spam/ham
    while line:
        spam, ham = spamORham(line,spam,ham,spamhamDictionary,stopWords)
        try:
            line = trainFile.readline()
        except:
            break
    # creates percentages dictionary of wirds in spam/ham
    createPseudoCount(1,spamhamDictionary,spam,ham)
    return spam, ham, spamhamDictionary, stopWords

def stopWordsFunc(line,stopWords):
    line.lower()
    line.strip()
    for word in line:
        stopWords.append(word)

def spamORham(line,spam,ham,spamhamDictionary,stopWords):
    spamham = int(line[:1])
    if spamham == 1:
        spam += 1
    elif spamham ==0:
        ham += 1
    else:
        return # edge case where 0,1 is not first input
    line = cleanLine(line)
    for word in line:
        if word in spamhamDictionary and word not in stopWords:
            if spamham == 1:
                spamhamDictionary[word][1] += 1
            else:
                spamhamDictionary[word][0] += 1
        elif word not in spamhamDictionary and word not in stopWords:
            if spamham == 1:
                spamhamDictionary[word] = [0,1]
            else:
                spamhamDictionary[word] = [1,0]
    return spam, ham

def cleanLine(line):
    line = line[1:].lower()
    line.strip()
    for letters in line:
        if letters in "[]!.?,~""-@;:$#%^&*()+-=_}{/><|'":
            line = line.replace(letters, " ")
    line = line.split()
    line = set(line)
    return line

def createPseudoCount(k,spamhamDictionary,spam,ham):
    for word in spamhamDictionary.keys():
        spamhamDictionary[word][0] = (spamhamDictionary[word][0]+k)/(2*k+ham)
        spamhamDictionary[word][1] = (spamhamDictionary[word][1]+k)/(2*k+spam)

def P_sl_S_func(line,spamhamDictionary):
    probabilityList = []
    for word,percentList in spamhamDictionary.items():
        if word in line:
            probabilityList.append(percentList[1])
        else:
            probabilityList.append(1-percentList[1])
    probability = 0
    for num in probabilityList:
        probability += math.log(num)
    probability = math.exp(probability)
    return probability

def P_sl_nS_func(line,spamhamDictionary):
    probabilityList = []
    for word,percentList in spamhamDictionary.items():
        if word in line:
            probabilityList.append(percentList[0])
        else:
            probabilityList.append(1-percentList[0])
    probability = 0
    for num in probabilityList:
        probability += math.log(num)
    probability = math.exp(probability)
    return probability

def P_sl_S_nS(line,spamhamDictionary,pSpamHam):
    P_sl_S = P_sl_S_func(line,spamhamDictionary)
    P_sl_nS = P_sl_nS_func(line,spamhamDictionary)
    normalization_constant = (P_sl_S*pSpamHam[1]) + (P_sl_nS*pSpamHam[0])
    #normalization_constant = 1/ (1+normalization_constant)
    return normalization_constant, P_sl_S

def P_S_sl(line,spamhamDictionary,pSpamHam,rawDataList):
    normalization_constant, P_sl_S = P_sl_S_nS(line,spamhamDictionary,pSpamHam)
    prior = pSpamHam[1]
    likelihood = P_sl_S
    rawDataList.append((likelihood * prior)/normalization_constant)
    
def calculateProbability(line,spamhamDictionary,stopWords,rawData,pSpamHam):
    spamham = int(line[:1])
    rawDataList = [] # [spam/ham, percentage]
    if spamham == 1:
        rawDataList.append(1)
    elif spamham == 0:
        rawDataList.append(0)
    else:
        return
    
    # cleans line
    line = cleanLine(line)
    line = list(line)
    # removes stop words
    for word in line:
        if word in stopWords:
            line.remove(word)
    #calculates probability
    P_S_sl(line,spamhamDictionary,pSpamHam,rawDataList)
    rawData.append(rawDataList)

def readTestFile(testFilename,spamhamDictionary,stopWords,pSpamHam):
    testFile = open(testFilename,"r")
    line = testFile.readline()
    rawData = [] # list of lists
    # creates dictionary of tally of words in spam/ham
    while line:
        cleanLine(line)
        calculateProbability(line,spamhamDictionary,stopWords,rawData,pSpamHam)
        try:
            line = testFile.readline()
        except:
            break
    return rawData

def classificationTest(rawData):
    TN,FN,TP,FP = 0,0,0,0
    for subjectLine in rawData:
        if subjectLine[0] == 0: # actual ham
            if subjectLine[1] >= 0.5: # predict spam
                FN += 1
            else: # predict ham
                TN += 1
        else: # actual spam
            if subjectLine[1] >= 0.5: # predict spam
                TP += 1
            else: # predict ham
                FP += 1
    return TP,FP,TN,FN

def metrics(TN,FP,FN,TP):
    accuracy,precision,recall,F1 = 0,0,0,0
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(1/( (1/precision)+(1/recall) ))
    return accuracy, precision, recall, F1

trainFile = input("Enter train file name: ")
stopFile = input("Enter stop words file name: ")
# returns: # spam/ham emails, dictionary of spam/ham words

spam, ham, spamhamDictionary, stopWords= readTrainFile(trainFile,stopFile)
# spamhamDictionary = { word : [% num ham, % num spam] }

# [P(-S) P(S)]
pSpamHam = [ham/(spam+ham),spam/(spam+ham)]

testFile = input("Enter test file name: ")
rawData = readTestFile(testFile,spamhamDictionary,stopWords,pSpamHam)
# rawData = [ [spam/ham, probability subject line is spam] ]

TP,FP,TN,FN = classificationTest(rawData)
accuracy, precision, recall ,F1 = metrics(TN,FP,FN,TP)
print("Spam: {0}, Ham: {1}".format(spam,ham))
print("False Positive: {0}, True Positive: {1}".format(FP,TP))
print("False Negative: {0}, True Negative: {1}".format(FN,TN))
print("Accuracy: {0}".format(accuracy))
print("Precision: {0}".format(precision))
print("Recall: {0}".format(recall))
print("F1: {0}".format(F1))
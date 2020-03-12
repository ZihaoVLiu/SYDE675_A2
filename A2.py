import matplotlib.pyplot as plt
from time import time
import random
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

#open the data using read_csv of pandas
game = pd.read_csv('tic-tac-toe.data', header = None)
wine = pd.read_csv('wine.data', header= None)

#modify the first column of wine dataset to the last
order = [1,2,3,4,5,6,7,8,9,10,11,12,13,0]
wine = wine[order]

#modify the header name
gameHeaders = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'Class']
wineHeaders = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline', 'Class']

#change the data structure from dataframe into list
gameList = []
for item in range(game.shape[0]):
    gameList.append(list(game.iloc[item,:]))

wineList = []
for item in range(wine.shape[0]):
    wineList.append(list(wine.iloc[item,:]))
'''
#randmoly shuffle rows and keep the index order
#this operation disruptes data consistency, which could enhance the classification accuracy
gameR = game.sample(frac=1).reset_index(drop=True)
ga = pd.read_csv('wocao.data')
ga = ga.iloc[:, 1:]'''


def getEnt(data):
    '''Calculate the  of given dataset
    input parameter is the original dataset
    return value is the entropy vlaue'''
    numSample = len(data) # Total number of input dataset
    classList = [classLabel[-1] for classLabel in data]  # store all the class labels in a list
    classes = set(classList)  # get the name of class labels
    infoSat = [classList.count(className) for className in classes]
    p = [item/numSample for item in infoSat] # the probability of each class
    return sum([-item * np.log2(item) for item in p]) # get the entropy of input dataset

def getIG(data):
    '''Calculate the information gain value of each attribute
    input parameter is the original dataset
    return value is the information gain of each attribute'''
    IG = list() # initialed a list for information gain of each attribute
    originalEnt = getEnt(data) # get the initial entropy
    numAttribute = len(data[0]) - 1 # get the total number of attribute
    for i in range(numAttribute): # iterate each attribute
        attributValues = [sample[i] for sample in data]
        valueName = set(attributValues)  # get all unique value of the attribute
        attributEnt = 0
        for j in valueName: # iterate each value in the attribute
            itemData = [data[index] for index in range(len(attributValues)) if attributValues[index] == j]
            # get dataset of indicated attribute value ('j' in this iteration)
            itemEnt = getEnt(itemData) # get the entropy of the specific attribute value
            attributEnt += (len(itemData) / len(data)) * itemEnt
        #print('The %d attribute entropy is %f.' % (i, attributEnt))
        #print('The %d attribute Information Gain is %f.' % (i, originalEnt - attributEnt))
        IG.append(originalEnt - attributEnt) # calculate the information gain
    return IG

def getGR(data):
    '''
    calculate the information gain ratio value of each attribute
    :param data: the original dataset
    :return: the gain ratio of each attribute
    '''
    SI = []  # a list of Intrinsic Information of the Attribute
    IG = list() # initialed a list for information gain of each attribute
    originalEnt = getEnt(data) # get the initial entropy
    numAttribute = len(data[0]) - 1 # get the total number of attribute
    numSample = len(data)  # Total number of input dataset
    for i in range(numAttribute): # iterate each attribute
        attributValues = [sample[i] for sample in data]
        valueName = set(attributValues)  # get all unique value of the attribute
        attributEnt = 0
        infoSat = [attributValues.count(value) for value in valueName]  # statistic information of given class
        p = [item / numSample for item in infoSat]  # the probability of each class
        splitInformation = sum([-item * np.log2(item) for item in p])   # get the entropy of Intrinsic Information of the Attribute
        SI.append(splitInformation)
        for j in valueName: # iterate each value in the attribute
            itemData = [data[index] for index in range(len(attributValues)) if attributValues[index] == j]
            # get dataset of indicated attribute value ('j' in this iteration)
            itemEnt = getEnt(itemData) # get the entropy of the specific attribute value
            attributEnt += (len(itemData) / len(data)) * itemEnt
        #print('The %d attribute entropy is %f.' % (i, attributEnt))
        #print('The %d attribute Information Gain is %f.' % (i, originalEnt - attributEnt))
        IG.append(originalEnt - attributEnt) # calculate the information gain

        '''the following if statement in lambda function is very important, 
        because the Intrinsic Information of the Attribute (denominator) may be 0 in some cases, which is invalid
        and when the Intrinsic Information of the Attribute equals to 0, the information gain will also be 0
        in this case, the gain ratio is assigned a value 0, or can be seen as information gain value
        Because the gain ratio always applied by continuous dataset(C4.5 method), not discrete dataset'''
        GR = list(map(lambda x, y: 0 if x == y and x == 0 else x / y, IG, SI))  # calculate the information gain ratio
    return GR

def getMax(valueslist):
    '''Output the attribute index with the highest information gain
    input parameter is a information gain list (get from getIG() or getGR() function)
    return the index number'''
    return valueslist.index(max(valueslist))

def discardAttri(data, index, attributeValue):
    '''discard indicated attribute but keep attribute value of given column
    data: original dataset
    index: the value returned from getMax. Indicated attribute will be discarded
    attributeValue: indicated attribute value'''
    newDataSet = []
    for sample in data:  # iterate each sample in dataset
        if sample[index] == attributeValue:  # only select sample equals to the given attributeValue
            newDataSet.append(sample[:index] + sample[(index + 1):])  # add the rest values into the sample
    return newDataSet

def buildTree(data, labels, function):
    '''build a decision tree recursively based on ID3 algorithm
        input parameter is the original dataset
        labels provides the attribute label name to this function
        function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
        return a dictionary type tree'''
    newLables = copy.copy(labels)
    classList = [sample[-1] for sample in data]  # store all the class labels in a list
    # define the recursive termination condition
    # all labels are positive or negative
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # only one attribute is left.
    if len(data[0]) == 1:
        classNames = list(set(classList))
        classNameCount = [classList.count(className) for className in classNames]
        return classNames[classNameCount.index(max(classNameCount))]
    IGorGRList = function(data)  # get the information gain or gain ratio list
    maxIGorGRIndex = getMax(IGorGRList)  # get index with the highest IG
    bestAttributeName = newLables[maxIGorGRIndex]  # pair the index with corresponding attribute name
    myTree = {bestAttributeName: {}}  # store the decision tree into a dictionary
    # (this is root node before starting recursive)
    newLables.remove(bestAttributeName)  # remove the attribute just utilized
    bestFeature = [sample[maxIGorGRIndex] for sample in data]
    values = set(bestFeature)  # get the unique attribute value of corresponding attribute
    for value in values:  # build branch for each attribute value
        subLabels = newLables[:]
        myTree[bestAttributeName][value] = buildTree(discardAttri(data, maxIGorGRIndex, value), subLabels, function)
    return myTree

#tree = buildTree(game)

def saveTree(name, tree):
    '''Save the tree under the working direction path
    name: a string given by user
    tree: return from buildTree() function'''
    np.save(name, tree)

def readTree(path):
    '''read the tree from the path
    path: the working dictionary of target tree
    return a tree structure (dictionary)'''
    return np.load(path, allow_pickle=True).item()

def typeJudge(branches, twig):
    '''judge whether the type of input is a dictionary
    branches is a dictionary of tree structure
    key is one of the branch
    return a bool value'''
    branchesType = type(branches[twig]) # get the type of branch
    if branchesType == dict:
        return True
    else:
        return False

def predictResult(tree, labels, sample, train):
    '''use decision tree to predict the class label of the input sample
        tree: decision tree build by buildTree() function
        labels: to get the header of data (using .columns to get the list)
        sample: a list, the input test sample
        return the prediction value'''
    classNameList = [sample[-1] for sample in train]
    classNames = list(set(classNameList))
    classNameCount = [classNameList.count(className) for className in classNames]
    maxClassLabel = classNames[classNameCount.index(max(classNameCount))]  # class label with highest value count

    node = list(tree.keys())  # get the upper tree node of each recursion in a list
    nodeName = node[0]  # get the attribute name of node
    branches = tree[nodeName]  # get branches under node
    labelIndex = labels.index(nodeName)  # get nodeName index of labels
    sampleValue = sample[labelIndex]  # get the attribute value of corresponding sample based on the labelIndex
    '''This part is very important, if without the following if statement, and sampleValue 
                    did not exist under this branch, a "False" value will be returned.'''
    if sampleValue in branches:
        twig = branches[sampleValue]  # the branches under parent branches ('twig' is used here)
    else:
        return maxClassLabel

    if type(twig) == dict:
        return predictResult(twig, labels, sample, train)  # continue recursion but use twig branch
    else:
        return twig

def predictResult1(tree, labels, sample, train):
    '''use decision tree to predict the class label of the input sample
    tree: decision tree build by buildTree() function
    test: to get the header of data (using .columns to get the list)
    sample: a list, the input test sample
    train: the training dataset providing the class label with highest value count
    return the prediction value'''
    if type(tree) == str: # in case the input tree only has a root node
        return tree

    classNameList = [sample[-1] for sample in train]
    classNames = list(set(classNameList))
    classNameCount = [classNameList.count(className) for className in classNames]
    maxClassLabel = classNames[classNameCount.index(max(classNameCount))]  # class label with highest value count

    upperNode = list(tree.keys())[0] # get the upper tree node of each recursion
    upperNodeIndex = labels.index(upperNode) # get the index number of corresponding tree node
    branches = tree[upperNode] # get the branches under the upperNode
    for twig in branches.keys(): # iterate each twig under the upperNode key (among branches)
        if sample[upperNodeIndex] == twig: # detect if the corresponding node name equals to twig
            if typeJudge(branches, twig):

                '''This part is very important, if without the following if statement, and the branches[twig] 
                did not exist under this branch, a "None" value will be returned.'''
                if predictResult(branches[twig], labels, sample, train) is None:
                    return maxClassLabel
                else:
                    return predictResult(branches[twig], labels, sample, train) # continue recursion
                    # but use twig branch

            else:
                return branches[twig] # return the class result
        else:
            continue

def batchPredict(train, test, labels, function):
    '''output the test set with a column of prediction values
    train: training dataset
    test: testing dataset
    labels is the dataset headers
    function: the function name of decision-tree learning algorithms (one of getIG or getGR)
    :return a list with prediction results'''
    testSampleNumber = len(test)  # get the sample number of test set
    tree = buildTree(train, labels, function) # construct the tree
    result = [] # initial a list to store prediction value
    for sample in test:  # iterate each sample in test dataset
        result.append(predictResult(tree, labels, sample, train))
    if testSampleNumber != len(result):
        print('Error: The number of test data is wrong.')
    return result

def plotConfusionMatrix(test, predictResult):
    '''output the confusion matrix and draw heatmap
    test: dataset used to construct a classifier
    predictResult: result from batchPredict() function
    print the confusion matrix and draw a confusion matrix heatmap'''
    tResult = [label[-1] for label in test]
    pResult = predictResult
    c = confusion_matrix(tResult, pResult)
    #sns.heatmap(c, annot=True, ax=ax)
    #ax.set_title('Confusion Matrix')
    #ax.set_xlabel('predict')
    #ax.set_ylabel('true ground')
    #plt.show()
    return c

def calAccuracy(test, predictResult):
    '''calculate the accuracy of test prediction dataset
    test: dataset used to construct a classifier
    predictResult: result from batchPredict() function
    return a accuracy value'''
    if len(test) != len(predictResult):
        print("Error: The lengths between two input parameters are different")
        return 0
    tResult = [label[-1] for label in test]  # store the test class result into a list
    pResult = predictResult
    totalTrue = len([1 for index in range(len(tResult)) if tResult[index] == pResult[index]])
    # calculate the same total number between ground truth(test result) and prediction(predict result)
    total = len(predictResult)
    accuracy = totalTrue / total
    return accuracy

def randomShuffle(data):
    '''randomly shuffle the input dataset
    the input parameter is the original dataset
    return a randomly shuffle list'''
    return random.shuffle(data)

def getThreshold(data):
    '''function get the threshold of each attribute based on information gain approach
    input parameter is the original dataset
    return dataset is a discrete dataset'''

    def bubble_sort(list):
        '''
        used in the following list sort procedure
        :param list:  list with inner list, and the first element of inner list is a float number
        :return: an ascending sort the list
        '''
        count = len(list)
        for i in range(count):
            for j in range(i + 1, count):
                if list[i][0] > list[j][0]:
                    list[i], list[j] = list[j], list[i]
        return list

    start = time()
    originalEnt = getEnt(data) # get the initial entropy
    numSample = len(data) # number of samples
    numAttribute = len(data[0]) - 1 # number of attributes (remove the Class attribute)
    #headerName = labels  # store the header name into a list
    thresholdList = [] # initial a list to store threshold of attribute
    for j in range(numAttribute): # iterate for each attribute
        column = [[sample[j], sample[-1]] for sample in data]
        column = bubble_sort(column)  # ascending sort the dataframe
        IGList = [] # initial a list to store information gain of each sample
        for i in range(numSample-1): # iterate for each sample
            before = column[:i+1] # the samples before i (including i)
            after = column[i+1:] # the samples after i (excluding i)
            beforeEnt = getIG(before) # get the entropy
            afterEnt = getIG(after)
            conditionEnt = ((i + 1) * beforeEnt[0] / numSample) + \
                           ((numSample - i - 1) * afterEnt[0] / numSample) # calculate the conditional entropy
            IGList.append(originalEnt - conditionEnt) # append the information gain into the list
        #print(IGList)
        print('%d attribute IG calculating done' % (j+1))
        index = IGList.index(max(IGList)) # get the index of maximum information gain
        threshold = (column[index][0] + column[index + 1][0]) / 2 # get the threshold
        # (mean of index value and next index value)
        thresholdList.append(threshold) # append the threshold into list
    print(thresholdList)
    stop = time()
    elapsed = stop - start
    print('Continuous data to discrete data time is: %f seconds' % elapsed)
    return thresholdList

def con2dis(thresholdList, data):
    '''function turn the continuous dataset into discrete dataset
    :param thresholdList: a list of threshold for each attribute
    :param data: the original continuous dataset
    :return: a discrete dataset
    '''
    newData = copy.deepcopy(data)
    if len(thresholdList) == (len(newData[0]) - 1):
        for attribute in range(len(thresholdList)):
            for sample in newData:
                sample[attribute] = 1 if sample[attribute] > thresholdList[attribute] else 0
        return newData
    else:
        print('Lengths of input parameters do not match')
        return 0

def foldCV(k, data, labels, function):
    '''this function computes the average and variance accuracy of input data
    k is the number of fold
    data is the original data
    labels is the dataset headers
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    return a list of accuracy and a list of confusion matrix'''
    random.shuffle(data) # permute training data at each run
    sampleCount = len(data) # get the total number of samples
    cvCount = round(sampleCount / k) # number of cross validation fold
    accuractList = [] # initialed a list to save accuracy
    confusionMatrixList = [] # initialed a list to save confusion matrix
    for i in range(k): # iterate k times
        cvFold = data[i*cvCount:(i+1)*cvCount] # get the cross validation dataset based on number of fold
        trainFold = data[0:i*cvCount] + data[(i+1)*cvCount:]# set the rest of samples
        # as training dataset
        testResult = batchPredict(trainFold, cvFold, labels, function)  # get the prediction result
        accuractList.append(calAccuracy(cvFold, testResult))  # compute accuracy and append into a list
        confusionMatrixList.append(plotConfusionMatrix(cvFold, testResult))  # same as above but confusion matrix
        #print('***The %d fold cross validation done***' % (i+1))
    return accuractList, confusionMatrixList

def tentenCV(k, data, labels, function):
    '''implement the 10-times-10-fold cross validation approach
    k is the number of fold
    data is the original data
    labels is the dataset headers
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    return a list of accuracy and a list of confusion matrix'''
    start = time()
    accuracyList = []  # initial a list to store the accuracy
    confusionMatrixList = []  # initial a list to store the confusion matrix
    for i in range(10):  # iterate ten time of k-fold cross validation
        accuracy, confusionMatrix = foldCV(k, data, labels, function)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
        print('%d time 10-fold cross validation is done' % (i + 1))
    stop = time()
    elapsed = stop - start
    print('10 - 10 fold cross validation time is: %f seconds\n' % elapsed)
    return accuracyList, confusionMatrixList

def calMean(accuracyList):
    '''
    calculate the mean value of 10-times-10-fold cross validation approach.
    :param accuracyList: the accuracy list returned from tentenCV() function
    :return: the mean value of 10-times-10-fold cross validation results
    '''
    mean10CVFoldList = list(map(lambda item: np.mean(item), accuracyList))
    return np.mean(mean10CVFoldList)

def calVariance(accuracyList):
    '''
    calculate the variance value of 10-times-10-fold cross validation approach.
    :param accuracyList: the accuracy list returned from tentenCV() function
    :return: the variance value of 10-times-10-fold cross validation results
    '''
    var10CVFoldList = list(map(lambda item: np.var(item), accuracyList))
    return np.mean(var10CVFoldList)

def getMaxAccuracyIndex(accList):
    '''
    return the index of maximum accuracy value in accList return from tentenCV() function
    :param accList: a list stored accuracy return from tentenCV() function
    :return: a list of indexes
    '''
    tempMax = []  # initial a list to store max value in each child list
    tempMaxIndex = []  # initial a list to store max value index in each child list
    for eachlist in accList:  # iterate list
        tempMax.append(max(eachlist))  # get the maximum value of each child list
        tempMaxIndex.append(eachlist.index(max(eachlist)))  # get the maximum value index of each child list
    foldIndex = tempMax.index(max(tempMax))  # get the index of which child list has maximum accuracy value
    return [foldIndex, tempMaxIndex[foldIndex]]  # return a list of index value

def plotHeatCM(CMList, indexList):
    '''
    plot the heat map of which has the maximum accuracy values
    :param CMList: a list stored confusion matrix return from tentenCV() function
    :param indexList: index of highest accuracy return form getMaxAccuracyIndex():
    :return: a confusion matrix and plot a heatmap
    '''
    CM = CMList[indexList[0]][indexList[1]]
    return CM


#Question 2:
thresholdList = getThreshold(wineList)
# returned by getThreshold() function (To save time and every run time the results are same)
# thresholdList = [12.78, 2.2350000000000003, 2.0300000000000002, 18.0, 88.5, 2.335, 1.5750000000000002,
# 0.395, 1.27, 3.46, 0.785, 2.475, 755.0]

heatmapGameLabels = ['Negative', 'Positive']
heatmapWineLabels = ['3', '2', '1']

# compute 10-times-10-fold of *game dataset* using *information gain*
print('*************** 10-times-10-fold of *game dataset* using *information gain* Start')
acc_game_IG, CM_game_IG = tentenCV(10, gameList, gameHeaders, getIG)
mean_game_IG = calMean(acc_game_IG)
var_game_IG = calVariance(acc_game_IG)
indexGIG = getMaxAccuracyIndex(acc_game_IG)
CM1 = plotHeatCM(CM_game_IG, indexGIG)
f1, ax1 = plt.subplots()
sns.heatmap(CM1, square=True, annot=True, ax=ax1, cmap="Blues")
ax1.set_xticklabels(heatmapGameLabels)
ax1.set_yticklabels(heatmapGameLabels)
ax1.set_title('Confusion Matrix of game_IG')
ax1.set_xlabel('predict')
ax1.set_ylabel('ground truth')
plt.savefig('game_IG')

# compute 10-times-10-fold of *game dataset* using *gain ratio*
print('*************** 10-times-10-fold of *game dataset* using *gain ratio* Start')
acc_game_GR, CM_game_GR = tentenCV(10, gameList, gameHeaders, getGR)
mean_game_GR = calMean(acc_game_GR)
var_game_GR = calVariance(acc_game_GR)
indexGGR = getMaxAccuracyIndex(acc_game_GR)
CM2 = plotHeatCM(CM_game_GR, indexGGR)
f2, ax2 = plt.subplots()
sns.heatmap(CM2, square=True, annot=True, ax=ax2, cmap="Blues")
ax2.set_xticklabels(heatmapGameLabels)
ax2.set_yticklabels(heatmapGameLabels)
ax2.set_title('Confusion Matrix of game_GR')
ax2.set_xlabel('predict')
ax2.set_ylabel('ground truth')
plt.savefig('game_GR')

# transform the continuous dataset into continuous dataset
wineDis = con2dis(thresholdList, wineList)

# compute 10-times-10-fold of *wine dataset* using *information gain*
print('*************** 10-times-10-fold of *wine dataset* using *information gain* Start')
acc_wine_IG, CM_wine_IG = tentenCV(10, wineDis, wineHeaders, getIG)
mean_wine_IG = calMean(acc_wine_IG)
var_wine_IG = calVariance(acc_wine_IG)
indexWIG = getMaxAccuracyIndex(acc_wine_IG)
CM3 = plotHeatCM(CM_wine_IG, indexWIG)
f3, ax3 = plt.subplots()
sns.heatmap(CM3, square=True, annot=True, ax=ax3, cmap="Blues")
ax3.set_xticklabels(heatmapWineLabels)
ax3.set_yticklabels(heatmapWineLabels)
ax3.set_title('Confusion Matrix of wine_IG')
ax3.set_xlabel('predict')
ax3.set_ylabel('ground truth')
plt.savefig('wine_IG')

# compute 10-times-10-fold of *wine dataset* using *gain ratio*
print('*************** 10-times-10-fold of *wine dataset* using *gain ratio* Start')
acc_wine_GR, CM_wine_GR = tentenCV(10, wineDis, wineHeaders, getGR)
mean_wine_GR = calMean(acc_wine_GR)
var_wine_GR = calVariance(acc_wine_GR)
indexWGR = getMaxAccuracyIndex(acc_wine_GR)
CM4 = plotHeatCM(CM_wine_GR, indexWGR)
f4, ax4 = plt.subplots()
sns.heatmap(CM4, square=True, annot=True, ax=ax4, cmap="Blues")
ax4.set_xticklabels(heatmapWineLabels)
ax4.set_yticklabels(heatmapWineLabels)
ax4.set_title('Confusion Matrix of wine_GR')
ax4.set_xlabel('predict')
ax4.set_ylabel('ground truth')
plt.savefig('wine_GR')

print('The mean and variance of the accuracy of 10 times-10 fold cross validation\n'
      'game dataset with information gain approach are:\n%f and %f' % (mean_game_IG, var_game_IG))

print('The mean and variance of the accuracy of 10 times-10 fold cross validation\n'
      'game dataset with gain ratio approach are:\n%f and %f' % (mean_game_GR, var_game_GR))

print('The mean and variance of the accuracy of 10 times-10 fold cross validation\n'
      'wine dataset with information gain approach are:\n%f and %f' % (mean_wine_IG, var_wine_IG))

print('The mean and variance of the accuracy of 10 times-10 fold cross validation\n'
      'wine dataset with gain ratio approach are:\n%f and %f' % (mean_wine_GR, var_wine_GR))
print('\n')



#Question 3 functions:
def addAttNoise(data, L, case):
    '''
    add L percent noise into each attribute (means L percent samples values in that attributes will be flipped)
    :param data: the list data structure of input data
    :param L: the percent of data be added noises
    :param case: the case of data type (discrete or continuous) one of  'dis' or 'con'
    :return: a new noised dataset (would not change input data)
    '''
    newdata = copy.deepcopy(data)  # deep copy a dataset
    numAttribute = len(newdata[0]) - 1  # total number of attributes
    numSample = len(newdata) # total number of samples
    LSampleCount = round(numSample * L)  # the number of samples should be added noise
    if numAttribute == 9:
        attributeValueList = ['o', 'x', 'b']
    elif numAttribute == 13:
        attributeValueList = [0, 1]
    else:
        print("Error: input data is not wine or game.")
        return 0
    #attributeValueList = max([list(set(sample[:-1])) for sample in newdata])
    # store all the values which under the corresponding attributes into a list uniquely
    if case == 'dis':
        for attribute in range(numAttribute):  # iterate each attributes
            random.shuffle(newdata)  # shuffle the data
            for sampleIndex in range(LSampleCount):  # iterate L percent number samples
                originalValue = newdata[sampleIndex][attribute]  # get the original value of corresponding attribute
                attributeValueList.remove(originalValue)  # remove the original value from the list
                newdata[sampleIndex][attribute] = attributeValueList[0]  # assign another value into the sample
                attributeValueList.append(originalValue)  # append back original value
        random.shuffle(newdata)  # shuffle the data
        return newdata
    elif case == 'con':
        for attribute in range(numAttribute):  # iterate each attributes
            random.shuffle(newdata)  # shuffle the data
            for sampleIndex in range(LSampleCount):  # iterate L percent number samples
                newdata[sampleIndex][attribute] += np.random.normal(loc=0, scale=1)
        random.shuffle(newdata)  # shuffle the data
        return newdata
    else:
        print('Error: Input parameter "sources" is invalid, should be "con" or "dis".')
        return 0



def addClassNoise(data, L, sources):
    '''
    Add class noises for L% of total samples (flip the class label randomly)
    :param data: list data structure of input data
    :param L: the percent of data be added noises
    :param sources: sources for class of noises, 'con' or 'mis'.
    :return: a new noised dataset (would not changed input data)
    '''
    newdata = copy.deepcopy(data)  # deep copy a dataset
    numSample = len(newdata)  # total number of samples
    LSampleCount = round(numSample * L)  # the number of samples should be added noise
    classLabelList = list(set([sample[-1] for sample in newdata]))  # store all the values which under the class into a list
    random.shuffle(newdata)  # shuffle the data in case always first several sample are deleted

    if len(classLabelList) == 1:
        print('Only one class in the dataset, cannot add class noise in this dataset')
        return 0.
    if sources == 'con':
        del newdata[:LSampleCount]  # delete first LSampleCount number of sample
        noises = copy.deepcopy(newdata[:LSampleCount])  # copy the first LSampleCount of removed data
        for sample in noises:
            originalValue = sample[-1]  # get and store the original class label
            classLabelList.remove(originalValue)  # remove that label
            sample[-1] = classLabelList[0]  # assign another class label to that sample
            classLabelList.append(originalValue)  # put original value back to the list
        newdata.extend(noises)  # put the noises back into the dataset
        #return random.shuffle(newdata)
        return newdata
    elif sources == 'mis':
        for sample in newdata[:LSampleCount]:
            originalValue = sample[-1]  # get and store the original class label
            classLabelList.remove(originalValue)  # remove that label
            sample[-1] = classLabelList[0]  # assign another class label to that sample
            classLabelList.append(originalValue)  # put original value back to the list
        #return random.shuffle(newdata)
        return newdata
    else:
        print('Error: Input parameter "sources" is invalid, should be "con" or "mis".')
        return 0


def foldCVNoise(k, data, labels, function, noiseP, whom, case):
    '''this function computes the average and variance accuracy of input data
    k is the number of fold
    data is the original data
    labels is the dataset headers
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    noiseP the percentage of noise
    whom indicates which dataset should be add noised
    case: the case of data type (discrete or continuous) one of  'dis' or 'con'
    return a list of accuracy and a list of confusion matrix'''
    random.shuffle(data) # permute training data at each run
    sampleCount = len(data) # get the total number of samples
    cvCount = round(sampleCount / k) # number of cross validation fold
    accuractList = [] # initialed a list to save accuracy
    confusionMatrixList = [] # initialed a list to save confusion matrix
    if whom == 'train':
        for i in range(k): # iterate k times
            cvFold = data[i*cvCount:(i+1)*cvCount] # get the cross validation dataset based on number of fold
            trainFold = addAttNoise(data[0:i*cvCount] + data[(i+1)*cvCount:], noiseP, case)# set the rest of samples
            # as training dataset
            testResult = batchPredict(trainFold, cvFold, labels, function)  # get the prediction result
            accuractList.append(calAccuracy(cvFold, testResult))  # compute accuracy and append into a list
            confusionMatrixList.append(plotConfusionMatrix(cvFold, testResult))  # same as above but confusion matrix
            #print('***The %d fold cross validation done***' % (i+1))
        return accuractList, confusionMatrixList
    elif whom == 'test':
        for i in range(k): # iterate k times
            cvFold = addAttNoise(data[i*cvCount:(i+1)*cvCount], noiseP, case) # get the cross validation dataset based on number of fold
            trainFold = data[0:i*cvCount] + data[(i+1)*cvCount:]# set the rest of samples
            # as training dataset
            testResult = batchPredict(trainFold, cvFold, labels, function)  # get the prediction result
            accuractList.append(calAccuracy(cvFold, testResult))  # compute accuracy and append into a list
            confusionMatrixList.append(plotConfusionMatrix(cvFold, testResult))  # same as above but confusion matrix
            #print('***The %d fold cross validation done***' % (i+1))
        return accuractList, confusionMatrixList
    elif whom == 'both':
        for i in range(k): # iterate k times
            cvFold = addAttNoise(data[i*cvCount:(i+1)*cvCount], noiseP, case) # get the cross validation dataset based on number of fold
            trainFold = addAttNoise(data[0:i*cvCount] + data[(i+1)*cvCount:], noiseP, case)# set the rest of samples
            # as training dataset
            testResult = batchPredict(trainFold, cvFold, labels, function)  # get the prediction result
            accuractList.append(calAccuracy(cvFold, testResult))  # compute accuracy and append into a list
            confusionMatrixList.append(plotConfusionMatrix(cvFold, testResult))  # same as above but confusion matrix
            #print('***The %d fold cross validation done***' % (i+1))
        return accuractList, confusionMatrixList
    elif whom == 'neither':
        for i in range(k): # iterate k times
            cvFold = data[i*cvCount:(i+1)*cvCount]  # get the cross validation dataset based on number of fold
            trainFold = data[0:i*cvCount] + data[(i+1)*cvCount:]  # set the rest of samples
            # as training dataset
            testResult = batchPredict(trainFold, cvFold, labels, function)  # get the prediction result
            accuractList.append(calAccuracy(cvFold, testResult))  # compute accuracy and append into a list
            confusionMatrixList.append(plotConfusionMatrix(cvFold, testResult))  # same as above but confusion matrix
            #print('***The %d fold cross validation done***' % (i+1))
        return accuractList, confusionMatrixList
    else:
        print('Error: input parameter is wrong.')
        return 0

def tentenCVNoise(k, data, labels, function, noiseP, whom, case):
    '''implement the 10-times-10-fold cross validation approach
    k is the number of fold
    data is the original data
    labels is the dataset headers
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    noiseP the percentage of noise
    whom indicates which dataset should be add noised
    case: the case of data type (discrete or continuous) one of  'dis' or 'con'
    return a list of accuracy and a list of confusion matrix'''
    start = time()
    accuracyList = []  # initial a list to store the accuracy
    confusionMatrixList = []  # initial a list to store the confusion matrix
    for i in range(10):  # iterate ten time of k-fold cross validation
        accuracy, confusionMatrix = foldCVNoise(k, data, labels, function, noiseP, whom, case)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
        print('%d time 10-fold cross validation is done' % (i + 1))
    stop = time()
    elapsed = stop - start
    print('10 - 10 fold cross validation time is: %f seconds\n' % elapsed)
    return accuracyList, confusionMatrixList


#Question3:
#Game with noise
#CvsC
print('Game with noise start.')
print('CvsC accuracy generating start.')
acc_game_IG_CC_0, CM_game_IG_CC_0 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0, 'neither', 'dis')
acc_game_IG_CC_5, CM_game_IG_CC_5 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.05, 'neither', 'dis')
acc_game_IG_CC_10, CM_game_IG_CC_10 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.1, 'neither', 'dis')
acc_game_IG_CC_15, CM_game_IG_CC_15 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.15, 'neither', 'dis')
CCAccuracyMean = [calMean(acc_game_IG_CC_0), calMean(acc_game_IG_CC_5), calMean(acc_game_IG_CC_10),
                  calMean(acc_game_IG_CC_15)]
CCAccuracyVar = [calVariance(acc_game_IG_CC_0), calVariance(acc_game_IG_CC_5), calVariance(acc_game_IG_CC_10),
                 calVariance(acc_game_IG_CC_15)]
# DvsC
print('DvsC accuracy generating start.')
acc_game_IG_DC_0, CM_game_IG_DC_0 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0, 'train', 'dis')
acc_game_IG_DC_5, CM_game_IG_DC_5 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.05, 'train', 'dis')
acc_game_IG_DC_10, CM_game_IG_DC_10 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.1, 'train', 'dis')
acc_game_IG_DC_15, CM_game_IG_DC_15 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.15, 'train', 'dis')
DCAccuracyMean = [calMean(acc_game_IG_DC_0), calMean(acc_game_IG_DC_5), calMean(acc_game_IG_DC_10),
                  calMean(acc_game_IG_DC_15)]
DCAccuracyVar = [calVariance(acc_game_IG_DC_0), calVariance(acc_game_IG_DC_5), calVariance(acc_game_IG_DC_10),
                 calVariance(acc_game_IG_DC_15)]
# CvsD
print('CvsD accuracy generating start.')
acc_game_IG_CD_0, CM_game_IG_CD_0 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0, 'test', 'dis')
acc_game_IG_CD_5, CM_game_IG_CD_5 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.05, 'test', 'dis')
acc_game_IG_CD_10, CM_game_IG_CD_10 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.1, 'test', 'dis')
acc_game_IG_CD_15, CM_game_IG_CD_15 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.15, 'test', 'dis')
CDAccuracyMean = [calMean(acc_game_IG_CD_0), calMean(acc_game_IG_CD_5), calMean(acc_game_IG_CD_10),
                  calMean(acc_game_IG_CD_15)]
CDAccuracyVar = [calVariance(acc_game_IG_CD_0), calVariance(acc_game_IG_CD_5), calVariance(acc_game_IG_CD_10),
                 calVariance(acc_game_IG_CD_15)]
# DvsD
print('DvsD accuracy generating start.')
acc_game_IG_DD_0, CM_game_IG_DD_0 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0, 'both', 'dis')
acc_game_IG_DD_5, CM_game_IG_DD_5 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.05, 'both', 'dis')
acc_game_IG_DD_10, CM_game_IG_DD_10 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.1, 'both', 'dis')
acc_game_IG_DD_15, CM_game_IG_DD_15 = tentenCVNoise(10, gameList, gameHeaders, getIG, 0.15, 'both', 'dis')
DDAccuracyMean = [calMean(acc_game_IG_DD_0), calMean(acc_game_IG_DD_5), calMean(acc_game_IG_DD_10),
                  calMean(acc_game_IG_DD_15)]
DDAccuracyVar = [calVariance(acc_game_IG_DD_0), calVariance(acc_game_IG_DD_5), calVariance(acc_game_IG_DD_10),
                 calVariance(acc_game_IG_DD_15)]

#Wine with noise
#CvsC
print('Wine with noise start.')
print('CvsC accuracy generating start.')
acc_wine_IG_CC_0, CM_wine_IG_CC_0 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0, 'neither', 'dis')
acc_wine_IG_CC_5, CM_wine_IG_CC_5 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.05, 'neither', 'dis')
acc_wine_IG_CC_10, CM_wine_IG_CC_10 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.1, 'neither', 'dis')
acc_wine_IG_CC_15, CM_wine_IG_CC_15 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.15, 'neither', 'dis')
CCWineAccuracyMean = [calMean(acc_wine_IG_CC_0), calMean(acc_wine_IG_CC_5), calMean(acc_wine_IG_CC_10),
                  calMean(acc_wine_IG_CC_15)]
CCWineAccuracyVar = [calVariance(acc_wine_IG_CC_0), calVariance(acc_wine_IG_CC_5), calVariance(acc_wine_IG_CC_10),
                 calVariance(acc_wine_IG_CC_15)]
# DvsC
print('DvsC accuracy generating start.')
acc_wine_IG_DC_0, CM_wine_IG_DC_0 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0, 'train', 'dis')
acc_wine_IG_DC_5, CM_wine_IG_DC_5 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.05, 'train', 'dis')
acc_wine_IG_DC_10, CM_wine_IG_DC_10 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.1, 'train', 'dis')
acc_wine_IG_DC_15, CM_wine_IG_DC_15 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.15, 'train', 'dis')
DCWineAccuracyMean = [calMean(acc_wine_IG_DC_0), calMean(acc_wine_IG_DC_5), calMean(acc_wine_IG_DC_10),
                  calMean(acc_wine_IG_DC_15)]
DCWineAccuracyVar = [calVariance(acc_wine_IG_DC_0), calVariance(acc_wine_IG_DC_5), calVariance(acc_wine_IG_DC_10),
                 calVariance(acc_wine_IG_DC_15)]
# CvsD
print('CvsD accuracy generating start.')
acc_wine_IG_CD_0, CM_wine_IG_CD_0 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0, 'test', 'dis')
acc_wine_IG_CD_5, CM_wine_IG_CD_5 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.05, 'test', 'dis')
acc_wine_IG_CD_10, CM_wine_IG_CD_10 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.1, 'test', 'dis')
acc_wine_IG_CD_15, CM_wine_IG_CD_15 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.15, 'test', 'dis')
CDWineAccuracyMean = [calMean(acc_wine_IG_CD_0), calMean(acc_wine_IG_CD_5), calMean(acc_wine_IG_CD_10),
                  calMean(acc_wine_IG_CD_15)]
CDWineAccuracyVar = [calVariance(acc_wine_IG_CD_0), calVariance(acc_wine_IG_CD_5), calVariance(acc_wine_IG_CD_10),
                 calVariance(acc_wine_IG_CD_15)]
# DvsD
print('DvsD accuracy generating start.')
acc_wine_IG_DD_0, CM_wine_IG_DD_0 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0, 'both', 'dis')
acc_wine_IG_DD_5, CM_wine_IG_DD_5 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.05, 'both', 'dis')
acc_wine_IG_DD_10, CM_wine_IG_DD_10 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.1, 'both', 'dis')
acc_wine_IG_DD_15, CM_wine_IG_DD_15 = tentenCVNoise(10, wineDis, wineHeaders, getIG, 0.15, 'both', 'dis')
DDWineAccuracyMean = [calMean(acc_wine_IG_DD_0), calMean(acc_wine_IG_DD_5), calMean(acc_wine_IG_DD_10),
                  calMean(acc_wine_IG_DD_15)]
DDWineAccuracyVar = [calVariance(acc_wine_IG_DD_0), calVariance(acc_wine_IG_DD_5), calVariance(acc_wine_IG_DD_10),
                 calVariance(acc_wine_IG_DD_15)]



'''
train = ga.iloc[50:,:]
test = ga.iloc[:50,:]
test = batchPredict(train, test)
print(test)
print('The accuracy is: %f' % calAccuracy(test))
plotConfusionMatrix(test)
'''
#accuracy, CM = tentenCV(10, game)

'''
wine = randomShuffle(wine)
wine1 = con2dis(thresholdList, wine)
trainWine = wine1.iloc[20:, :]
testWine = wine1.iloc[:20, :]
testWine = batchPredict(trainWine, testWine)
print('The accuracy is: %f' % calAccuracy(testWine))
'''
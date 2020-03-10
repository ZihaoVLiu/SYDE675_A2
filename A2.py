from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots()
sns.set()

#open the data using read_csv of pandas
game = pd.read_csv('tic-tac-toe.data', header = None)
wine = pd.read_csv('wine.data', header= None)

#modify the first column of wine dataset to the last
order = [1,2,3,4,5,6,7,8,9,10,11,12,13,0]
wine = wine[order]

#modify the header name
game.columns = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'Class']
wineHeaders = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline', 'Class']
wine.columns = wineHeaders

#randmoly shuffle rows and keep the index order
#this operation disruptes data consistency, which could enhance the classification accuracy
gameR = game.sample(frac=1).reset_index(drop=True)
ga = pd.read_csv('wocao.data')
ga = ga.iloc[:, 1:]

def getEnt(data):
    '''Calculate the  of given dataset
    input parameter is the original dataset
    return value is the entropy vlaue'''
    numSample = data.shape[0] # Total number of input dataset
    infoSat = data.iloc[:,-1].value_counts() # statistic information of given class
    p = infoSat/numSample # the probability of each class
    return sum(-p * np.log2(p)) # get the entropy of input dataset

def getIG(data):
    '''Calculate the information gain value of each attribute
    input parameter is the original dataset
    return value is the information gain of each attribute'''
    IG = list() # initialed a list for information gain of each attribute
    originalEnt = getEnt(data) # get the initial entropy
    numAttribute = data.shape[1] - 1 # get the total number of attribute
    for i in range(numAttribute): # iterate each attribute
        vIndex = data.iloc[:,i].value_counts().index # get all unique value of the attribute
        attributEnt = 0
        for j in vIndex: # iterate each value in the attribute
            itemData = data[data.iloc[:, i] == j] # get dataset of indicated attribute value
            # ('j' in this iteration)
            itemEnt = getEnt(itemData) # get the entropy of the specific attribute value
            attributEnt += (itemData.shape[0] / data.shape[0]) * itemEnt
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
    IG = getIG(data)
    originalEnt = getEnt(data)  # get the initial entropy
    numAttribute = data.shape[1] - 1  # get the total number of attribute
    numSample = data.shape[0]  # Total number of input dataset
    for i in range(numAttribute): # iterate each attribute
        vIndex = data.iloc[:,i].value_counts().index # get all unique value of the attribute
        attributEnt = 0
        infoSat = data.iloc[:, i].value_counts()  # statistic information of given class
        p = infoSat / numSample  # the probability of each class
        splitInformation = sum(-p * np.log2(p))  # get the entropy of Intrinsic Information of the Attribute
        SI.append(splitInformation)
        for j in vIndex: # iterate each value in the attribute
            itemData = data[data.iloc[:, i] == j] # get dataset of indicated attribute value
            # ('j' in this iteration)
            itemEnt = getEnt(itemData) # get the entropy of the specific attribute value
            attributEnt += (itemData.shape[0] / data.shape[0]) * itemEnt
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

def getMax(IG):
    '''Output the attribute index with the highest information gain
    input parameter is a information gain list (get from getIG() or getGR() function)
    return the index number'''
    return IG.index(max(IG))

def discardAttri(data, index, attributeValue):
    '''discard indicated attribute but keep attribute value of given column
    data: original dataset
    index: the value returned from getMax. Indicated attribute will be discarded
    attributeValue: indicated attribute value'''
    column = data.columns[index]
    return data.loc[data[column] == attributeValue, :].drop(column, axis=1)

def buildTree(data, function):
    '''build a decision tree recursively based on ID3 algorithm
    input parameter is the original dataset
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    return a dictionary type tree'''
    attributeList = list(data.columns) # store all attribute index into a list
    resultList = data.iloc[:, -1].value_counts() # store the result
    # statistical information into a list
    dataShape = data.shape

    # define the recursive termination condition
    # only one attribute is left or the results of leaf node are same.
    if dataShape[1] == 1 or resultList.values[0] == dataShape[0]:
        return resultList.index[0] # return the result label

    IGList = function(data) # get the information gain list
    maxIGIndex = getMax(IGList) # get index with the highest IG
    attributeName = attributeList[maxIGIndex] # pair the index with corresponding attribute name
    #print('The attribute name is: %s' %attributeName)
    decisionTree = {attributeName: {}} # store the decision tree into a dictionary
    attributeList.remove(attributeName) # remove the attribute just utilized
    values = data.iloc[:, maxIGIndex].value_counts().index # get the unique attribute value of
    # corresponding attribute
    for value in values: # build branch for each attribute value
        if data.iloc[:,maxIGIndex].value_counts()[value] == 0:
            decisionTree[attributeName][value] = resultList.index[0]
        else:
            decisionTree[attributeName][value] = buildTree(discardAttri(data, maxIGIndex, value), function)
    return decisionTree

# 决策树创建
def buildTree1(data, function):
    attributeList = list(data.columns)  # store all attribute index into a list
    resultList = list(data.iloc[:, -1])  # store the class labels result
    dataSet = []
    for item in range(data.shape[0]):
        dataSet.append(list(data.iloc[item, :]))

    # 获取标签属性，dataSet最后一列，区别于labels标签名称
    #classList = [example[-1] for example in dataSet]
    # 树极端终止条件判断
    # 标签属性值全部相同，返回标签属性第一项值
    if resultList.count(resultList[0]) == len(resultList):
        return resultList[0]
    # 只有一个特征（1列）
    if len(dataSet[0]) == 1:
        return data.iloc[:, -1].value_counts().index[0]
    # 获取最优特征列索引
    #bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    # 获取最优索引对应的标签名称
    #bestFeatureLabel = attributeList[bestFeatureIndex]

    IGList = function(data)  # get the information gain list
    maxIGIndex = getMax(IGList)  # get index with the highest IG
    attributeName = attributeList[maxIGIndex]  # pair the index with corresponding attribute name

    # 创建根节点
    myTree = {attributeName: {}}
    # 去除最优索引对应的标签名，使labels标签能正确遍历
    del (attributeList[maxIGIndex])
    # 获取最优列
    bestFeature = [example[maxIGIndex] for example in dataSet]
    uniquesVals = set(bestFeature)
    for value in uniquesVals:
        # 子标签名称集合
        subLabels = attributeList[:]
        # 递归
        myTree[attributeName][value] = buildTree(discardAttri(data, maxIGIndex, value), function)
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
    test: to get the header of data (using .columns to get the list)
    sample: a list, the input test sample
    train: the training dataset providing the class label with highest value count
    return the prediction value'''
    if type(tree) == str: # in case the input tree only has a root node
        return tree

    maxClassLabel = train.iloc[:, -1].value_counts().index[0] # class label with highest value count
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

def batchPredict(train, test, function):
    '''output the test set with a column of prediction values
    train: training dataset
    function: the function name of decision-tree learning algorithms (one of getIG or getGR)
    test: testing dataset'''
    testSampleNumber = test.shape[0] # get the sample number of test set
    tree = buildTree(train, function) # construct the tree
    labels = list(train.columns) # get all the header number for predictResult function
    result = [] # initial a list to store prediction value
    for item in range(testSampleNumber): # iterate each sample in test dataset
        sample = test.iloc[item, :-1]
        result.append(predictResult(tree, labels, sample, train))
    test.insert(test.shape[1], 'predict', result)
    #test['predict'] = result
    return test

def plotConfusionMatrix(data):
    '''output the confusion matrix and draw heatmap
    data: dataset returned from batchPredict() function
    print the confusion matrix and draw a confusion matrix heatmap'''
    tResult = data.iloc[:, -2]
    pResult = data.iloc[:, -1]
    c = confusion_matrix(tResult, pResult)
    #sns.heatmap(c, annot=True, ax=ax)
    #ax.set_title('Confusion Matrix')
    #ax.set_xlabel('predict')
    #ax.set_ylabel('true ground')
    #plt.show()
    return c

def calAccuracy(test):
    '''calculate the accuracy of test prediction dataset
    test is the input test dataset returned from batchPredict
    return a accuracy value'''
    totalTrue = sum(test.iloc[:,-1]==test.iloc[:,-2]) # calculate the same total number
    # between ground truth and prediction
    total = test.shape[0]
    accuracy = totalTrue / total
    return accuracy

def randomShuffle(data):
    '''randomly shuffle the input dataset
    the input parameter is the original dataset
    return a randomly shuffle dataset'''
    RSdata = data.sample(frac=1).reset_index(drop=True)
    return RSdata

def getThreshold(data):
    '''function get the threshold of each attribute based on information gain approach
    input parameter is the original dataset
    return dataset is a discrete dataset'''
    start = time()
    originalEnt = getEnt(data) # get the initial entropy
    numSample = data.shape[0] # number of samples
    numAttribute = data.shape[1]-1 # number of attributes (remove the Class attribute)
    headerName = data.columns # store the header name into a list
    thresholdList = [] # initial a list to store threshold of attribute
    for j in range(numAttribute): # iterate for each attribute
        column = data.iloc[:, [j,-1]] # get the corresponding attribute and Class attribute into dataframe
        column = column.sort_values(headerName[j])  # ascending sort the dataframe
        IGList = [] # initial a list to store information gain of each sample
        for i in range(numSample): # iterate for each sample
            before = column.iloc[:i+1, :] # the samples before i (including i)
            after = column.iloc[i+1:, :] # the samples after i (excluding i)
            beforeEnt = getIG(before) # get the entropy
            afterEnt = getIG(after)
            conditionEnt = ((i + 1) * beforeEnt[0] / numSample) + \
                           ((numSample - i - 1) * afterEnt[0] / numSample) # calculate the conditional entropy
            IGList.append(originalEnt - conditionEnt) # append the information gain into the list
        print(IGList)
        print('%d attribute IG calculating done' % (j+1))
        index = IGList.index(max(IGList)) # get the index of maximum information gain
        threshold = (column.iloc[index, 0] + column.iloc[index + 1, 0]) / 2 # get the threshold
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
    if len(thresholdList) == (data.shape[1] - 1):
        headerName = list(data.columns) # get the header of dataset
        newdata = np.arange(data.shape[0] * data.shape[1]).reshape(data.shape[0], data.shape[1]) # create new dataframe
        newdata_df = pd.DataFrame(newdata)
        newdata_df.iloc[:, -1] = data.iloc[:, -1]
        newdata_df.columns = headerName
        numAttribute = data.shape[1]-1 # get the number of attribute (except the Class label)
        for i in range(numAttribute): # iterate each attribute
            column = data.iloc[:,i] # get the corresponding attribute
            indexLargeList = column[column > thresholdList[i]].index # get index that larger than threshold
            indexSmallList = column[column <= thresholdList[i]].index # get index that smaller than threshold
            newdata_df.iloc[indexLargeList, i] = 1 # set 1 to those larger index
            newdata_df.iloc[indexSmallList, i] = 0 # set 0 to those smaller index
        return newdata_df
    else:
        print('Lengths of input parameters do not natch')

def foldCV(k, data, function):
    '''this function computes the average and variance accuracy of input data
    k is the number of fold
    data is the original data
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    return a list of accuracy and a list of confusion matrix'''
    data = randomShuffle(data) # permute training data at each run
    sampleCount = data.shape[0] # get the total number of samples
    cvCount = round(sampleCount / k) # number of cross validation fold
    accuractList = [] # initialed a list to save accuracy
    confusionMatrixList = [] # initialed a list to save confusion matrix
    for i in range(k): # iterate k times
        cvFold = data.iloc[i*cvCount:(i+1)*cvCount,:] # get the cross validation dataset based on number of fold
        trainFold = pd.concat([data.iloc[0:i*cvCount,:], data.iloc[(i+1)*cvCount:,:]]) # set the rest of samples
        # as training dataset
        cvFold = batchPredict(trainFold, cvFold, function)  # get the prediction result
        accuractList.append(calAccuracy(cvFold))  # compute accuracy and append into a list
        confusionMatrixList.append(plotConfusionMatrix(cvFold))  # same as above but confusion matrix
        print('***The %d fold cross validation done***' % (i+1))
    return accuractList, confusionMatrixList

def tentenCV(k, data, function):
    '''implement the 10-times-10-fold cross validation approach
    k is the number of fold
    data is the original data
    function parameter is the function name of decision-tree learning algorithms (one of getIG or getGR)
    return a list of accuracy and a list of confusion matrix'''
    start = time()
    accuracyList = []  # initial a list to store the accuracy
    confusionMatrixList = []  # initial a list to store the confusion matrix
    for i in range(10):  # iterate ten time of k-fold cross validation
        accuracy, confusionMatrix = foldCV(k, data, function)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
        print('%d time 10-fold cross validation is done' % (i + 1))
    stop = time()
    elapsed = stop - start
    print('10 - 10 fold cross validation time is: %f seconds' % elapsed)
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

def addAttNoise(data, L):
    '''
    Add an attributes noise for L% of total samples (flip the attribute randomly).
    :param data: the target noise adding dataset
    :param L: the percentage of noise (must between 0 and 1)
    :return: a dataset with L% noise
    '''
    import random
    numAttribute = data.shape[1] - 1  # get the number of attribute (except the class)
    numSample = data.shape[0]  # get the number of samples
    numL = round(L * numSample)  # get the number of L * total
    data = randomShuffle(data)  # shuffle the input data to make sure the noise randomly
    newData = data.iloc[:,:].copy()  # copy the input data
    for i in range(numL):  # iterate numL times to select first numL samples
        attributeRandomIndex = random.randint(0, numAttribute - 1)  # get the random attribute index
        valuesInAttribute = list(data.iloc[:, attributeRandomIndex].value_counts().index)  # store all the values
        # under indicated attribute
        valueRandomIndex = random.randint(0, len(valuesInAttribute) - 2)  # get the random value index under attribute
        valuesInAttribute.remove(data.iloc[i, attributeRandomIndex])  # remove attribute value already exist in the list
        noise = valuesInAttribute[valueRandomIndex]  # select another value as noise
        newData.iloc[i, attributeRandomIndex] = noise  # assign the noise value
    return randomShuffle(newData)


# thresholdList = getThreshold(wine)
# returned by getThreshold() function (To save time and every run time the results are same)
thresholdList = [12.78, 2.2350000000000003, 2.0300000000000002, 18.0, 88.5, 2.335, 1.5750000000000002,
                 0.395, 1.27, 3.46, 0.785, 2.475, 755.0]
'''
# compute 10-times-10-fold of *game dataset* using *information gain*
print('*************** 10-times-10-fold of *game dataset* using *information gain* Start')
acc_game_IG, CM_game_IG = tentenCV(10, game, getIG)
mean_game_IG = calMean(acc_game_IG)
var_game_IG = calVariance(acc_game_IG)
# compute 10-times-10-fold of *game dataset* using *gain ratio*
print('*************** 10-times-10-fold of *game dataset* using *gain ratio* Start')
acc_game_GR, CM_game_GR = tentenCV(10, game, getGR)
mean_game_GR = calMean(acc_game_GR)
var_game_GR = calVariance(acc_game_GR)

# transform the continuous dataset into continuous dataset
wineDis = con2dis(thresholdList, wine)

# compute 10-times-10-fold of *wine dataset* using *information gain*
print('*************** 10-times-10-fold of *wine dataset* using *information gain* Start')
acc_wine_IG, CM_wine_IG = tentenCV(10, wineDis, getIG)
mean_wine_IG = calMean(acc_wine_IG)
var_wine_IG = calVariance(acc_wine_IG)
# compute 10-times-10-fold of *wine dataset* using *gain ratio*
print('*************** 10-times-10-fold of *wine dataset* using *gain ratio* Start')
acc_wine_GR, CM_wine_GR = tentenCV(10, wineDis, getGR)
mean_wine_GR = calMean(acc_wine_GR)
var_wine_GR = calVariance(acc_wine_GR)
'''




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
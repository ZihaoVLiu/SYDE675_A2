import numpy as np
import pandas as pd
import os
os.environ['PATH']

#open the data using read_csv of pandas
game = pd.read_csv('tic-tac-toe.data', header = None)
wine = pd.read_csv('wine.data', header= None)
sample = pd.read_csv('sample.data', header = None)

#modify the header name
game.columns = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'Class']
sample.columns = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'Class']

#randmoly shuffle rows and keep the index order
#this operation disruptes data consistency, which could enhance the classification accuracy
gameR = game.sample(frac=1).reset_index(drop=True)

def createDataSet():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

dataSet = createDataSet()

#open the data using read_csv of pandas
game = pd.read_csv('tic-tac-toe.data', header = None)
wine = pd.read_csv('wine.data', header= None)
game.columns = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'Class']

def calEnt(dataSet):
    n = dataSet.shape[0]                             #数据集总行数
    iset = dataSet.iloc[:,-1].value_counts()         #标签的所有类别
    p = iset/n                                       #每一类标签所占比
    ent = (-p*np.log2(p)).sum()                      #计算信息熵
    return ent

def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                                #计算原始熵
    bestGain = 0                                             #初始化信息增益
    axis = -1                                                #初始化最佳切分列，标签列
    for i in range(dataSet.shape[1]-1):                      #对特征的每一列进行循环
        levels= dataSet.iloc[:,i].value_counts().index       #提取出当前列的所有取值
        ents = 0                                             #初始化子节点的信息熵
        for j in levels:                                     #对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:,i]==j]         #某一个子节点的dataframe
            ent = calEnt(childSet)                           #计算某一个子节点的信息熵
            ents += (childSet.shape[0]/dataSet.shape[0])*ent #计算当前列的信息熵
        #print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt-ents                              #计算当前列的信息增益
        #print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain                              #选择最大信息增益
            axis = i                                         #最大信息增益所在列的索引
    return axis

def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)
    return redataSet

def createTree(dataSet):
    featlist = list(dataSet.columns)                          #提取出数据集所有的列
    classlist = dataSet.iloc[:,-1].value_counts()             #获取最后一列类标签
    #判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist[classlist.index[0]]==dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]                             #如果是，返回类标签
    axis = bestSplit(dataSet)                                 #确定出当前最佳切分列的索引
    bestfeat = featlist[axis]                                 #获取该索引对应的特征
    myTree = {bestfeat:{}}                                    #采用字典嵌套的方式存储树信息
    del featlist[axis]                                        #删除当前特征
    valuelist = set(dataSet.iloc[:,axis])                     #提取最佳切分列所有属性值
    for value in valuelist:                                   #对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
    return myTree




def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree))                   #获取决策树第一个节点
    secondDict = inputTree[firstStr]                   #下一个字典
    featIndex = labels.index(firstStr)                 #第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict :
                if classify(secondDict[key], labels, testVec) is None:
                    print('fuck')
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def acc_classify(train,test):
    inputTree = createTree(train)                       #根据测试集生成一棵树
    labels = list(train.columns)                        #数据集所有的列名称
    result = []
    for i in range(test.shape[0]):                      #对测试集中每一条数据进行循环
        testVec = test.iloc[i,:-1]                      #测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec) #预测该实例的分类
        result.append(classLabel)                       #将分类结果追加到result列表中
    test[10]=result                              #将预测结果追加到测试集最后一列
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()     #计算准确率
    print(f'模型预测准确率为{acc}')
    return test

def predictResult(tree, labels, sample):
    '''use decision tree to predict the class label of the input sample
    tree: decision tree build by buildTree() function
    test: to get the header of data (using .columns to get the list)
    sample: a list, the input test sample
    return the prediction value'''

    '''if type(tree) == str: # in case the input tree only has a root node
        return tree'''

    #upperNode = list(tree.keys())[0] # get the upper tree node of each recursion
    upperNode = next(iter(tree))
    upperNodeIndex = labels.index(upperNode) # get the index number of corresponding tree node
    branches = tree[upperNode] # get the branches under the upperNode
    for twig in branches.keys(): # iterate each twig under the upperNode key (among branches)
        if sample[upperNodeIndex] == twig: # detect if the corresponding node name equals to twig
            if type(branches[twig]) == dict:
                result = predictResult(branches[twig], labels, sample) # continue recursion but use twig branch
            else:
                result = branches[twig] # return the class result
    return result

def batchPredict(train, test):
    '''output the test set with a column of prediction values
    train: training dataset
    test: testing dataset'''
    testSampleNumber = test.shape[0] # get the sample number of test set
    tree = buildTree(train) # construct the tree
    labels = list(train.columns) # get all the header number for predictResult function
    result = [] # initial a list to store prediction value
    for item in range(testSampleNumber): # iterate each sample in test dataset
        sample = test.iloc[item, :-1]
        result.append(predictResult(tree, labels, sample))
    test.insert(test.shape[1], 'predict', result)
    #test['predict'] = result
    return test

ga = pd.read_csv('wocao.data')
ga = ga.iloc[:, 1:]

train = ga.iloc[50:,:]
test = ga.iloc[10,:]
tree = createTree(train)
labels = list(train.columns)
predictResult(tree,labels,test)
#test = gameR.iloc[:50,:]
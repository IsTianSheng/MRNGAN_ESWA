# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""

from collections import defaultdict

import numpy as np
import torch
import pandas as pd

def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    for line in open(trainFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        trainSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Training data loading done" )
    return trainSet,userCount,itemCount


def loadTrainingData_modify(trainFile, splitMark):
    trainSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1

    for line in open(trainFile):
        data = line.strip().split(splitMark)
        userId = int(data[0]) - 1
        itemId = int(data[1]) - 1
        trainSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1

    # To get two matrixes
    interact_matrix, rating_matrix = np.zeros((userCount, itemCount)), np.zeros((userCount, itemCount))
    for line in open(trainFile):
        data = line.strip().split(splitMark)
        userId = int(data[0]) - 1
        itemId = int(data[1]) - 1
        interact_matrix[userId][itemId] = 1
        rating_matrix[userId][itemId] = data[2]

    # print("Training data loading done")
    return trainSet, userCount, itemCount, interact_matrix, rating_matrix



def loadTestData(testFile, splitMark):
    testSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    for line in open(testFile):
        data = line.strip().split(splitMark)
        userId = int(data[0]) - 1
        itemId = int(data[1]) - 1
        testSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    # print("Test data loading done")
    return testSet, userCount, itemCount


def to_Vectors(trainSet, userCount, itemCount, userList_test, mode):
    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount  # ¸Ä¶¯  Ö±½ÓÐ´³ÉuserCount
    if mode == "itemBased":  # ¸Ä¶¯  itemCount userCount»¥»»   batchCountÊÇÎïÆ·Êý
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            if mode == "userBased":
                trainDict[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = 1.0
    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])
    #    print("Converting to vectors done....")
    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount


def to_Vectors_trueRate(trainSet, userCount, itemCount, userList_test, mode, rating_matrix):
    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount  # ¸Ä¶¯  Ö±½ÓÐ´³ÉuserCount
    if mode == "itemBased":  # ¸Ä¶¯  itemCount userCount»¥»»   batchCountÊÇÎïÆ·Êý
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            if mode == "userBased":
                trainDict[userId][itemId] = rating_matrix[userId,itemId]
            else:
                trainDict[itemId][userId] = rating_matrix[userId,itemId]
    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])
    #    print("Converting to vectors done....")
    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount



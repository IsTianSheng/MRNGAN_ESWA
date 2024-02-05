import random
import copy
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import Utils
import cfgan
import warnings
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")



def main(userCount, itemCount, testSet, trainVector, trainMaskVector, \
         KRNN, NS, TopN, epochCount):
    KRNN = torch.tensor(np.array(KRNN).astype(np.float32))

    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1
    # Build the generator and discriminator
    G = cfgan.generator(itemCount, userCount)
    # G = cfgan.generator_no_userInfo(itemCount)
    D = cfgan.discriminator(itemCount)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step = 5
    D_step = 2
    batchSize_G = 32
    batchSize_D = 32

    result_quotas = np.zeros([epochCount, 15])

    for epoch in range(epochCount):

        # ---------------------
        #  Train Generator
        # ---------------------

        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(trainVector[leftIndex:leftIndex + batchSize_G])
            tu = Variable(trainVector[leftIndex:leftIndex + batchSize_G])

            KRNNeighbors = Variable(copy.deepcopy(KRNN[leftIndex:leftIndex + batchSize_G]))

            realData_zp = Variable(torch.ones_like(realData)) * tu

            # Generate a batch of new purchased vector
            G_prediction = G(realData, KRNNeighbors)
            # G_prediction = G(realData)
            G_prediction_ZP = G_prediction * tu
            G_prediction_result = D(G_prediction_ZP)

            # Train the discriminator
            g_loss = np.mean(np.log(1. - G_prediction_result.detach().numpy() + 10e-5)) + alpha * regularization(
                G_prediction, realData)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_D - 1)
            realData = Variable(trainVector[leftIndex:leftIndex + batchSize_D])
            tu = Variable(trainVector[leftIndex:leftIndex + batchSize_G])

            KRNNeighbors = Variable(copy.deepcopy(KRNN[leftIndex:leftIndex + batchSize_G]))


            ns_items = NS[leftIndex:leftIndex + batchSize_G]
            ns_items = ns_items.astype(numpy.float32)
            ns_u = Variable(torch.tensor(ns_items))

            # Generate a batch of new purchased vector
            G_prediction = G(realData, KRNNeighbors)
            # G_prediction = G(realData)
            G_prediction_ZP = G_prediction * (tu + ns_u)

            # Train the discriminator
            G_prediction_result = D(G_prediction_ZP)
            realData_result = D(realData)
            d_loss = -np.mean(np.log(realData_result.detach().numpy() + 10e-5)
                                    + np.log(1. - G_prediction_result.detach().numpy() + 10e-5))  \
                                    + 0 * regularization(G_prediction_ZP, realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 1 == 0):
            for topN in range(len(TopN)):
                index = 0
                precisions = 0
                P, R, NDCG, MAP, MRR = [], [], [], [], []
                for testUser in testSet.keys():
                    data = Variable(trainVector[testUser])
                    # #  Exclude the purchased vector that have occurred in the training set
                    neighbors = Variable(KRNN[index])
                    # result = G(data.reshape(1,itemCount)) + Variable(copy.deepcopy(trainMaskVector[index]))
                    result = G(data.reshape(1, itemCount), neighbors.reshape(1, userCount)) \
                             + Variable(trainMaskVector[index])
                    result = result.reshape(itemCount)

                    p, r, ndcg, map, mrr = Utils.P_R_N_AP_RR(testSet[testUser], result, TopN[topN])
                    P.append(p)
                    R.append(r)
                    NDCG.append(ndcg)
                    MAP.append(map)
                    MRR.append(mrr)
                    index += 1

                result_quotas[epoch, 5 * topN + 0] = np.mean(P)
                result_quotas[epoch, 5 * topN + 1] = np.mean(R)
                result_quotas[epoch, 5 * topN + 2] = np.mean(NDCG)
                result_quotas[epoch, 5 * topN + 3] = np.mean(MAP)
                result_quotas[epoch, 5 * topN + 4] = np.mean(MRR)

            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
                                                                                 d_loss.item(),
                                                                                 g_loss.item(),
                                                                                 np.mean(P)), precisions)

    return result_quotas


if __name__ == '__main__':

    dataset = 'ml100k'
    sep = '\t'

    k_reciprocal_num = 50
    k_nearest_num = 100
    # K_nearest_num = [20, 50, 100, 150, 200]
    # for k_nearest_num in K_nearest_num:
    TopN = [5, 10, 20]
    epochs = 1000

    trainSet, train_use, train_item, interact_matrix, rating_matrix = Utils.loadTrainingData_modify(
        "data/" + dataset + "/" + dataset + ".train", sep)
    testSet, test_use, test_item = Utils.loadTestData("data/" + dataset + "/" + dataset + ".test", sep)
    userCount = max(train_use, test_use)
    itemCount = max(train_item, test_item)
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount = Utils.to_Vectors_trueRate(trainSet, userCount, \
                                                                         itemCount, userList_test,
                                                                         rating_matrix)

    K2 = 100
    RKNN_u = Utils.Neighbours.Get_NS_RKFN_U(rating_matrix, interact_matrix, K2)
    RKNN_i = Utils.Neighbours.Get_NS_RKNN_I(rating_matrix.T, interact_matrix.T, K2)
    NS_i = Utils.Neighbours.get_NS_u(RKNN_i, rating_matrix)
    NS_u = Utils.Neighbours.get_NS_u(RKNN_u, rating_matrix)
    NS = Utils.Neighbours.get_NS(RKNN_u, RKNN_i, rating_matrix)

    # k_nearest_num, k_reciprocal_num = 50, 50
    KRNN = Utils.Neighbours.select_KRNN(rating_matrix, interact_matrix, k_nearest_num, k_reciprocal_num)

    result_quotas = main(userCount, itemCount, testSet, trainVector, trainMaskVector,
                         KRNN, NS, TopN, epochs)

    pd.DataFrame(result_quotas).to_csv('../result/NS.txt')



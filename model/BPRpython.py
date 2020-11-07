# Implement BPR.
# Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback.
# Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI, 2009. 
# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
#import scores

class BPR(object):
    '''
    parameter
    train_sample_size : 訓練時，每個正樣本，我sample多少負樣本
    test_sample_size : 測試時，每個正樣本，我sample多少負樣本
    num_k : item embedding的維度大小
    evaluation_at : recall@多少，及正樣本要排前幾名，我們才視為推薦正確
    '''
    def __init__(self,model_path,node_u_num,node_v_num,vectors_u,vectors_v,dim,n_train,n_user,train_item,n_epochs=10,batch_size=512):
        self.user_count = node_u_num
        #print(self.user_count)
        self.item_count = node_v_num
        self.model_path=model_path
        #latent_factors = 20
        self.lr = 0.01
        self.reg = 0.01
        self.train_count = 1000
        self.train_data_path = os.path.join(self.model_path,"ratings_train.dat")
        self.test_data_path = os.path.join(self.model_path,"ratings_test.dat")
        self.size_u_i = self.user_count * self.item_count
    # latent_factors of U & V
        self.U = vectors_u
        self.V = vectors_v
        self.biasV = np.random.rand(self.item_count) * 0.01
        self.test_data = np.zeros((self.user_count, self.item_count))
        self.test = np.zeros(self.size_u_i)
        predict_ = np.zeros(self.size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i, rate = line.strip().split("\t")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.strip().split("\t")
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
   
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)[0]
            # sample a negative item from the unobserved items
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            a=u
            b=i
            c=j
            
            a-=1
            b-=1
            c-=1
            
            r_ui = np.dot(self.U[str(u)], self.V[str(i)].T) + self.biasV[b]
            r_uj = np.dot(self.U[str(u)], self.V[str(j)].T) + self.biasV[c]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            
            self.U[str(u)].setflags(write=1)
            self.V[str(i)].setflags(write=1)
            self.V[str(j)].setflags(write=1) 
            
            #update U and V
            self.U[str(u)] += -self.lr * (loss_func * (self.V[str(i)] - self.V[str(j)]) + self.reg * self.U[str(u)])
            self.V[str(i)] += -self.lr * (loss_func * self.U[str(u)] + self.reg * self.V[str(i)])
            self.V[str(j)] += -self.lr * (loss_func * (-self.U[str(u)]) + self.reg * self.V[str(j)])
            #update biasV
            self.biasV[b] += -self.lr * (loss_func + self.reg * self.biasV[b])
            self.biasV[c] += -self.lr * (-loss_func + self.reg * self.biasV[c])

    def predict(self, user, item):
        #predict = np.mat(user) * np.mat(item.T)
        predict = user * (item.T)
        return predict

    def fit(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # training
        for i in range(self.train_count):
            self.train(user_ratings_train)
        return self.U,self.V
        #predict_matrix = self.predict(self.U, self.V)
        ## prediction
        #self.predict_ = predict_matrix.getA().reshape(-1)
        #self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        #auc_score = roc_auc_score(self.test, self.predict_)
        #print('AUC:', auc_score)
        # Top-K evaluation
        #scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict



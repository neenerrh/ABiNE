# Implement BPR.
# Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback.
# Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI, 2009. 
# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
#import scores
import sys

class BPR(object):
    '''
    parameter
    train_sample_size : 訓練時，每個正樣本，我sample多少負樣本
    test_sample_size : 測試時，每個正樣本，我sample多少負樣本
    num_k : item embedding的維度大小
    evaluation_at : recall@多少，及正樣本要排前幾名，我們才視為推薦正確
    '''
    def __init__(self,model_path,node_u_num,node_v_num,vectors_u,vectors_v,users,items,users_list,items_list,n_train,train_user,train_item,dim,lam):
        self.user_count = len(users_list)
        #print(self.user_count)
        self.item_count = len(items_list)
        self.user_embed= node_u_num
        self.item_embed= node_v_num
        #latent_factors = 20
        self.n_epochs=10
        self.batch_size=512
        self.lr = lam  #learning rate
        self.train_users=users
        self.reg = 0.01
        self.train_count = 500
        self.train_data_path = '../data/mooc/ratings_train.dat'
        self.test_data_path = '../data/mooc/ratings_test.dat'
        self.size_u_i = self.user_count * self.item_count
    # latent_factors of U & V
        self.U = vectors_u
        self.V = vectors_v
        self.biasV = np.random.rand(self.item_count) * 0.01
        self.test_data = np.zeros((self.user_count, self.item_count))
        self.test = np.zeros(self.size_u_i)
        predict_ = np.zeros(self.size_u_i)
        #self.users=users
        #self.items=items
        self.users_list=users_list
        self.items_list=items_list

    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i, rate = line.strip().split("\t")
              
                
                user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.strip().split("\t")
            user = line[0]
            item = line[1]
            self.test_data[user][item] = line[2]

    def train(self, user_ratings_train):
        
        #print(user_ratings_train.keys())
        for user in range(self.user_count):
            # sample a user
            #u = random.randint(1, self.user_count)
            
            u = random.sample(self.users_list,1)
            #print(f"u {u} ")
            u=u[0]
            if str(u) not in user_ratings_train.keys():
                #print(f"string u {str(u)} ")             
                continue
            if str(u) not in self.train_users:
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)
            i=i[0]
            #z = random.sample(user_ratings_train[u], 1)
            # sample a negative item from the unobserved items
            j = random.sample(self.items_list, 1)
            j=j[0]
            while j in user_ratings_train[u]:
                j = random.sample(self.items_list, 1)
                j=j[0]
           
            
            
            self.U[str(u)].setflags(write=1)
            self.V[str(i)].setflags(write=1)
            self.V[str(j)].setflags(write=1) 
            
            
            
            #r_ui = np.dot(self.U[str(u)], self.V[str(i)].T) + self.biasV[str(i)]
            #r_uj = np.dot(self.U[str(u)], self.V[str(j)].T) + self.biasV[str(j)]
            #r_uij = r_ui - r_uj
       
          
            
            #Method2
            self.r_ui = np.dot(self.U[str(u)], self.V[str(i)].T)
            self.r_uj = np.dot(self.U[str(u)], self.V[str(j)].T) 
            self.r_uij =self. r_ui - self.r_uj
            self.sigmoid = np.exp(-self.r_uij) / (1.0 + np.exp(-self.r_uij))        
            # update using gradient descent
            self.grad_u = self.sigmoid * (self.V[str(i)] - self.V[str(j)]) + self.reg * self.U[str(u)]
            self.grad_i = self.sigmoid * -self.U[str(u)] + self.reg * self.V[str(i)]
            self.grad_j = self.sigmoid * self.U[str(u)] + self.reg * self.V[str(j)]
            self.U[str(u)] -= self.lr * self.grad_u
            self.V[str(i)] -= self.lr * self.grad_i
            self.V[str(j)] -= self.lr * self.grad_j
        
    def predict(self, user, item):
        #predict = np.mat(user) * np.mat(item.T)
        predict = user * (item.T)
        return predict

    def fit(self):
        print(self.lr)        
        print(self.user_count)
        print(self.user_embed)
        print(self.item_count)
        print(self.item_embed)
        user_ratings_train = self.load_data(self.train_data_path)
        #self.biasV=dict(zip(self.items_list,self.V))
      
        #self.load_test_data(self.test_data_path)
        #self.load_test_data(self.test_data_path)
        #for u in range(self.user_count):
            #for item in range(self.item_count):
                #if int(self.test_data[u][item]) == 1:
                    #self.test[u * self.item_count + item] = 1
                #else:
                    #self.test[u * self.item_count + item] = 0
        last_loss, count, epsilon = 0, 0, 1e-3
        for i in range(self.train_count):
            s1 = "\r[%s%s]%0.2f%%"%("*"* i," "*(self.train_count-i),i*100.0/(self.train_count-1))
            self.train(user_ratings_train)
            delta_loss = abs(self.sigmoid - last_loss)
            if last_loss > self.sigmoid:
                self.lr *= 1.05
            else:
                self.lr *= 0.95
            last_loss = self.sigmoid
            if delta_loss < epsilon:
                break
            sys.stdout.write(s1)
            sys.stdout.flush()           
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



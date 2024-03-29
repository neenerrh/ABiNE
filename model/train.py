#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
from scipy import linalg
import decimal
from numpy import float128
from sklearn import preprocessing
from data_utils import DataUtils
from graph_utils import GraphUtils
import random
import math
import os
import sys
sys_path=sys.path.append('C:/Users/Administrator/Desktop/New_folder/experiment/Bine')
import pandas as pd
from sklearn import metrics
from one_mode import OneMode
from BPRpython import BPR
from functools import cmp_to_key
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
import time
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.classify import ncClassifier, lpClassifier, read_node_label
from libnrl.graph import *
from libnrl.utils import *
from libnrl import abrw #ANE method; Attributed Biased Random Walk
from libnrl import tadw #ANE method
from libnrl import aane #ANE method
from libnrl import attrcomb #ANE method
from libnrl import attrpure #NE method simply use svd or pca for dim reduction
from libnrl import node2vec #PNE method; including deepwalk and node2vec



def init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args):
    # """
    # initialize embedding vectors
    # :param node_u:
    # :param node_v:
    # :param node_list_u:
    # :param node_list_v:
    # :param args:
    # :return:
    # """
    #user
    for i in node_u:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_u[i] = {}
        node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')
    
    for i in items_list:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_v[i] = {}
        node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')



def walk_generator(gul,args):
    """
    walk generator
    :param gul:
    :param args:
    :return:
    """
    #gul.calculate_centrality(args.mode)
    if args.large == 0:
        gul.homogeneous_graph_random_walks(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage=args.p, maxT=args.maxT, minT=args.minT)
        
    elif args.large == 2:
                gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile=args.train_data,percentage=args.p,maxT=args.maxT, minT=args.minT)
    return gul

def get_negative_samples(gul,args):
    # :param gul:
    # :param args:
    # :return:  neg_dict_u, neg_dict_v,gul.node_u,gul.node_v
    # """
    if args.large == 0:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        #print len(gul.walks_u),len(gul.walks_u)
        print("negative samples is ok.....")
        
    return neg_dict_u, neg_dict_v,gul.node_u,gul.node_v

def get_context_and_negative_samples2(gul, walks_u, walks_v, g_u, g_v,args):
    """
    get context and negative samples offline
    :param gul:
    :param args:
    :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
    """
    if args.large == 0:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(g_u, walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(g_v, walks_v, args.ws, args.ns, neg_dict_v)
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        # print len(gul.walks_u),len(gul.walks_u)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns, neg_dict_v)

    return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
   
def skip_gram2(center, contexts, negs, node_list, lam, pa):
    """
    skip-gram
    :param center:
    :param contexts:
    :param negs:
    :param node_list:
    :param lam:
    :param pa:
    :return:
    """
    loss = 0
    I_z = {center: 1}  # indication function
    for node in negs:
        I_z[node] = 0
    V = np.array(node_list[contexts]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is  None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(V.dot(Theta.T))
        #print(X)
        sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
        update += pa * lam * (I_z[u] - sigmod) * Theta
        node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
        
        
        
        try:
            loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
        except:
            pass
            # print "skip_gram:",
            # print(V,Theta,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update, loss


def get_context_and_negative_samples(gul,walks_u,walks_v,g_u,g_v, args):
    # """
    # get context and negative samples offline
    # :param gul:
    # :param args:
    # :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
    # """

    if args.large == 0:
        #neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        context_dict_u = gul.get_context(g_u, walks_u, args.ws)
        context_dict_v = gul.get_context(g_v, walks_v, args.ws)
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        #print len(gul.walks_u),len(gul.walks_u)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns, neg_dict_v)

    return context_dict_u, context_dict_v,gul.node_u,gul.node_v


def skip_gram(center, contexts, node_list, lam, pa):
    # """
    # skip-gram
    # :param center:
    # :param contexts:
    # :param negs:
    # :param node_list:
    # :param lam:
    # :param pa:
    # :return:
    # """
    loss = 0
    I_z = {center: 1}  # indication function
    #for node in negs:
        #I_z[node] = 0
    V = np.array(node_list[contexts]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is  None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(V.dot(Theta.T))
        sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
        update += pa * lam * (I_z[u] - sigmod) * Theta
        node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
        try:
            loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
        except:
            pass
            ##print "skip_gram:",
            ##print(V,Theta,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update, loss


def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param node_list_u:
    :param node_list_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    e_ij = edge_dict_u[u][v]

    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X = float(U.dot(V.T))

    sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))

    update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * V
    update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * U

    try:
        loss += gamma * e_ij * math.log(sigmod)
    except:
        pass
        # print "KL:",
        # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update_u, update_v, loss
    
def Log_Like(u,i,j ,node_list_u, node_list_v, lam, reg,gamma,biasV):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param vectors_u:
    :param vectors_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    #e_ij = edge_dict_u[u][v]

    update_u = 0
    update_i = 0
    update_j =0
    biasV_i =0
    biasV_j =0
    U = np.array(node_list_u[u]['embedding_vectors'] )  
    I = np.array(node_list_v[i]['embedding_vectors'] )
    J = np.array(node_list_v[j]['embedding_vectors'] )
    r_ui = float(U.dot(I.T)) + biasV[i]
    r_uj = float(U.dot(J.T)) + biasV[j]
    r_uij= r_ui-r_uj

    sigmoid=1.0 / (1 + np.exp(-r_uij))
    #r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
    #r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
    #r_uij = r_ui - r_uj
    loss_func = -1.0 / (1 + np.exp(r_uij))
            # update U and V
    update_u += -lam *gamma* (loss_func * (node_list_v[i]['embedding_vectors'] - node_list_v[j]['embedding_vectors']) + reg * node_list_u[u]['embedding_vectors'])
    update_i += -lam *gamma*  (loss_func * node_list_u[u]['embedding_vectors'] + reg * node_list_v[i]['embedding_vectors'])
    update_j += -lam *gamma*  (loss_func * (-node_list_u[u]['embedding_vectors']) + reg * node_list_v[j]['embedding_vectors'])
    #update biasV
    biasV_i += -lam *gamma*  (loss_func + reg * biasV[str(i)])
    biasV_j += -lam *gamma*  (-loss_func + reg * biasV[str(j)]) 

    try:
        loss += gamma * math.log(sigmod)
    except:
        pass
        # print "KL:",
        # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))

    return update_u, update_i,update_j, loss , biasV_i,biasV_j

def Log_Like2(u,i,j ,node_list_u, node_list_v, lam, reg,gamma):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param vectors_u:
    :param vectors_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    #e_ij = edge_dict_u[u][v]

    update_u = 0
    update_i = 0
    update_j =0
    #biasV_i =0
    #biasV_j =0
    U = np.array(node_list_u[u]['embedding_vectors'] )  
    I = np.array(node_list_v[i]['embedding_vectors'] )
    J = np.array(node_list_v[j]['embedding_vectors'] )
    r_ui = float(U.dot(I.T))
    r_uj = float(U.dot(J.T)) 
    r_uij= r_ui-r_uj
    loss_func = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
    #loss_func= expit(r_uij)
    #r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
    #r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
    #r_uij = r_ui - r_uj
    sigmod = 1.0 / (1 + np.exp(-r_uij))

    
            # update U and V
  
    update_u -= gamma*lam * (loss_func * (node_list_v[j]['embedding_vectors'] - node_list_v[i]['embedding_vectors']) - reg * node_list_u[u]['embedding_vectors'])
    update_i -= gamma*lam * (loss_func * (-node_list_u[u]['embedding_vectors']) - reg * node_list_v[i]['embedding_vectors'])
    update_j -= gamma*lam * (loss_func * (node_list_u[u]['embedding_vectors']) - reg * node_list_v[j]['embedding_vectors'])
    #update biasV
    #biasV_i += -lam *gamma* (loss_func + reg * biasV[str(i)])
    #biasV_j += -lam  * gamma*(-loss_func + reg * biasV[str(j)])
    
    try:
        loss += gamma * (math.log(sigmod) )
    except:
        pass
        # print "KL:",
        # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
          
    return update_u, update_i,update_j, loss
def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]['embedding_vectors'])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]['embedding_vectors'])
                    pre = U.dot(V.T)[0][0]
            recommend_dict[u][v] = float(pre)


    
    
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:
        
        
        from functools import cmp_to_key
        def cmp(x, y):                   # emulate cmp from Python 2
            if (x< y):
                return -1
            elif (x == y):
                return 0
            elif (x > y):
                return 1
        tmp_r = sorted(recommend_dict[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[0:min(len(recommend_dict[u]),top_n)]
      
        
        tmp_t = sorted(test_rate[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[0:min(len(test_rate[u]),top_n)]
              
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        pre, rec = precision_and_recall(tmp_r_list,tmp_t_list)
        ap = AP(tmp_r_list,tmp_t_list)
        rr = RR(tmp_r_list,tmp_t_list)
        ndcg = nDCG(tmp_r_list,tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    #print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1,map,mrr,mndcg,recall

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def generateFeatureFile(filecase,filevector_u,filevector_v,fileout,factors):
    vectors_u = {}
    vectors_v = {}
    with open(filevector_u,'r') as fu:
        for line in fu.readlines():
            items = line.strip().split(' ')
            vectors_u[items[0]] = items[1:]
    with open(filevector_v,'r') as fv:
        for line in fv.readlines():
            items = line.strip().split(' ')
            vectors_v[items[0]] = items[1:]
    with open(filecase,'r') as fc, open(fileout,'w') as fo:
        for line in fc.readlines():
            items = line.strip().split('\t')
            if vectors_u.get(items[0]) == None:
                vectors_u[items[0]] = ['0'] * factors
            if vectors_v.get(items[1]) == None:
                vectors_v[items[1]] = ['0'] * factors
            if items[-1] == '1':
                fo.write('{}\t{}\t{}\n'.format('\t'.join(vectors_u[items[0]]),'\t'.join(vectors_v[items[1]]),1))
            else:
                fo.write('{}\t{}\t{}\n'.format('\t'.join(vectors_u[items[0]]),'\t'.join(vectors_v[items[1]]),0))

def link_prediction(args):
    filecase_a = args.case_train
    filecase_e = args.case_test
    filevector_u = args.vectors_u
    filevector_v = args.vectors_v
    filecase_a_c = r'../data/features_train.dat'
    filecase_e_c = r'../data/features_test.dat'
    generateFeatureFile(filecase_a,filevector_u,filevector_v,filecase_a_c,args.d)
    generateFeatureFile(filecase_e,filevector_u,filevector_v,filecase_e_c,args.d)

    df_data_train = pd.read_csv(filecase_a_c,header = None,sep='\t',encoding='utf-8')
    X_train = df_data_train.drop(len(df_data_train.keys())-1,axis = 1)
    y_train = df_data_train[len(df_data_train.keys())-1]

    df_data_test = pd.read_csv(filecase_e_c,header = None,sep='\t',encoding='utf-8')
    X_test = df_data_test.drop(len(df_data_train.keys())-1,axis = 1)
    X_test = X_test.fillna(X_test.mean())
    y_test = df_data_test[len(df_data_test.keys())-1]
    y_test_list = list(y_test)

    lg = LogisticRegression(penalty='l2',C=0.001)
    lg.fit(X_train,y_train)
    lg_y_pred_est = lg.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(y_test,lg_y_pred_est)
    average_precision = average_precision_score(y_test, lg_y_pred_est)
    os.remove(filecase_a_c)
    os.remove(filecase_e_c)
    return metrics.auc(fpr,tpr), average_precision

def train_by_sampling(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam,reg = args.alpha, args.beta, args.gamma, args.lam,args.lr
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    #model_path1=os.path.join('../content/ABiNE/', args.model_name)
    #datafile= os.path.join(model_path,"ratings.dat")
    
    dul = DataUtils(model_path)
    #data = pd.read_csv('/content/ABiNE/data/mooc/months1/march_users1.csv')
    #data = data.rename(columns={'user  ': 'user', 'item ':'item','label ':'label'})
   
    #train_data,test_data= dul.split_by_ratio(data, test_size=args.testRatio)
    #train_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_train.csv',index=False,header=False)
    ##files.download('/content/new_BiNE/data/mooc/months1/ratings_train.csv')
    #test_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_test.csv',index=False,header=False)
    
    train_user,train_item,train_rate=dul.read_train_data(args.train_data)
    test_user, test_item, test_rate = dul.read_test_data(args.test_data)
    n_train=len(train_item) 
    items=list(train_item) + list(test_item)
    res2 = [] 
    [res2.append(x) for x in items if x not in res2]
    items=res2 
    users_list=list(train_user) + list(test_user)
    res3 = [] 
    [res3.append(x) for x in users_list if x not in res3]
    users_list=res3   

    print(len(train_item))
    print(len(test_item)) 
    print(len(items)) 
    user_ratings_train,items= dul.load_data(args.train_data)
 
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    
    walk_generator(gul,args)
    model_path1 = os.path.join('../', 'model')    
    one=OneMode(model_path)
    
    one.u_load_data(args.u_graph_file,args.weighted,args.directed,args.graph_format,args.u_attribute_file,args.method)
    one.v_load_data(args.v_graph_file,args.weighted,args.directed,args.graph_format,args.v_attribute_file,args.method)
    
    if args.uattr == True:
        one.u_load_attr(args.u_attribute_file,args.method)   
        
    if args.vattr == True: 
        one.v_load_attr(args.v_attribute_file,args.method)
    
    walks_u,g_u=one.u_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.u_emb_file,args.uattr)
    walks_v,g_v=one.v_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.v_emb_file,args.vattr)
    print(g_v.G.nodes())
    items_list=g_v.G.nodes()
    print("getting context and negative samples....")
    context_dict_u, context_dict_v, node_u, node_v = get_context_and_negative_samples(gul,walks_u,walks_v,g_u,g_v, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args)
    last_loss, count, epsilon  = 0, 0, 0.0001
    print(len(node_list_u.keys()))
    print(len(users_list))
    print(len(node_list_v.keys()))
    #biasV = np.random.rand(len(items_list)) * 0.01
    #biasV=dict(zip(items_list,biasV))
    print("============== training ==============")
    for iter in range(0, args.max_iter):
       
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        
       
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        

        #random.shuffle(edge_list)
        for user in range(len(users_list)):
            u = random.sample(users_list,1)
           
            u=u[0]

           
            #if str(u) not in user_ratings_train.keys():
                #print(f"string u {str(u)} ")             
                #continue
            if str(u) not in train_user:
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)
            i=i[0]
            
            #z = random.sample(user_ratings_train[u], 1)
            # sample a negative item from the unobserved items
            j = random.sample(items_list, 1)
            j=j[0]
            while j in user_ratings_train[u]:
                j = random.sample(items_list, 1)
                j=j[0]
             
              
             
            

            length = len(context_dict_u[u])
            random.shuffle(context_dict_u[u])
            if visited_u.get(u) < length:
                # print(u)
                index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                   
                    # center,context,neg,node_list,eta
                    for z in context_u:                       
                        tmp_z, tmp_loss = skip_gram(u, z, node_list_u, lam, alpha)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = index_list[-1]+3

            length = len(context_dict_v[i])
            random.shuffle(context_dict_v[i])
            if visited_v.get(i) < length:
                # print(v)
                index_list = list(range(visited_v.get(i),min(visited_v.get(i)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[i][index]
                   
                    # center,context,neg,node_list,eta

                    for z in context_v:                       
                                           
                        tmp_z, tmp_loss = skip_gram(i, z, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[i] = index_list[-1]+3

            length = len(context_dict_v[j])
            random.shuffle(context_dict_v[j])
            if visited_v.get(j) < length:
                # print(v)
                index_list = list(range(visited_v.get(j),min(visited_v.get(j)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[j][index]                   
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        
                        tmp_z, tmp_loss = skip_gram(j, z, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        #loss += tmp_loss
                        loss = tmp_loss
                visited_v[j] = index_list[-1]+3
              

            # r_ui = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[i]['embedding_vectors']).T)
            # r_uj = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[j]['embedding_vectors']).T)
            # r_uij =r_ui - r_uj
            # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
            ##sigmoid_tiled = np.tile(sigmoid, (128, 1)).T        
            ##update using gradient descent
            # grad_u = sigmoid * (node_list_v[i]['embedding_vectors'] - node_list_v[j]['embedding_vectors']) + reg * node_list_u[u]['embedding_vectors']
            # grad_i = sigmoid * -node_list_u[u]['embedding_vectors'] + reg * node_list_v[i]['embedding_vectors']
            # grad_j = sigmoid * node_list_u[u]['embedding_vectors'] + reg * node_list_v[j]['embedding_vectors']
            # node_list_u[u]['embedding_vectors'] -= lam * grad_u
            # node_list_v[i]['embedding_vectors'] -= lam * grad_i
            # node_list_v[j]['embedding_vectors'] -= lam * grad_j
           
            #update_u, update_i,update_j, tmp_loss,biasV_i,biasV_j = Log_Like( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma,biasV)
            update_u, update_i,update_j, tmp_loss= Log_Like2( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[i]['embedding_vectors'] += update_i
            node_list_v[j]['embedding_vectors'] += update_j
            #biasV[i]+= biasV_i
            #biasV[j]+= biasV_j
            
           
        delta_loss = abs(loss - last_loss)
        #if last_loss > loss:
            #lam *= 1.05
        #else:
            #lam *= 0.95
        #print(loss)
       
        last_loss = loss
        print(last_loss)
        #if last_loss < -1:
            #break
        sys.stdout.write(s1)
        sys.stdout.flush()
    #print("============== training ==============")
    
    #if args.loss_function == 0 :
        #pointwise(vectors_u,vectors_v, edge_list, edge_dict_u,args.max_iter,alpha, beta, gamma, lam)
        
    #else :
        #bpr=BPR(model_path,node_u_num,node_v_num,vectors_u,vectors_v,users,items,users_list,items_list,n_train,train_user,train_item,args.dim,args.lam) 
        #vectors_u,vectors_v=bpr.fit()
        
     
    #save_to_file(vectors_u,vectors_v,model_path,args) 
    
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg,recall = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))
        
def train_by_neg(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam,reg = args.alpha, args.beta, args.gamma, args.lam,args.lr
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    #model_path1=os.path.join('../content/ABiNE/', args.model_name)
    #datafile= os.path.join(model_path,"ratings.dat")
    
    dul = DataUtils(model_path)
    #data = pd.read_csv('/content/ABiNE/data/mooc/months1/march_users1.csv')
    #data = data.rename(columns={'user  ': 'user', 'item ':'item','label ':'label'})
   
    #train_data,test_data= dul.split_by_ratio(data, test_size=args.testRatio)
    #train_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_train.csv',index=False,header=False)
    ##files.download('/content/new_BiNE/data/mooc/months1/ratings_train.csv')
    #test_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_test.csv',index=False,header=False)
    
    train_user,train_item,train_rate=dul.read_train_data(args.train_data)
    test_user, test_item, test_rate = dul.read_test_data(args.test_data)
    n_train=len(train_item) 
    items=list(train_item) + list(test_item)
    res2 = [] 
    [res2.append(x) for x in items if x not in res2]
    items=res2 
    users_list=list(train_user) + list(test_user)
    res3 = [] 
    [res3.append(x) for x in users_list if x not in res3]
    users_list=res3   

    print(len(train_item))
    print(len(test_item)) 
    print(len(items)) 
    user_ratings_train,items= dul.load_data(args.train_data)
 
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    gul.construct_test_graph(args.test_data)
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    
    walk_generator(gul,args)
    model_path1 = os.path.join('../', 'model')    
    one=OneMode(model_path)
    
    one.u_load_data(args.u_graph_file,args.weighted,args.directed,args.graph_format,args.u_attribute_file,args.method)
    one.v_load_data(args.v_graph_file,args.weighted,args.directed,args.graph_format,args.v_attribute_file,args.method)
    
    if args.uattr == True:
        one.u_load_attr(args.u_attribute_file,args.method)   
        
    if args.vattr == True: 
        one.v_load_attr(args.v_attribute_file,args.method)
    
    walks_u,g_u=one.u_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.u_emb_file,args.uattr)
    walks_v,g_v=one.v_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.v_emb_file,args.vattr)
    print(g_v.G.nodes())
    items_list=g_v.G.nodes()
    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v= get_context_and_negative_samples2(gul,walks_u,walks_v,g_u,g_v, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args)
    last_loss, count, epsilon  = 0, 0, 0.0001
    print(len(node_list_u.keys()))
    print(len(users_list))
    print(len(node_list_v.keys()))
    #biasV = np.random.rand(len(items_list)) * 0.01
    #biasV=dict(zip(items_list,biasV))
    t0 = time.time()
    print("============== training ==============")
    for iter in range(0, args.max_iter):
       
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        
       
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        

        #random.shuffle(edge_list)
        for user in range(len(users_list)):
            u = random.sample(users_list,1)
           
            u=u[0]

           
            #if str(u) not in user_ratings_train.keys():
                #print(f"string u {str(u)} ")             
                #continue
            if str(u) not in train_user:
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)
            i=i[0]
            
            #z = random.sample(user_ratings_train[u], 1)
            # sample a negative item from the unobserved items
            j = random.sample(items_list, 1)
            j=j[0]
            while j in user_ratings_train[u]:
                j = random.sample(items_list, 1)
                j=j[0]
             
              
             
            

            length = len(context_dict_u[u])
            random.shuffle(context_dict_u[u])
            if visited_u.get(u) < length:
                # print(u)
                index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for z in context_u:                       
                        tmp_z, tmp_loss = skip_gram2(u, z, neg_u, node_list_u, lam, beta)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = index_list[-1]+3

            length = len(context_dict_v[i])
            random.shuffle(context_dict_v[i])
            if visited_v.get(i) < length:
                # print(v)
                index_list = list(range(visited_v.get(i),min(visited_v.get(i)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[i][index]
                    neg_v = neg_dict_v[i][index]
                    # center,context,neg,node_list,eta

                    for z in context_v:                       
                                           
                        tmp_z, tmp_loss = skip_gram2(i, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[i] = index_list[-1]+3

            length = len(context_dict_v[j])
            random.shuffle(context_dict_v[j])
            if visited_v.get(j) < length:
                # print(v)
                index_list = list(range(visited_v.get(j),min(visited_v.get(j)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[j][index]
                    neg_v = neg_dict_v[j][index]                   
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        
                        tmp_z, tmp_loss = skip_gram2(j, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        #loss += tmp_loss
                        loss = tmp_loss
                visited_v[j] = index_list[-1]+3
              

            # r_ui = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[i]['embedding_vectors']).T)
            # r_uj = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[j]['embedding_vectors']).T)
            # r_uij =r_ui - r_uj
            # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
            ##sigmoid_tiled = np.tile(sigmoid, (128, 1)).T        
            ##update using gradient descent
            # grad_u = sigmoid * (node_list_v[i]['embedding_vectors'] - node_list_v[j]['embedding_vectors']) + reg * node_list_u[u]['embedding_vectors']
            # grad_i = sigmoid * -node_list_u[u]['embedding_vectors'] + reg * node_list_v[i]['embedding_vectors']
            # grad_j = sigmoid * node_list_u[u]['embedding_vectors'] + reg * node_list_v[j]['embedding_vectors']
            # node_list_u[u]['embedding_vectors'] -= lam * grad_u
            # node_list_v[i]['embedding_vectors'] -= lam * grad_i
            # node_list_v[j]['embedding_vectors'] -= lam * grad_j
           
            #update_u, update_i,update_j, tmp_loss,biasV_i,biasV_j = Log_Like( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma,biasV)
            update_u, update_i,update_j, tmp_loss= Log_Like2( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[i]['embedding_vectors'] += update_i
            node_list_v[j]['embedding_vectors'] += update_j
            #biasV[i]+= biasV_i
            #biasV[j]+= biasV_j
            
           
        delta_loss = abs(loss - last_loss)
        #if last_loss > loss:
            #lam *= 1.05
        #else:
            #lam *= 0.95
        #print(loss)
       
        last_loss = loss
        print(last_loss)
        #if last_loss < -1:
            #break
        sys.stdout.write(s1)
        sys.stdout.flush()
    #print("============== training ==============")
    
    #if args.loss_function == 0 :
        #pointwise(vectors_u,vectors_v, edge_list, edge_dict_u,args.max_iter,alpha, beta, gamma, lam)
        
    #else :
        #bpr=BPR(model_path,node_u_num,node_v_num,vectors_u,vectors_v,users,items,users_list,items_list,n_train,train_user,train_item,args.dim,args.lam) 
        #vectors_u,vectors_v=bpr.fit()
        
     
    #save_to_file(vectors_u,vectors_v,model_path,args) 
    t1 = time.time()
    print(f'Training done: end of training ; time cost: {(t1-t0):.2f}s')  
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg,recall = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))


def train_by_batch(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam,reg = args.alpha, args.beta, args.gamma, args.lam,args.lr
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    #model_path1=os.path.join('../content/ABiNE/', args.model_name)
    #datafile= os.path.join(model_path,"ratings.dat")
    
    dul = DataUtils(model_path)
    #data = pd.read_csv('/content/ABiNE/data/mooc/months1/march_users1.csv')
    #data = data.rename(columns={'user  ': 'user', 'item ':'item','label ':'label'})
   
    #train_data,test_data= dul.split_by_ratio(data, test_size=args.testRatio)
    #train_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_train.csv',index=False,header=False)
    ##files.download('/content/new_BiNE/data/mooc/months1/ratings_train.csv')
    #test_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_test.csv',index=False,header=False)
    
    train_user,train_item,train_rate=dul.read_train_data(args.train_data)
    test_user, test_item, test_rate = dul.read_test_data(args.test_data)
    n_train=len(train_item) 
    items=list(train_item) + list(test_item)
    res2 = [] 
    [res2.append(x) for x in items if x not in res2]
    items=res2 
    users_list=list(train_user) + list(test_user)
    res3 = [] 
    [res3.append(x) for x in users_list if x not in res3]
    users_list=res3   

    print(len(train_item))
    print(len(test_item)) 
    print(len(items)) 
    user_ratings_train,items= dul.load_data(args.train_data)
 
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
   
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    
    walk_generator(gul,args)
    model_path1 = os.path.join('../', 'model')    
    one=OneMode(model_path)
    
    one.u_load_data(args.u_graph_file,args.weighted,args.directed,args.graph_format,args.u_attribute_file,args.method)
    one.v_load_data(args.v_graph_file,args.weighted,args.directed,args.graph_format,args.v_attribute_file,args.method)
    
    if args.uattr == True:
        one.u_load_attr(args.u_attribute_file,args.method)   
        
    if args.vattr == True: 
        one.v_load_attr(args.v_attribute_file,args.method)
    
    walks_u,g_u=one.u_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.u_emb_file,args.uattr)
    walks_v,g_v=one.v_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.v_emb_file,args.vattr)   
    
    print(g_v.G.nodes())
    items_list=g_v.G.nodes()
    
    
    print("getting context and negative samples....")
    context_dict_u, context_dict_v, node_u, node_v = get_context_and_negative_samples(gul,walks_u,walks_v,g_u,g_v, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args)
    last_loss, count, epsilon  = 0, 0, 0.0001
    print(len(node_list_u.keys()))
    print(len(users_list))
    print(len(node_list_v.keys()))
    t0 = time.time()
    print("============== training ==============")
    #biasV = np.random.rand(len(items_list)) * 0.01
    #biasV=dict(zip(items_list,biasV))
    
    batch_size = args.batch_size
    if len(users_list) < batch_size:
            batch_size = len(users_list)
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(len(users_list)))

    batch_iters = len(users_list) // batch_size
        
        
        
        
    for iter in range(0, args.max_iter):
            s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
            loss = 0
            for _ in range(batch_iters):
               
        
       
                visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
                visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        

                #random.shuffle(edge_list)
                for _ in range(batch_size):
                    u = np.random.choice(users_list,1,replace=False) 
           
                    u=u[0]

                   #print(f"string u {str(u)} ")             
                
                    if str(u) not in train_user:
                        continue
                    # sample a positive item from the observed items
                    i = np.random.choice(user_ratings_train[u],1,replace=False)
                    i=i[0]
            
            
                    # sample a negative item from the unobserved items
                    j =np.random.choice(items_list,1,replace=False)
                    j=j[0]
                    while j in user_ratings_train[u]:
                        j = np.random.choice(items_list,1,replace=False)
                        j=j[0]            
              
             
            

                    length = len(context_dict_u[u])
                    random.shuffle(context_dict_u[u])
                    if visited_u.get(u) < length:
                        # print(u)
                        index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                        for index in index_list:
                            context_u = context_dict_u[u][index]
                           
                            # center,context,neg,node_list,eta
                            for z in context_u:                       
                                tmp_z, tmp_loss = skip_gram(u, z, node_list_u, lam, alpha)
                                node_list_u[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_u[u] = index_list[-1]+3

                    length = len(context_dict_v[i])
                    random.shuffle(context_dict_v[i])
                    if visited_v.get(i) < length:
                        # print(v)
                        index_list = list(range(visited_v.get(i),min(visited_v.get(i)+1,length)))
                        for index in index_list:
                            context_v = context_dict_v[i][index]
                           
                            # center,context,neg,node_list,eta

                            for z in context_v:                       
                                if z not in user_ratings_train[u]:
                                  continue                       
                                tmp_z, tmp_loss = skip_gram(i, z, node_list_v, lam, beta)
                                node_list_v[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_v[i] = index_list[-1]+3

                    length = len(context_dict_v[j])
                    random.shuffle(context_dict_v[j])
                    if visited_v.get(j) < length:
                        # print(v)
                        index_list = list(range(visited_v.get(j),min(visited_v.get(j)+1,length)))
                        for index in index_list:
                            context_v = context_dict_v[j][index]                   
                            # center,context,neg,node_list,eta
                            for z in context_v:
                                tmp_z, tmp_loss = skip_gram(j, z, node_list_v, lam, beta)
                                node_list_v[z]['embedding_vectors'] += tmp_z
                                #loss += tmp_loss
                                loss = tmp_loss
                        visited_v[j] = index_list[-1]+3
                      

                    # r_ui = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[i]['embedding_vectors']).T)
                    # r_uj = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[j]['embedding_vectors']).T)
                    # r_uij =r_ui - r_uj
                    # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
                    ##sigmoid_tiled = np.tile(sigmoid, (128, 1)).T        
                    ##update using gradient descent
                    # grad_u = sigmoid * (node_list_v[i]['embedding_vectors'] - node_list_v[j]['embedding_vectors']) + reg * node_list_u[u]['embedding_vectors']
                    # grad_i = sigmoid * -node_list_u[u]['embedding_vectors'] + reg * node_list_v[i]['embedding_vectors']
                    # grad_j = sigmoid * node_list_u[u]['embedding_vectors'] + reg * node_list_v[j]['embedding_vectors']
                    # node_list_u[u]['embedding_vectors'] -= lam * grad_u
                    # node_list_v[i]['embedding_vectors'] -= lam * grad_i
                    # node_list_v[j]['embedding_vectors'] -= lam * grad_j
                   
                    #update_u, update_i,update_j, tmp_loss,biasV_i,biasV_j = Log_Like( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma,biasV)
                    update_u, update_i,update_j, tmp_loss= Log_Like2( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma)
                    loss += tmp_loss
                    node_list_u[u]['embedding_vectors'] += update_u
                    node_list_v[i]['embedding_vectors'] += update_i
                    node_list_v[j]['embedding_vectors'] += update_j
                    #biasV[i]+= biasV_i
                    #biasV[j]+= biasV_j

      
           
         
            
           
            delta_loss = abs(loss - last_loss)
            #if last_loss > loss:
                #lam *= 1.05
            #else:
                #lam *= 0.95
            #print(loss)
           
            last_loss = loss
            print(last_loss)
            #if last_loss < -1:
                #break
            sys.stdout.write(s1)
            sys.stdout.flush()
    t1 = time.time()
    print(f'Training done: end of training ; time cost: {(t1-t0):.2f}s')        
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg,recall = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))

def train_by_negbatch(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam,reg = args.alpha, args.beta, args.gamma, args.lam,args.lr
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, reg : %0.4f, ws : %d, ns : %d, l : % d, r : %d, max_iter : %d,dim : %d top-n : %d,top-k : %d,ABRW-beta : %d' % (alpha, beta, gamma, lam, args.lr, args.ws, args.ns,args.walk_length,args.number_walks,args.max_iter, args.d,args.top_n,args.ABRW_topk,args.ABRW_beta))
    print('========== processing data ===========')
    #model_path1=os.path.join('../content/ABiNE/', args.model_name)
    #datafile= os.path.join(model_path,"ratings.dat")
    
    dul = DataUtils(model_path)
    #data = pd.read_csv('/content/ABiNE/data/mooc/months1/march_users1.csv')
    #data = data.rename(columns={'user  ': 'user', 'item ':'item','label ':'label'})
   
    #train_data,test_data= dul.split_by_ratio(data, test_size=args.testRatio)
    #train_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_train.csv',index=False,header=False)
    ##files.download('/content/new_BiNE/data/mooc/months1/ratings_train.csv')
    #test_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_test.csv',index=False,header=False)
    
    train_user,train_item,train_rate=dul.read_train_data(args.train_data)
    test_user, test_item, test_rate = dul.read_test_data(args.test_data)
    n_train=len(train_item) 
    items=list(train_item) + list(test_item)
    res2 = [] 
    [res2.append(x) for x in items if x not in res2]
    items=res2 
    users_list=list(train_user) + list(test_user)
    res3 = [] 
    [res3.append(x) for x in users_list if x not in res3]
    users_list=res3   

    print(len(train_item))
    print(len(test_item)) 
    print(len(items)) 
    user_ratings_train,items= dul.load_data(args.train_data)
 
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
   
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    
    walk_generator(gul,args)
    model_path1 = os.path.join('../', 'model')    
    one=OneMode(model_path)
    
    one.u_load_data(args.u_graph_file,args.weighted,args.directed,args.graph_format,args.u_attribute_file,args.method)
    one.v_load_data(args.v_graph_file,args.weighted,args.directed,args.graph_format,args.v_attribute_file,args.method)
    
    if args.uattr == True:
        one.u_load_attr(args.u_attribute_file,args.method)   
        
    if args.vattr == True: 
        one.v_load_attr(args.v_attribute_file,args.method)
    
    walks_u,g_u=one.u_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.u_emb_file,args.uattr)
    walks_v,g_v=one.v_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.v_emb_file,args.vattr)   
    
    #print(g_v.G.nodes())
    items_list=g_v.G.nodes()
    
    
    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v= get_context_and_negative_samples2(gul,walks_u,walks_v,g_u,g_v, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args)
    last_loss, count, epsilon  = 0, 0, 0.0001
    print(len(node_list_u.keys()))
    print(len(users_list))
    print(len(node_list_v.keys()))
    #biasV = np.random.rand(len(items_list)) * 0.01
    #biasV=dict(zip(items_list,biasV))
    t0 = time.time()
    batch_size = args.batch_size
    if len(users_list) < batch_size:
            batch_size = len(users_list)
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(len(users_list)))

    batch_iters = len(users_list) // batch_size
        
        
        
        
    for iter in range(0, args.max_iter):
            s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
            loss = 0
            for _ in range(batch_iters):
               
        
       
                visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
                visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        

                #random.shuffle(edge_list)
                for _ in range(batch_size):
                    u = random.sample(users_list,1) 
           
                    u=u[0]

                   #print(f"string u {str(u)} ")             
                
                    if str(u) not in train_user:
                        continue
                    # sample a positive item from the observed items
                    #print(user_ratings_train[u])
                    #user_ratings_train[u] = np.array(user_ratings_train[u])
                    #print(user_ratings_train[u])
                    i = random.sample(user_ratings_train[u],1)
                    i=i[0]
            
            
                    # sample a negative item from the unobserved items
                    j =random.sample(items_list,1)
                    j=j[0]
                    while j in user_ratings_train[u]:
                        j = random.sample(items_list,1)
                        j=j[0]            
              
             
            

                    length = len(context_dict_u[u])
                    random.shuffle(context_dict_u[u])
                    if visited_u.get(u) < length:
                        # print(u)
                        index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                        for index in index_list:
                            context_u = context_dict_u[u][index]
                            neg_u = neg_dict_u[u][index]
                            # center,context,neg,node_list,eta
                            for z in context_u:                       
                                tmp_z, tmp_loss = skip_gram2(u, z, neg_u, node_list_u, lam, beta)
                                node_list_u[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_u[u] = index_list[-1]+3

                    length = len(context_dict_v[i])
                    random.shuffle(context_dict_v[i])
                    if visited_v.get(i) < length:
                        # print(v)
                        index_list = list(range(visited_v.get(i),min(visited_v.get(i)+1,length)))
                        for index in index_list:
                            context_v = context_dict_v[i][index]
                            neg_v = neg_dict_v[i][index]
                            # center,context,neg,node_list,eta

                            for z in context_v:                       
                                                  
                                tmp_z, tmp_loss = skip_gram2(i, z, neg_v, node_list_v, lam, beta)
                                node_list_v[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_v[i] = index_list[-1]+3

                    length = len(context_dict_v[j])
                    random.shuffle(context_dict_v[j])
                    if visited_v.get(j) < length:
                        # print(v)
                        index_list = list(range(visited_v.get(j),min(visited_v.get(j)+1,length)))
                        for index in index_list:
                            context_v = context_dict_v[j][index]
                            neg_v = neg_dict_v[j][index]                   
                            # center,context,neg,node_list,eta
                            for z in context_v:
                                
                                tmp_z, tmp_loss = skip_gram2(j, z, neg_v, node_list_v, lam, beta)
                                node_list_v[z]['embedding_vectors'] += tmp_z
                                #loss += tmp_loss
                                loss = tmp_loss
                        visited_v[j] = index_list[-1]+3
                      

                    # r_ui = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[i]['embedding_vectors']).T)
                    # r_uj = np.dot(np.array(node_list_u[u]['embedding_vectors']), np.array(node_list_v[j]['embedding_vectors']).T)
                    # r_uij =r_ui - r_uj
                    # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
                    ##sigmoid_tiled = np.tile(sigmoid, (128, 1)).T        
                    ##update using gradient descent
                    # grad_u = sigmoid * (node_list_v[i]['embedding_vectors'] - node_list_v[j]['embedding_vectors']) + reg * node_list_u[u]['embedding_vectors']
                    # grad_i = sigmoid * -node_list_u[u]['embedding_vectors'] + reg * node_list_v[i]['embedding_vectors']
                    # grad_j = sigmoid * node_list_u[u]['embedding_vectors'] + reg * node_list_v[j]['embedding_vectors']
                    # node_list_u[u]['embedding_vectors'] -= lam * grad_u
                    # node_list_v[i]['embedding_vectors'] -= lam * grad_i
                    # node_list_v[j]['embedding_vectors'] -= lam * grad_j
                   
                    #update_u, update_i,update_j, tmp_loss,biasV_i,biasV_j = Log_Like( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma,biasV)
                    update_u, update_i,update_j, tmp_loss= Log_Like2( u, i, j, node_list_u, node_list_v, lam, reg,args.gamma)
                    loss += tmp_loss
                    node_list_u[u]['embedding_vectors'] += update_u
                    node_list_v[i]['embedding_vectors'] += update_i
                    node_list_v[j]['embedding_vectors'] += update_j
                    #biasV[i]+= biasV_i
                    #biasV[j]+= biasV_j  
           
         
            
           
            delta_loss = abs(loss - last_loss)
            #if last_loss > loss:
                #lam *= 1.05
            #else:
                #lam *= 0.95
            #print(loss)
           
            last_loss = loss
            print(last_loss)
            #if last_loss < -1:
                #break
            sys.stdout.write(s1)
            sys.stdout.flush()
    t1 = time.time()
    print(f'Training done: end of training ; time cost: {(t1-t0):.2f}s')        
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg,recall = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, reg : %0.4f, ws : %d, ns : %d, l : % d, r : %d, max_iter : %d,dim : %d top-n : %d,top-k : %d,ABRW-beta : %d' % (alpha, beta, gamma, lam, args.lr, args.ws, args.ns,args.walk_length,args.number_walks,args.max_iter, args.d,args.top_n,args.ABRW_topk,args.ABRW_beta))
def train_by_abpoint(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam,reg = args.alpha, args.beta, args.gamma, args.lam,args.lr
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, reg : %0.4f, ws : %d, ns : %d, l : % d, r : %d, max_iter : %d,dim : %d top-n : %d,top-k : %d,ABRW-beta : %d' % (alpha, beta, gamma, lam, args.lr, args.ws, args.ns,args.walk_length,args.number_walks,args.max_iter, args.d,args.top_n,args.ABRW_topk,args.ABRW_beta))
    print('========== processing data ===========')
    #model_path1=os.path.join('../content/ABiNE/', args.model_name)
    #datafile= os.path.join(model_path,"ratings.dat")
    
    dul = DataUtils(model_path)
    #data = pd.read_csv('/content/ABiNE/data/mooc/months1/march_users1.csv')
    #data = data.rename(columns={'user  ': 'user', 'item ':'item','label ':'label'})
   
    #train_data,test_data= dul.split_by_ratio(data, test_size=args.testRatio)
    #train_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_train.csv',index=False,header=False)
    ##files.download('/content/new_BiNE/data/mooc/months1/ratings_train.csv')
    #test_data.to_csv('/content/ABiNE/data/mooc/months1/ratings_test.csv',index=False,header=False)
    
    train_user,train_item,train_rate=dul.read_train_data(args.train_data)
    test_user, test_item, test_rate = dul.read_test_data(args.test_data)
    n_train=len(train_item) 
    items=list(train_item) + list(test_item)
    res2 = [] 
    [res2.append(x) for x in items if x not in res2]
    items=res2 
    users_list=list(train_user) + list(test_user)
    res3 = [] 
    [res3.append(x) for x in users_list if x not in res3]
    users_list=res3   

    print(len(train_item))
    print(len(test_item)) 
    print(len(items)) 
    user_ratings_train,items= dul.load_data(args.train_data)
 
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
   
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    
    walk_generator(gul,args)
    model_path1 = os.path.join('../', 'model')    
    one=OneMode(model_path)
    
    one.u_load_data(args.u_graph_file,args.weighted,args.directed,args.graph_format,args.u_attribute_file,args.method)
    one.v_load_data(args.v_graph_file,args.weighted,args.directed,args.graph_format,args.v_attribute_file,args.method)
    
    if args.uattr == True:
        one.u_load_attr(args.u_attribute_file,args.method)   
        
    if args.vattr == True: 
        one.v_load_attr(args.v_attribute_file,args.method)
    
    walks_u,g_u=one.u_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.u_emb_file,args.uattr)
    walks_v,g_v=one.v_embedding(args.method,args.dim,args.ABRW_topk,args.ABRW_beta,args.ABRW_beta_mode,args.ABRW_alpha,args.number_walks,args.walk_length,args.window_size,args.workers,args.save_emb,args.v_emb_file,args.vattr)   
    
    #print(g_v.G.nodes())
    items_list=g_v.G.nodes()
    
    
    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v= get_context_and_negative_samples2(gul,walks_u,walks_v,g_u,g_v, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, items_list,node_list_u, node_list_v, args)
    last_loss, count, epsilon  = 0, 0, 0.0001
    print(len(node_list_u.keys()))
    print(len(users_list))
    print(len(node_list_v.keys()))
    #biasV = np.random.rand(len(items_list)) * 0.01
    #biasV=dict(zip(items_list,biasV))
    print("============== training ==============")
    for iter in range(0, args.max_iter):
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        random.shuffle(edge_list)
        for i in range(len(edge_list)):
            u, v, w = edge_list[i]
              
            length = len(context_dict_u[u])
            random.shuffle(context_dict_u[u])
            if visited_u.get(u) < length:
               
                index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for z in context_u:
                        tmp_z, tmp_loss = skip_gram2(u, z, neg_u, node_list_u, lam, alpha)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = index_list[-1]+3
                
          
            length = len(context_dict_v[v])
            random.shuffle(context_dict_v[v])
            if visited_v.get(v) < length:
                # print(v)
                index_list = list(range(visited_v.get(v),min(visited_v.get(v)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[v][index]
                    neg_v = neg_dict_v[v][index]
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        tmp_z, tmp_loss = skip_gram2(v, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[v] = index_list[-1]+3
            # adj_list_u= node_list_u[z]['embedding_vectors'].dot(context_dict_u[u].T)
                # adj_list_v= node_list_v[z]['embedding_vectors'].dot(context_dict_v[v].T)
            #save_adj_to_file(node_list_u,node_list_v,model_path,args)  
                

            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v

        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        #print(delta_loss)
        print(loss)
        print(last_loss)
        if delta_loss < epsilon:
            break
        sys.stdout.write(s1)
        sys.stdout.flush()
  
    t1 = time.time()
         
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg,recall = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, reg : %0.4f, ws : %d, ns : %d, l : % d, r : %d, max_iter : %d,dim : %d top-n : %d,top-k : %d,ABRW-beta : %d' % (alpha, beta, gamma, lam, args.lr, args.ws, args.ns,args.walk_length,args.number_walks,args.max_iter, args.d,args.top_n,args.ABRW_topk,args.ABRW_beta))
def ndarray_tostring(array):
    string = ""
    for item in array:
        string += str(item).strip()+" "
    return string+"\n"

def save_to_file(node_list_u,node_list_v,model_path,args):
    with open(args.vectors_u,"w") as fw_u:
        for u in vectors_u.keys():
            fw_u.write(u+" "+ ndarray_tostring(vectors_u[u]))
    with open(args.vectors_v,"w") as fw_v:
        for v in vectors_v.keys():
            fw_v.write(v+" "+ndarray_tostring(vectors_v[v])) 
 


def main():
    parser = ArgumentParser("BiNE",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

   
    parser.add_argument('--train-data', default=r'../data/mooc/months1/ratings_train.csv',
                        help='Input graph file.')

    parser.add_argument('--test-data', default=r'../data/mooc/months1/ratings_test.csv')
    parser.add_argument('--model-name', default='data/mooc',
                        help='name of model.')

    parser.add_argument('--vectors-u', default=r'../data/mooc/vectors_u.dat',
                        help="file of embedding vectors of U")

    parser.add_argument('--vectors-v', default=r'../data/mooc/vectors_v.dat',
                        help="file of embedding vectors of V")

    parser.add_argument('--case-train', default=r'data/wiki/case_train.dat',
                        help="file of training data for LR")

    parser.add_argument('--case-test', default=r'data/wiki/case_test.dat',
                        help="file of testing data for LR")

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=128, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.01, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='trade-off parameter gamma.')

    parser.add_argument('--lam', default=0.01, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--top-n', default=10, type=int,
                        help='recommend top-n items for each user.')

    parser.add_argument('--rec', default=1, type=int,
                        help='calculate the recommendation metrics.')

    parser.add_argument('--lip', default=0, type=int,
                        help='calculate the link prediction metrics.')
    parser.add_argument('--testRatio', type =float, default=0.40,
                        help="Test to training ratio.Train percentage")

    parser.add_argument('--large', default=0, type=int,
                        help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph')

    parser.add_argument('--mode', default='hits', type=str,
                        help='metrics of centrality')
    parser.add_argument('--loss_function', default=1, type=int,
                        help='0-pointwise loss function, 1-pairwise loss function')
    parser.add_argument('--uattr', default=False, type=bool,
                        help='when user attribute is available set True when not available set false')
    parser.add_argument('--vattr', default=True, type=bool,
                        help='when item attribute is available set True when not available set false')
    
     #-----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--graph_format', default='edgelist', choices=['adjlist', 'edgelist'],
                        help='graph/network format')
    parser.add_argument('--u_graph_file', default= r'../data/mooc/homogeneous_u.dat',
                        help='graph/network file')
    parser.add_argument('--v_graph_file', default= r'../data/mooc/homogeneous_v.dat',
                        help='graph/network file')
    parser.add_argument('--u_attribute_file', default=r'data/mooc/u_attr.txt',
                        help='node attribute/feature file')
    parser.add_argument('--v_attribute_file', default=r'data/mooc/v_attr.txt',
                        help='node attribute/feature file')
    parser.add_argument('--label-file', default='data/cora/cora_label.txt',
                        help='node label file')     
    parser.add_argument('--dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='lp_and_nc', choices=['none', 'lp', 'nc', 'lp_and_nc'],
                        help='choices of downstream tasks: none, lp, nc, lp_and_nc')
    parser.add_argument('--link-remove', default=0.1, type=float, 
                        help='simulate randomly missing links if necessary; a ratio ranging [0.0, 1.0]')
    parser.add_argument('--label-reserved', default=0.7, type=float,
                        help='for nc task, train/test split, a ratio ranging [0.0, 1.0]')
    parser.add_argument('--directed', default=False, action='store_true',
                        help='directed or undirected graph')
    parser.add_argument('--weighted', default=True, action='store_true',
                        help='weighted or unweighted graph')
    parser.add_argument('--save-emb', default=False, action='store_true',
                        help='save emb to disk if True')
    parser.add_argument('--u-emb-file', default='../data/mooc/u_node_embs.txt',
                        help='node embeddings file; suggest: data_method_dim_embs.txt')
    parser.add_argument('--v-emb-file', default='../data/mooc/v_node_embs.txt',
                        help='node embeddings file; suggest: data_method_dim_embs.txt')
    #-------------------------------------------------method settings-----------------------------------------------------------
    parser.add_argument('--method', default='abrw', choices=['deepwalk', 'node2vec', 'line', 'grarep',
                                                             'abrw', 'attrpure', 'attrcomb', 'tadw', 'aane',
                                                             'sagemean', 'sagegcn', 'gcn', 'asne'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--ABRW-topk', default=30, type=int,
                        help='select the most attr similar top k nodes of a node; ranging [0, # of nodes]')
    parser.add_argument('--ABRW-alpha', default=2.71828, type=float,
                        help='control the shape of characteristic curve of adaptive beta, ranging [0, inf]')
    parser.add_argument('--ABRW-beta-mode', default=1, type=int,
                        help='1: fixed; 2: adaptive based on average degree; 3: adaptive based on each node degree')
    parser.add_argument('--ABRW-beta', default=0.2, type=float,
                        help='balance struc and attr info; ranging [0, 1]; disabled if beta-mode 2 or 3')
    parser.add_argument('--AANE-lamb', default=0.05, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--AANE-rho', default=5, type=float,
                        help='penalty parameter; ranging [0, inf]')
    parser.add_argument('--AANE-maxiter', default=10, type=int,
                        help='max iter')
    parser.add_argument('--TADW-lamb', default=0.2, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--TADW-maxiter', default=20, type=int,
                        help='max iter')
    parser.add_argument('--ASNE-lamb', default=1.0, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--AttrComb-mode', default='concat', type=str,
                        help='choices of mode: concat, elementwise-mean, elementwise-max')
    parser.add_argument('--Node2Vec-p', default=0.5, type=float,  # if p=q=1.0 node2vec = deepwalk
                        help='trade-off BFS and DFS; grid search [0.25; 0.50; 1; 2; 4]')
    parser.add_argument('--Node2Vec-q', default=0.5, type=float,
                        help='trade-off BFS and DFS; grid search [0.25; 0.50; 1; 2; 4]')
    parser.add_argument('--GraRep-kstep', default=4, type=int,
                        help='use k-step transition probability matrix, error if dim%Kstep!=0')
    parser.add_argument('--LINE-order', default=3, type=int,
                        help='choices of the order(s): 1->1st, 2->2nd, 3->1st+2nd')
    parser.add_argument('--LINE-negative-ratio', default=5, type=int,
                        help='the negative ratio')
    # for walk based methods; some Word2Vec SkipGram parameters are not specified here
    parser.add_argument('--number-walks', default=10, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=60, type=int,
                        help='length of each random walk')
    parser.add_argument('--window-size', default=10, type=int,
                        help='window size of skipgram model')
    parser.add_argument('--workers', default=36, type=int,
                        help='# of parallel processes.')
    # for deep learning based methods; parameters about layers and neurons used are not specified here
    parser.add_argument('--learning-rate', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=500, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='epochs')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout rate (1 - keep probability)')

    args = parser.parse_args()
    train_by_negbatch(args)
    



if __name__ == "__main__":
    sys.exit(main())


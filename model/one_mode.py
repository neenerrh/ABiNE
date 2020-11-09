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
import matplotlib.pyplot as plt
import networkx as nx


class OneMode(object):
    
    def __init__(self, model_path):
        self.g_u=Graph()
        self.g_v=Graph()
        

        #self.g = Graph() #see graph.py for commonly-used APIs and use g.G to access NetworkX APIs
       

    def u_load_data(self,graph_file,weighted,directed,graph_format,attribute_file,method):
        #---------------------------------------STEP1: load data-----------------------------------------------------
        print('\nSTEP1: start loading data......')
        t1 = time.time()
        #load graph structure info------
        if graph_format == 'adjlist':
            self.g.read_adjlist(path=graph_file, directed=directed)
          
        elif graph_format == 'edgelist':
             self.g_u.read_edgelist(path=graph_file, weighted=weighted, directed=directed)
             #self.g.read_edgelist(path=graph_file, weighted=weighted, directed=directed)
         
         #-------------------------------------load node attribute info------
        #is_ane = (method == 'abrw' or method == 'tadw' or method == 'attrpure' or method == 'attrcomb' or method == 'aane')
        #if is_ane:
            #assert attribute_file != ''
            #self.g_u.read_node_attr(attribute_file)
            #self.g.read_node_attr(attribute_file)
        #load node label info------
        #t2 = time.time()
        
    def v_load_data(self,graph_file,weighted,directed,graph_format,attribute_file,method):
        #---------------------------------------STEP1: load data-----------------------------------------------------
        print('\nSTEP1: start loading data......')
        t1 = time.time()
        #load graph structure info------
        if graph_format == 'adjlist':
            g.read_adjlistu(path=args.graph_file, directed=args.directed)
            g.read_adjlistv(path=args.graph_file, directed=args.directed)
        elif graph_format == 'edgelist':
            self.g_v.read_edgelist(path=graph_file, weighted=weighted, directed=directed)
       
        
    def u_load_attr(self,attribute_file,method):   
        #-------------------------------------load node attribute info------
        is_ane = (method == 'abrw' or method == 'tadw' or method == 'attrpure' or method == 'attrcomb' or method == 'aane')
        if is_ane:
            assert attribute_file != ''
            self.g_u.read_node_attr(attribute_file)
            #self.g.read_node_attr(attribute_file)
            
        #load node label info------
           
        t2 = time.time()
        
          
        
        
       
        
    def v_load_attr(self,attribute_file,method):   
        #-------------------------------------load node attribute info------
        is_ane = (method == 'abrw' or method == 'tadw' or method == 'attrpure' or method == 'attrcomb' or method == 'aane')
        if is_ane:
            assert attribute_file != ''
          
            self.g_v.read_node_attr(attribute_file)
        #load node label info------
            
        t2 = time.time()
        #print(f'STEP1: end loading data; time cost: {(t2-t1):.2f}s')

    def prepare_data(self, task):
        #---------------------------------------STEP2: prepare data----------------------------------------------------
        print('\nSTEP2: start preparing data for link pred task......')
        t1 = time.time()
        test_node_pairs=[]
        test_edge_labels=[]
        if args.task == 'lp' or args.task == 'lp_and_nc':
            edges_removed = g.remove_edge(ratio=args.link_remove)
            test_node_pairs, test_edge_labels = generate_edges_for_linkpred(graph=g, edges_removed=edges_removed, balance_ratio=1.0)
        t2 = time.time()
        print(f'STEP2: end preparing data; time cost: {(t2-t1):.2f}s')

    
    def u_embedding(self,method,dim, ABRW_topk,ABRW_beta,ABRW_beta_mode,ABRW_alpha,number_walks,walk_length,window_size,workers, save_emb,emb_file):
        #-----------------------------------STEP3: upstream embedding task-------------------------------------------------
        print('\nSTEP3: start learning embeddings......')
        #print(f'the graph: {self.g}; \nthe model used: {method}; \
            #\nthe # of edges used during embedding (edges maybe removed if lp task): {self.g.get_num_edges()}; \
            #\nthe # of nodes: {self.g.get_num_nodes()}; \nthe # of isolated nodes: {self.g.get_num_isolates()}; \nis directed graph: {self.g.get_isdirected()}')
        t1 = time.time()
        model = None
        if  method == 'abrw':
            
            model = abrw.ABRW(graph=self.g_u, dim=dim, topk=ABRW_topk,beta=ABRW_beta, beta_mode=ABRW_beta_mode,alpha=ABRW_alpha, number_walks=number_walks,walk_length=walk_length, window=window_size, workers=workers)
            
            node_u_num= model.save_embeddings(emb_file + time.strftime(' %Y%m%d-%H%M%S', time.localtime()))
            print(f'Save node embeddings in file: {emb_file}')
            
        elif method == 'aane':
            model = aane.AANE(graph=g, dim=args.dim, lambd=args.AANE_lamb, rho=args.AANE_rho, maxiter=args.AANE_maxiter, 
                            mode='comb') #mode: 'comb' struc and attri or 'pure' struc
        elif method == 'tadw':
            model = tadw.TADW(graph=g, dim=args.dim, lamb=args.TADW_lamb, maxiter=args.TADW_maxiter)
        elif method == 'attrpure':
            model = attrpure.ATTRPURE(graph=g, dim=args.dim, mode='pca')  #mode: pca or svd
        elif method == 'attrcomb':
            model = attrcomb.ATTRCOMB(graph=g, dim=args.dim, comb_with='deepwalk', number_walks=args.number_walks, walk_length=args.walk_length,
                                    window=args.window_size, workers=args.workers, comb_method=args.AttrComb_mode)  #comb_method: concat, elementwise-mean, elementwise-max
        elif method == 'deepwalk':
            model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, dw=True)
        elif method == 'node2vec':
            model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, p=args.Node2Vec_p, q=args.Node2Vec_q)
        else:
            print('method not found...')
            exit(0)
        t2 = time.time()
        print(f'STEP3: end learning embeddings; time cost: {(t2-t1):.2f}s')
        vectors_u=model.vectors
        #print(vectors_u)
        return vectors_u,node_u_num
        
     
           
    def v_embedding(self,method,dim, ABRW_topk,ABRW_beta,ABRW_beta_mode,ABRW_alpha,number_walks,walk_length,window_size,workers, save_emb,emb_file):
        #-----------------------------------STEP3: upstream embedding task-------------------------------------------------
        print('\nSTEP3: start learning embeddings......')
        #print(f'the graph: {self.g}; \nthe model used: {method}; \
            #\nthe # of edges used during embedding (edges maybe removed if lp task): {self.g.get_num_edges()}; \
            #\nthe # of nodes: {self.g.get_num_nodes()}; \nthe # of isolated nodes: {self.g.get_num_isolates()}; \nis directed graph: {self.g.get_isdirected()}')
        t1 = time.time()
        model = None
        if  method == 'abrw': 
            model = abrw.ABRW(graph=self.g_v, dim=dim, topk=ABRW_topk,beta=ABRW_beta, beta_mode=ABRW_beta_mode,alpha=ABRW_alpha, number_walks=number_walks,walk_length=walk_length, window=window_size, workers=workers)
            #model.save_embeddings(emb_file + time.strftime(' %Y%m%d-%H%M%S', time.localtime()))
            #print(f'Save node embeddings in file: {emb_file}')
            node_v_num=model.save_embeddings(emb_file + time.strftime(' %Y%m%d-%H%M%S', time.localtime()))
            print(f'Save node embeddings in file: {emb_file}')
        elif method == 'aane':
            model = aane.AANE(graph=g, dim=args.dim, lambd=args.AANE_lamb, rho=args.AANE_rho, maxiter=args.AANE_maxiter, 
                            mode='comb') #mode: 'comb' struc and attri or 'pure' struc
        elif method == 'tadw':
            model = tadw.TADW(graph=g, dim=args.dim, lamb=args.TADW_lamb, maxiter=args.TADW_maxiter)
        elif method == 'attrpure':
            model = attrpure.ATTRPURE(graph=g, dim=args.dim, mode='pca')  #mode: pca or svd
        elif method == 'attrcomb':
            model = attrcomb.ATTRCOMB(graph=g, dim=args.dim, comb_with='deepwalk', number_walks=args.number_walks, walk_length=args.walk_length,
                                    window=args.window_size, workers=args.workers, comb_method=args.AttrComb_mode)  #comb_method: concat, elementwise-mean, elementwise-max
        elif method == 'deepwalk':
            model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, dw=True)
        elif method == 'node2vec':
            model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, p=args.Node2Vec_p, q=args.Node2Vec_q)
        else:
            print('method not found...')
            exit(0)
        t2 = time.time()
        print(f'STEP3: end learning embeddings; time cost: {(t2-t1):.2f}s')
        vectors_v=model.vectors
        #print(vectors_v)
        return vectors_v,node_v_num
    
    def downstream(self):
        #---------------------------------------STEP4: downstream task-----------------------------------------------
        print('\nSTEP4: start evaluating ......: ')
        t1 = time.time()
        vectors = model.vectors
        del model, g
        #------lp task
        if args.task == 'lp' or args.task == 'lp_and_nc':
            #X_test_lp, Y_test_lp = read_edge_label(args.label_file)  #if you want to load your own lp testing data
            print(f'Link Prediction task; the percentage of positive links for testing: {(args.link_remove*100):.2f}%'
                + ' (by default, also generate equal negative links for testing)')
            clf = lpClassifier(vectors=vectors)     #similarity/distance metric as clf; basically, lp is a binary clf probelm
            clf.evaluate(test_node_pairs, test_edge_labels)
        #------nc task
        if args.task == 'nc' or args.task == 'lp_and_nc':
            X, Y = read_node_label(args.label_file)
            print(f'Node Classification task; the percentage of labels for testing: {((1-args.label_reserved)*100):.2f}%')
            clf = ncClassifier(vectors=vectors, clf=LogisticRegression())   #use Logistic Regression as clf; we may choose SVM or more advanced ones
            clf.split_train_evaluate(X, Y, args.label_reserved)
        t2 = time.time()
        print(f'STEP4: end evaluating; time cost: {(t2-t1):.2f}s')
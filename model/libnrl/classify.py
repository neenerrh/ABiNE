"""
modified by Chengbin Hou 2018

originally from https://github.com/thunlp/OpenNE
"""

import numpy as np
import math
import random
import networkx as nx
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, roc_curve, auc
from sklearn.preprocessing import MultiLabelBinarizer

# node classification classifier
class ncClassifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)  #here clf is LR
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        #np.random.seed(seed) 
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)  #why??? for binarizer.transform?? 
        return self.evaluate(X_test, Y_test)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)  #to support multi-labels, fit means dict mapping {orig cat: binarized vec}
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)  #since we have use Y_all fitted, then we simply transform
        self.clf.fit(X_train, Y)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        # see TopKRanker(OneVsRestClassifier)
        Y = self.clf.predict(X_, top_k_list=top_k_list)  # the top k probs to be output...
        return Y

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]  #multi-labels, diff len of labels of each node
        Y_ = self.predict(X, top_k_list)  #pred val of X_test i.e. Y_pred
        Y = self.binarizer.transform(Y)   #true val i.e. Y_test
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        print(results)
        return results

class TopKRanker(OneVsRestClassifier):  #orignal LR or SVM is for binary clf
    def predict(self, X, top_k_list):   #re-define predict func of OneVsRestClassifier
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist() #denote labels
            probs_[:] = 0      #reset probs_ to all 0
            probs_[labels] = 1 #reset probs_ to 1 if labels denoted...
            all_labels.append(probs_)
        return np.asarray(all_labels)

'''
#note: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in samples with no true labels
#see: https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
'''

'''
import matplotlib.pyplot as plt
def plt_roc(y_test, y_score):
    """
    calculate AUC value and plot the ROC curve
    """
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
    plt.plot(fpr, tpr, color='black', lw = 1)
    plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
    plt.text(0.5,0.3,'ROC curve (area = %0.3f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return roc_auc
'''

# link prediction binary classifier
class lpClassifier(object):

    def __init__(self, vectors):
        self.embeddings = vectors

    def evaluate(self, X_test, Y_test, seed=0):  #clf here is simply a similarity/distance metric
        state = np.random.get_state()
        #np.random.seed(seed)
        test_size = len(X_test)
        #shuffle_indices = np.random.permutation(np.arange(test_size))
        #X_test = [X_test[shuffle_indices[i]] for i in range(test_size)]
        #Y_test = [Y_test[shuffle_indices[i]] for i in range(test_size)]

        Y_true = [int(i) for i in Y_test]
        Y_probs = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]]).reshape(-1,1)
            end_node_emb = np.array(self.embeddings[X_test[i][1]]).reshape(-1,1)
            score = cosine_similarity(start_node_emb, end_node_emb) #ranging from [-1, +1]
            Y_probs.append( (score+1)/2.0 )     #switch to prob... however, we may also directly y_score = score 
                                                #in sklearn roc... which yields the same reasult
        roc = roc_auc_score(y_true = Y_true, y_score = Y_probs)
        if roc < 0.5:
            roc = 1.0 - roc    #since lp is binary clf task, just predict the opposite if<0.5
        print("roc=", "{:.9f}".format(roc))
        #plt_roc(Y_true, Y_probs) #enable to plot roc curve and return auc value

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

def cosine_similarity(a, b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    #return sum/(norm(a) * norm(b))
    return sum/(norm(a) * norm(b) + 1e-20)  #fix numerical issue 1e-20 almost = 0!

'''
#cosine_similarity realized by use...
#or try sklearn....
        from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, cosine_distances, euclidean_distances  # we may try diff metrics
        #ref http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
'''

def lp_train_test_split(graph, ratio=0.5, neg_pos_link_ratio=1.0, test_pos_links_ratio=0.1):
    #randomly split links/edges into training set and testing set
    #*** note: we do not assume every node must be connected after removing links
    #*** hence, the resulting graph might have few single nodes --> more realistic scenario
    #*** e.g. a user just sign in a website has no link to others
    
    #graph: OpenANE graph data strcture
    #ratio: perc of links for training; ranging [0, 1]
    #neg_pos_link_ratio: 1.0 means neg-links/pos-links = 1.0 i.e. balance case; raning [0, +inf)
    g = graph
    test_pos_links = int(nx.number_of_edges(g.G) * test_pos_links_ratio)

    print("test_pos_links_ratio {:.2f}, test_pos_links {:.2f}, neg_pos_link_ratio is {:.2f}, links for training {:.2f}%,".format(test_pos_links_ratio, test_pos_links, neg_pos_link_ratio, ratio*100))
    test_pos_sample = []
    test_neg_sample = []

    #random.seed(2018) #generate testing set that contains both pos and neg samples
    test_pos_sample = random.sample(g.G.edges(), test_pos_links)
    #test_neg_sample = random.sample(list(nx.classes.function.non_edges(g.G)), int(test_size * neg_pos_link_ratio)) #using nx build-in func, not efficient, to do...
    #more efficient way: 
    test_neg_sample = []
    num_neg_sample = int(test_pos_links * neg_pos_link_ratio)
    num = 0
    while num < num_neg_sample:
        pair_nodes = np.random.choice(g.look_back_list, size=2, replace=False)
        if pair_nodes not in g.G.edges():
            num += 1
            test_neg_sample.append(list(pair_nodes))
    
    test_edge_pair = test_pos_sample + test_neg_sample 
    test_edge_label = list(np.ones(len(test_pos_sample))) + list(np.zeros(len(test_neg_sample)))

    print('before removing, the # of links: ', nx.number_of_edges(g.G), ';   the # of single nodes: ', g.numSingleNodes())
    g.G.remove_edges_from(test_pos_sample)  #training set should NOT contain testing set i.e. delete testing pos samples
    g.simulate_sparsely_linked_net(link_reserved = ratio)  #simulate sparse net
    print('after removing,  the # of links: ', nx.number_of_edges(g.G), ';   the # of single nodes: ', g.numSingleNodes())
    print("# training links {0}; # positive testing links {1}; # negative testing links {2},".format(nx.number_of_edges(g.G), len(test_pos_sample), len(test_neg_sample)))
    return g.G, test_edge_pair, test_edge_label

#---------------------------------ulits for downstream tasks--------------------------------
def load_embeddings(filename):   
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {} 
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def read_edge_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[:2])
        Y.append(vec[2])
    fin.close()
    return X, Y
    
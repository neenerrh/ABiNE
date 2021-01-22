#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

import random
from io import open
import os
import numpy as np
from collections import defaultdict

class DataUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def rename(self, datafile):
        """
        Distinguish two types of node and rename
        """
        #datafile= os.path.join(self.model_path,"rating.dat")
        with open(os.path.join(self.model_path,"rating.dat"), "w") as fw:
            with open(datafile, "r", encoding="UTF-8") as fin:
                line = fin.readline()
                while line:
                    user, item, rating = line.strip("\t").split("\t")
                    fw.write(user + "\t" + item + "\t" + rating + "\n")
                    line = fin.readline()

    
    def load_data(self, path):
        user_ratings = defaultdict(set)
        items=[]
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i ,rate= line.strip().split(",")
                #u = int(u)
                #i = int(i)
                items.append(i)
                user_ratings[u].add(i)
        res = [] 
        [res.append(x) for x in items if x not in res]         
        return user_ratings,res

    def split_by_ratio(self,data, order=True, shuffle=False, test_size=None,
                   multi_ratios=None, filter_unknown=True, pad_unknown=False,
                   seed=42):
        np.random.seed(seed)
       
        print(data.columns)
        self.test_size=test_size
        self.multi_ratios=multi_ratios
        assert ('user' in data.columns), "data must contains user column"
        ratios, n_splits = self._check_and_convert_ratio(self.test_size, self.multi_ratios)
        print(ratios,n_splits)

        n_users = data.user.nunique()
        user_indices = data.user.to_numpy()
        user_split_indices = self._groupby_user(user_indices, order)

        cum_ratios = np.cumsum(ratios).tolist()[:-1]
        split_indices_all = [[] for _ in range(n_splits)]
        for u in range(n_users):
            u_data = user_split_indices[u]
            u_data_len = len(u_data)
            if u_data_len <= 3:   # keep items of rare users in trainset
                split_indices_all[0].extend(u_data)
            else:
                u_split_data = np.split(u_data, [
                    round(cum * u_data_len) for cum in cum_ratios
                ])
                for i in range(n_splits):
                    split_indices_all[i].extend(list(u_split_data[i]))

        if shuffle:
            split_data_all = tuple(
                np.random.permutation(data[idx]) for idx in split_indices_all)
        else:
            split_data_all = list(data.iloc[idx] for idx in split_indices_all)

        if filter_unknown:
            split_data_all = self._filter_unknown_user_item(split_data_all)
        elif pad_unknown:
            split_data_all = self._pad_unknown_user_item(split_data_all)
        return split_data_all
    
    def split_data(self, percent,loss_function):
        """
        split data
        :param percent:
        :return:
        """
        test_user,test_item,test_rate,rating = set(), set(), {},{}
        print(loss_function)
        print(self.model_path)
        
        if loss_function == 0:
            with open(os.path.join(self.model_path, "ratings1.dat"), "r") as fin, open(os.path.join(self.model_path, "ratings_train.dat"),"w") as ftrain, open(os.path.join(self.model_path,"ratings_test.dat"), "w") as ftest:
                for line in fin.readlines():
                    user, item, rate = line.strip().split("\t")
                    if rating.get(user) is None:
                        rating[user] = {}
                    rating[user][item] = rate
                for u in rating.keys():
                    item_list = rating[u].keys()
                    sample_list = random.sample(item_list, int(len(item_list) * percent))
                    for item in item_list:
                        if item in sample_list:
                            ftrain.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
                        else:
                            if test_rate.get(u) is None:
                                test_rate[u] = {}
                            test_rate[u][item] = float(rating[u][item])
                            test_user.add(u)
                            test_item.add(item)
                            ftest.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
        else:
            with open(os.path.join(self.model_path, "months1/ratings.dat"), "r") as fin, open(os.path.join(self.model_path, "months1/ratings_train.dat"),"w") as ftrain, open(os.path.join(self.model_path,"months1/ratings_test.dat"), "w") as ftest:
                for line in fin.readlines():
                    user, item, rate = line.strip().split("\t")
                    if rating.get(user) is None:
                        rating[user] = {}
                    rating[user][item] = rate
                for u in rating.keys():
                    item_list = rating[u].keys()
                    sample_list = random.sample(item_list, int(len(item_list) * percent))
                    for item in item_list:
                        if item in sample_list:
                            ftrain.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
                        else:
                            if test_rate.get(u) is None:
                                test_rate[u] = {}
                            test_rate[u][item] = float(rating[u][item])
                            test_user.add(u)
                            test_item.add(item)
                            ftest.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
        return test_user, test_item, test_rate
    
    def read_train_data(self,filename=None):
        if filename is None:
            filename = os.path.join(self.model_path,"ratings_train.dat")
        users,items,rates = set(), set(), {}
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rate = line.strip().split(',')
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][item] = float(rate)
                users.add(user)
                items.add(item)
                line = fin.readline()
        return users, items, rates
    def read_test_data(self,filename=None):
        if filename is None:
            filename = os.path.join(self.model_path,"ratings_test.dat")
        users,items,rates = set(), set(), {}
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rate = line.strip().split(',')
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][item] = float(rate)
                users.add(user)
                items.add(item)
                line = fin.readline()
        return users, items, rates

    def _check_and_convert_ratio(self,test_size, multi_ratios):
            if not test_size and not multi_ratios:
                raise ValueError("must provide either 'test_size' or 'multi_ratios'")

            elif test_size is not None:
                assert isinstance(test_size, float), "test_size must be float value"
                assert 0.0 < test_size < 1.0, "test_size must be in (0.0, 1.0)"
                ratios = [1 - test_size, test_size]
                return ratios, 2

            elif isinstance(multi_ratios, (list, tuple)):
                assert len(multi_ratios) > 1, (
                    "multi_ratios must at least have two elements")
                assert all([r > 0.0 for r in multi_ratios]), (
                    "ratios should be positive values")
                if math.fsum(multi_ratios) != 1.0:
                    ratios = [r / math.fsum(multi_ratios) for r in multi_ratios]
                else:
                    ratios = multi_ratios
                return ratios, len(ratios)

            else:
                raise ValueError("multi_ratios should be list or tuple")
    def _groupby_user(self,user_indices, order):
        sort_kind = "mergesort" if order else "quicksort"
        users, user_position, user_counts = np.unique(user_indices,
                                                      return_inverse=True,
                                                      return_counts=True)
        user_split_indices = np.split(np.argsort(user_position, kind=sort_kind),
                                      np.cumsum(user_counts)[:-1])
        return user_split_indices
      
    def _filter_unknown_user_item(self,data_list):
        train_data = data_list[0]
        unique_values = dict(user=set(train_data.user.tolist()),
                            item=set(train_data.item.tolist()))

        split_data_all = [train_data]
        for i, test_data in enumerate(data_list[1:], start=1):
            # print(f"Non_train_data {i} size before filtering: {len(test_data)}")
            out_of_bounds_row_indices = set()
            for col in ["user", "item"]:
                for j, val in enumerate(test_data[col]):
                    if val not in unique_values[col]:
                        out_of_bounds_row_indices.add(j)

            mask = np.arange(len(test_data))
            test_data_clean = test_data[~np.isin(
                mask, list(out_of_bounds_row_indices))]
            split_data_all.append(test_data_clean)
            # print(f"Non_train_data {i} size after filtering: "
            #      f"{len(test_data_clean)}")
        return split_data_all

    def _pad_unknown_user_item(data_list):
        train_data, test_data = data_list
        n_users = train_data.user.nunique()
        n_items = train_data.item.nunique()
        unique_users = set(train_data.user.tolist())
        unique_items = set(train_data.item.tolist())

        split_data_all = [train_data]
        for i, test_data in enumerate(data_list[1:], start=1):
            test_data.loc[~test_data.user.isin(unique_users), "user"] = n_users
            test_data.loc[~test_data.item.isin(unique_items), "item"] = n_items
            split_data_all.append(test_data)
        return split_data_all




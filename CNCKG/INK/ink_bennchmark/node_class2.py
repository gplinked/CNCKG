import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import numpy as np

import sys
import time
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
# import weka.core.converters as converters
# from weka.core.dataset import Instances
import os
# 添加ink模块的路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from ink.base.connectors import RDFLibConnector

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold

# from pyrdf2vec.graphs import KG
import pandas as pd

from multiprocessing import Pool
from hashlib import md5
from typing import List,Set, Tuple, Any
from tqdm import tqdm
import rdflib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from pympler import asizeof


import os.path
import csv

if not os.path.exists('results.csv'):
    with open('results.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Method', 'depth', 'NB_acc', 'NN_acc', 'DT_acc', 'SVC_acc', 'LR_acc', 'Extra_acc', 'Random_acc',
         'NB_weighted_F1', 'NN_weighted_F1', 'DT_weighted_F1', 'SVC_weighted_F1', 'LR_weighted_F1', 'Extra_weighted_F1', 'Random_weighted_F1',
         'NB_weighted_precision', 'NN_weighted_precision', 'DT_weighted_precision', 'SVC_weighted_precision', 'LR_weighted_precision', 'Extra_weighted_precision',
         'Random_weighted_precision', 'NB_weighted_recall', 'NN_weighted_recall', 'DT_weighted_recall', 'SVC_weighted_recall',
         'LR_weighted_recall', 'Extra_weighted_recall','Random_weighted_recall',
         'Create_time', 'NB_Train_time','NN_Train_time','DT_Train_time','SVC_Train_time','LR_Train_time','Extra_Train_time','Random_Train_time',
         'NB_Test_time', 'NN_Test_time', 'DT_Test_time', 'SVC_Test_time', 'LR_Test_time', 'Extra_Test_time','Random_Test_time',
         'Memory'])


""" parameters """
if __name__ == "__main__":

    rel = False

    dataset = sys.argv[1]#'BGS'#'BGS'
    depth = int(sys.argv[2]) #'1 or 2 or 3'
    method = sys.argv[3]   #'INK' or 'RDF2Vec'

    count = sys.argv[4]
    level = sys.argv[5]
    float_rpr = sys.argv[6]

    dir_kb = './data_node_class/'+dataset
    files = {'AIFB':'aifb_stripped.nt','BGS':'bgs_stripped_2.nt','MUTAG':'mutag.owl','AM':'am_stripped.nt','FOOTBALL':'FOOTBALL-2.nt'}
    file = files[dataset]#'AIFB.n3'#'rdf_am-data.ttl'
    print(file)
    formats = {'AIFB':'nt','BGS':'ttl','MUTAG':'xml','AM':'ttl','FOOTBALL':'n3'}

    format = formats[dataset]

    train = './data_node_class/'+dataset+'/'+dataset+'_train.tsv'
    test = './data_node_class/'+dataset+'/'+dataset+'_test.tsv'
    #train = 'mela/train.csv'
    #test = 'mela_tes'

    #每个baseline是否excludes_dict都相同？
    excludes_dict = {'AIFB':['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation'],'BGS':['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'],'MUTAG':['http://dl-learner.org/carcinogenesis#isMutagenic'],'AM':['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material'],'FOOTBALL':[]}

    excludes = excludes_dict[dataset]#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']

    labels_dict = {'AIFB':'label_affiliation','BGS':'label_lithogenesis','MUTAG':'label_mutagenic','AM':'label_cateogory','FOOTBALL':'result'}
    label_name = labels_dict[dataset]#'label_lithogenesis'#'label_affiliation'#'label_cateogory'#'label_lithogenesis'#'label_mutagenic'#'label_affiliation'

    items_dict = {'AIFB':'person','BGS':'rock','MUTAG':'bond','AM':'proxy','FOOTBALL':'match_id'}
    items_name = items_dict[dataset]#'rock'#'person'#'proxy'#'rock'#'bond'#'person'


    df_train = pd.read_csv(train, delimiter='\t')
    df_test = pd.read_csv(test, delimiter='\t')

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])


    pos_file = set(['<' + x + '>' for x in data[items_name].values])

    connector = RDFLibConnector(dir_kb+'/'+file, format)


    ## INK exrtact

    extractor = InkExtractor(connector, verbose=True)
    t0 = time.time()
    X_train, _ = extractor.create_dataset(depth, pos_file, set(), excludes, 4)
    extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=False)

    print("Create time : ", time.time() - t0)

    df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
    df_data.index = [x[1:-1] for x in extracted_data[1]]
    df_data.columns = extracted_data[2]

    df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]  # df_data.loc[[df_train['proxy']],:]
    df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]  # df_data.loc[[df_test['proxy']],:]

    df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
    df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

    # 定义输出目录
    output_dir = '../../IBIDI/datasets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存训练数据文件
    train_file = os.path.join(output_dir, f'{dataset}_train.csv')
    df_train_extr.to_csv(train_file, index=True, header=True, index_label='index')

    # 保存测试数据文件
    test_file = os.path.join(output_dir, f'{dataset}_test.csv')
    df_test_extr.to_csv(test_file, index=True, header=True, index_label='index')

    # 定义文件夹名
    folder_name = 'Revised_Embedding'

    # 创建文件夹（如果不存在）
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


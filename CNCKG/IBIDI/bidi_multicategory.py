import time
import os
from pathlib import Path

import pandas as pd
import random
import math
import os
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import argparse
from pathlib import Path

# print len(traindata.loc[traindata.sign==0,:])
# print len(traindata.loc[traindata.sign==1,:])
# 计算 0 - trainnumber 第一个大于等于 number的数组下标
def birary_search(number, d, train_number):
    pre_sum = [0] * train_number
    pre_sum[0] = d[0]
    #Calculate prefixes and arrays
    for i in range(1, train_number):
        pre_sum[i] = pre_sum[i - 1] + d[i]

    l = 0
    r = train_number - 1
    while l < r:
        mid = int((l + r) / 2)
        if pre_sum[mid] >= number:
            r = mid
        else:
            l = mid + 1
    return r

#The exact position of number in the array
def brute_search(number, d, train_number):
    for k in range(0, train_number):
        number -= d[k]
        if (number <= 0):
            return k

#Calculate the variance of d
def cal_var(d):
    number = len(d)
    sum1 = 0.0
    for i in range(0, number):
        sum1 = sum1 + d[i]
    sum1 = 1.0 * sum1 / number

    sum2 = 0.0
    for i in range(0, number):
        sum2 = sum2 + (sum1 - d[i]) * (sum1 - d[i])
    sum2 = 1.0 * sum2 / (number - 1)
    return sum2


# How many decision trees are integrated into a flag
def adadifficult(traindata, testdata, flag, pre, classes):
    feature_x = (list(traindata.columns.values))[:-4]
    train_number = len(traindata)
    test_number = len(testdata)
    train_diff = traindata['difficult'].copy()
    test_diff = testdata['difficult'].copy()

    train_loss = traindata['difficult'].copy()
    test_loss = testdata['difficult'].copy()


    t0 = time.time()
    cnt = (int)(train_number * 0.666)   #cnt是？
    clf = []
    a = []
    #Each instance has the same initial weight
    d = [1.0 / train_number] * train_number
    #How many base classifier should be integrated
    for i in range(0, flag):
        # clf.append(tree.DecisionTreeClassifier())
        clf.append(LogisticRegression())
        a.append(0.0)

    tot_loss = 0
    #Misclassification loss l (diffi) of instance xi
    for i in range(0, train_number):

        train_loss[i] = (1 / (1 + math.exp(pre * train_diff[i])))
    max_loss1 = max(train_loss)
    min_loss1 = min(train_loss)

    for i in range(0, test_number):
        test_loss[i] = (1 / (1 + math.exp(pre * test_diff[i])))

    max_loss = max(test_loss)
    min_loss = min(test_loss)

    max_loss = max(max_loss, max_loss1)
    min_loss = min(min_loss, min_loss1)


    #normalization
    for i in range(0, train_number):
        train_loss[i] = (train_loss[i] - min_loss) / (max_loss - min_loss)
        tot_loss = tot_loss + train_loss[i]

    for i in range(0, test_number):
        test_loss[i] = (test_loss[i] - min_loss) / (max_loss - min_loss)

    for i in range(0, train_number):
        d[i] = train_loss[i] / tot_loss     #d是干什么的？--->d才是归一化的结果   即初始化权重，论文中wt,l

    # print 'var:',cal_var(train_loss)
    # Start learning for each base classifier
    for i in range(0, flag):

        ex = []
        sum_d = sum(d)
        '''
        data_list=[]
        for j in range(0,train_number):
            data_list.append({'id': j, 'value':d[j]})

        data_list.sort(key=lambda obj:obj.get('value'),reverse=True)
        for j in range(1, cnt + 1):
            ex.append(data_list[j]['id'])
        '''
        # cnt is used to select the number of samples required for one iteration of the input AdaBoost algorithm
        for j in range(1, cnt + 1):
            s1 = random.uniform(0, sum_d)   #Select the sample index in the sample weight array by selecting a random number

            r1 = birary_search(s1, d, train_number)     # instances with low difficulty that were not learned are more likely to be selected in the next round
            ex.append(r1)

        temp1 = traindata.iloc[ex, :]   # 转换为行数据


        x = temp1.loc[:, feature_x]   #mutag加上.values
        y = temp1.loc[:, ['sign']]

        y = y.astype(int)
        if exp_name == 'movies':
            if item_name in x.columns:
                x_n = x.drop(columns=[item_name], axis=1).values
            else:
                x_n = x.values
            # x_n = x.drop(columns=[item_name], axis =1).values
            # x_n = x.values
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
            x_n = imputer.fit_transform(x_n)
            clf[i] = clf[i].fit(x_n, y.values.ravel())
            res = clf[i].predict(x_n)
        else:
            clf[i] = clf[i].fit(x, y.values.ravel())
            res = clf[i].predict(x)

        a[i] = 0

        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for j in range(0, cnt):
            indicate_res = 1 if traindata['sign'][j] == res[j] else -1
            a[i] = a[i] + d[j] * indicate_res * train_loss[j]



        a[i] = 0.5 * math.log((1 + a[i]) / (1 - a[i]), math.e) + math.log(len(classes)-1, math.e)
        z = 0
        # calculate z in the formula
        for j in range(0, cnt):
            indicate_res = 1 if traindata['sign'][j] == res[j] else -1
            z = z + d[j] * math.exp(-a[i] * indicate_res * train_loss[j])
        # Calculate the t-value for the next iteration
        for j in range(0, cnt):
            indicate_res = 1 if traindata['sign'][j] == res[j] else -1
            d[j] = d[j] * math.exp(-a[i] * indicate_res * train_loss[j]) / z
    print("BIDI_train_time:"+str(time.time()-t0))

    t0 = time.time()
    cou = 0
    sum_loss = 0
    test_sign = testdata['sign']             #y
    testdata = testdata.loc[:, feature_x]    #x
    if exp_name == 'movies':
        if item_name in testdata.columns:
            testdata = testdata.drop(columns=[item_name], axis=1).values
        else:
            print(f"Column {item_name} does not exist, skip deletion operation.")
            testdata = testdata.values
        # testdata = testdata.drop(columns=[item_name], axis=1).values
        # testdata = testdata.values
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
        testdata = imputer.fit_transform(testdata)
    res = []

    for j in range(0, test_number):
        best_score = 0
        best_class = ''
        for class_name in classes:
            result = 0
            for i in range(flag):
                if exp_name == 'movies':
                    # Retrieve the jth sample from testdata
                    sample = testdata[j, :]  # 使用NumPy的索引方式
                    # predict
                    predicted_class = clf[i].predict(sample.reshape(1, -1))[0]  # Ensure that the input is a two-dimensional array
                    indicate_res = 1 if class_name == predicted_class else 0
                else:
                    indicate_res = 1 if class_name == clf[i].predict(testdata.iloc[[j], :]) else 0
                result = result + a[i] * indicate_res
            if result > best_score:
                best_score = result
                best_class = class_name
        if(best_class == test_sign[j]):
            result = 1
            res.append(result)
            r = 1
        else:
            result = -1
            res.append(result)
            r = -1
            cou = cou + 1
            sum_loss = sum_loss + test_loss[j]

    print("BIDI_test_time:" + str(time.time() - t0))
    print("test_num:",test_number)
    print("test_cou:",cou)
    return cou, sum_loss, test_loss, res
          # a       b         c       d


avg_error = 0.0
avg_loss = 0.0

# a :  cou ，  Number of classification errors represented
# d :  The result of the res and adadi functions, list (if each component>0, it represents a successful prediction)
# e :  The results of the res and adboost functions are the same as above
#K: k-th iteration
def save(data2, data4, a, d, e, c, flag, k):
    data2['predict'] = d

    data2['loss'] = c
    # data4['loss'] = c
    data2['detail'] = 'correct'

    for j in range(0, len(data2)):
        if data2['predict'][j] > 0 :
            data2['detail'][j] = 'correct'
        else:
            data2['detail'][j] = 'error'


    if flag == 0:
        data2 = data2.loc[data2.detail == 'error', :]


    ########################################################################
    # 将error　重命名为　0 将correct 重命名为　1
    data2.loc[data2.detail == 'error', 'detail'] = 0
    data2.loc[data2.detail == 'correct', 'detail'] = 1

    #######################################################################

    myflod = "output_new/" + args.dataset + "/BIDI_output/"+ diff_name+"/exp_" + str(exp_num)
    if not os.path.exists(myflod):
        os.makedirs(myflod)


    path = myflod + "/Difficult_result_" + str(a) + "_" + str(k) + ".csv"  ######
    data_bidi = data2[[item_name, 'sign', 'difficult', 'predict', 'loss', 'detail']]
    data_bidi.to_csv(path)

    ###########################################################################
    # 返回统计量：准确率，错误样本难度均值
    data2_accuracy_return = len(data2.loc[data2.predict > 0]) / len(data2)
    if len(data2.loc[data2.predict <= 0 ]) == 0:
        data2_difficult_return = 0
    else:
        data2_difficult_return = data2.loc[data2.predict <= 0]['difficult'].sum() / len(data2.loc[data2.predict <=0])

    return data2_accuracy_return, data2_difficult_return
    ###########################################################################

def drop_items(exp_name, data):
    if (exp_name == 'mutag'):
        return data.drop(['bond'], axis=1)
    if (exp_name == 'aifb'):
        return data.drop(['person'], axis=1)
    if (exp_name == 'bgs'):
        return data.drop(['rock'], axis=1)
    if (exp_name == 'am'):
        return data.drop(['proxy'], axis=1)


def check_a(st, en, step, traindata, k, flag, kk):
    difficult_sum1 = 0
    difficult_sum2 = 0

    if k == 1:
        while st <= en:
            print(st)
            num_classes = traindata['sign'].nunique()
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
            data1 = traindata.loc[traindata.id % 3 != 0, :].copy()
            data2 = traindata.loc[traindata.id % 3 == 0, :].copy()
            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)
            data2['id'] = data2.index
            data1 = imputer.fit_transform(data1)
            data2 = imputer.fit_transform(data2)
            for i in range(0, kk):
                t_data = data2.loc[data2.id % kk == i, :].copy().reset_index(drop=True)
                data4 = t_data.copy()
                a, b, c, d = adadifficult(data1.copy(), t_data, 12, st, classes)
                difficult_sum1 = difficult_sum1 + a
                difficult_sum2 = difficult_sum2 + b
                print(a, b)
                e = 0
                save(t_data, data4, st, d, e, c, flag, i + 1)
            st = st + step
    if k != 1:
        while st <= en:
            # print(st)
            save_tuple = []
            cou_bidi = 0
            for i in range(0, k):
                imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
                data1 = traindata.loc[traindata.id % k != i, :].copy()
                data2 = traindata.loc[traindata.id % k == i, :].copy()
                data1 = data1.reset_index(drop=True)
                data2 = data2.reset_index(drop=True)
                data2['id'] = data2.index

                data3 = data1.copy()
                data4 = data2.copy()

                a, b, c, d = adadifficult(data1, data2, 100, st, classes)
                cou_bidi = cou_bidi + a
                difficult_sum1 = difficult_sum1 + a
                difficult_sum2 = difficult_sum2 + b
                e = 0
                save_tuple.append(save(data2, data4, st, d, e, c, flag, i + 1))


            ##################################################################
            # Output statistics: accuracy, average difficulty of incorrect samples
            # print("====================" + exp_name + "一次十折验证的结果====================")
            print("BIDI准确率均值：", np.mean([item[0] for item in save_tuple]))
            print("BIDI准确率方差：", np.var([item[0] for item in save_tuple]))
            print("cou_bidi:" , cou_bidi)
            ##################################################################
            st = st + step


def save_b(data, name, k, flag, kk, a):
    global avg_error
    global avg_loss
    sum1 = 0
    sum2 = 0
    data['detail'] = 'correct'
    for i in range(0, len(data)):
        sum2 = sum2 + 1
        if data['predict'][i] == data['sign'][i]:
            data['detail'][i] = 'correct'
        else:
            data['detail'][i] = 'error'
            sum1 = sum1 + 1
            avg_loss = avg_loss + data['loss'][i]

    avg_error += sum1
    avg_loss /= max(sum1, 1e-16)
    if flag == 0:
        data = data.loc[data.detail == 'error', :]

    ##############################################################
    data.loc[data.detail == 'error', 'detail'] = 0
    data.loc[data.detail == 'correct', 'detail'] = 1
    ##############################################################

    myflod = "output/" + exp_name + "/" + name + "_output/"+ diff_name+"/exp_" + str(exp_num) + "/"
    if not os.path.exists(myflod):
        os.makedirs(myflod)
    path = myflod + name + str(a) + '_' + str(k) + '_' + str(kk) + '.csv'  ######


    data.to_csv(path)
    # print(avg_error, avg_loss)

    ###########################################################################

    data_accuracy_return = len(data.loc[data.predict == data.sign]) / len(data)
    data_difficult_return = data.loc[data.predict != data.sign]['difficult'].sum() / len(data.loc[data.predict != data.sign])
    return data_accuracy_return, data_difficult_return
    ###########################################################################


def check__b(name, traindata, k, flag, pre, kk):
    global avg_error
    global avg_loss
    train_number = len(traindata)
    traindata['loss'] = 0.0
    train_diff = traindata['difficult'].copy()
    train_loss = traindata['difficult'].copy()
    for i in range(0, train_number):
        traindata['loss'][i] = (1 / (1 + math.exp(pre * train_diff[i])))
        # print traindata['loss'][i]
    max_loss = max(train_loss)
    min_loss = min(train_loss)
    for i in range(0, train_number):
        traindata['loss'][i] = (traindata['loss'][i] - min_loss) / (max_loss - min_loss)

    feature_x = (list(traindata.columns.values))[:-4]
    traindata['sign'] = traindata.sign.map(lambda x: -1 if x == 0 else 1)  ######
    clf = ""
    if name == 'SVM':
        clf = svm.LinearSVC()
    if name == 'RF':
        clf = RandomForestClassifier(n_estimators=10) # 适用于spambase、magic
    if name == 'DT':
        clf = tree.DecisionTreeClassifier()
    if name == 'NB':
        clf = GaussianNB()
    if name == 'KNN':
        clf = neighbors.KNeighborsClassifier()
    if name == 'BP':
        clf = MLPClassifier()
    if k == 1:
        print(traindata['sign'])
        data1 = traindata.loc[traindata.id % 3 != 0, :].copy()
        data2 = traindata.loc[traindata.id % 3 == 0, :].copy()
        data1 = data1.reset_index(drop=True)
        data2 = data2.reset_index(drop=True)
        data2['id'] = data2.index
        for i in range(0, kk):
            t_data = data2.loc[data2.id % kk == i, :].copy().reset_index(drop=True)
            clf = clf.fit(data1[feature_x], data1.sign)
            res = clf.predict(t_data[feature_x])
            res = pd.DataFrame(res)
            res.columns = ['predict']
            res = res.reset_index(drop=True)
            t_data['predict'] = res.predict.map(lambda x: x)
            save_b(t_data, name, k, flag, i + 1, pre)
    else:
        save_tuple = []
        for i in range(0, k):
            data1 = traindata.loc[traindata.id % k != i, :].copy()
            data2 = traindata.loc[traindata.id % k == i, :].copy()
            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)
            clf = clf.fit(data1[feature_x], data1.sign)
            res = clf.predict(data2[feature_x])
            res = pd.DataFrame(res)
            res.columns = ['predict']
            res = res.reset_index(drop=True)
            data2['predict'] = res.predict.map(lambda x: x)
            save_tuple.append(save_b(data2, name, k, flag, i + 1, pre))

        ##################################################################
        # print("====================" + exp_name + "一次十折验证的结果====================")
        print(name + "准确率均值：", np.mean([item[0] for item in save_tuple]))
        print(name + "准确率方差：", np.var([item[0] for item in save_tuple]))
        # print(name + "错分样本难度均值：", np.mean([item[1] for item in save_tuple]))
        # print("========================================================================")
        # print()
        ##################################################################

    # print(name, avg_loss / k, avg_error / k)


def load_dataset(base_name, data_dir):
    use_tff = base_name.endswith("_TFF")
    clean_name = base_name.replace("_TFF", "") if use_tff else base_name

    # 构建文件名模板
    file_template = f"{clean_name}_train_TFF.csv" if use_tff else f"{clean_name}_train.csv"
    train_path = Path(data_dir) / file_template
    test_path = Path(data_dir) / file_template.replace("_train", "_test")

    # 验证文件存在性
    if not train_path.exists():
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"测试文件不存在: {test_path}")

    # 加载并合并数据
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return pd.concat([df_train, df_test], ignore_index=True)


def get_exp_name(dataset_name):
    dataset_mapping = {
        'aifb': 'person',
        'bgs': 'rock',
        'mutag': 'bond',
        'am': 'proxy',
        'movies': 'DBpedia_URL'
    }

    base_name = dataset_name.split('_')[0].rstrip('1234567890').lower()

    if base_name.startswith('movie'):
        return base_name


    for prefix in dataset_mapping:
        if base_name.startswith(prefix):
            return base_name

    raise ValueError(f"未知数据集前缀: {dataset_name}")



parser = argparse.ArgumentParser(description='运行机器学习实验')
parser.add_argument('--dataset', type=str, required=True,
                    help='数据集文件名（不带路径）')
parser.add_argument('--exp_num', type=int, required=True,
                    help='交叉验证折数')
parser.add_argument('--exp_type', type=str, required=True,
                    help='实验类型（0-6）')
parser.add_argument('--param_a', type=float, required=True,
                    help='BIDI算法参数a')
parser.add_argument('--data_dir', type=str, default='datasets',
                    help='数据集存放目录')
parser.add_argument('--log_dir', type=str, default='experiment_logs',
                    help='日志输出目录')
parser.add_argument('--clf', type=int, default=10, help='num of base classifer')
args = parser.parse_args()

# 模拟原代码的sys.argv参数顺序
sys.argv = [
    sys.argv[0],
    args.dataset,  # exp_name
    str(args.exp_num),  # exp_num
    args.exp_type,  # exp_type
    str(args.param_a)  # param_a
]


# 原代码的初始化逻辑
    # ---------------------------
print(f"=== 实验配置 ===")
print(f"数据集: {args.dataset}")
print(f"交叉验证折数: {args.exp_num}")
print(f"实验类型: {args.exp_type}")
print(f"参数a: {args.param_a}")


exp_name = sys.argv[1]                         # 需要做的实验名称
exp_num = int(sys.argv[2])                     # 第几次交叉验证

# 设置系统参数模拟原代码的sys.argv
sys.argv = [
    sys.argv[0],
    Path(args.dataset).stem,  # 移除文件扩展名
    str(args.exp_num),
    args.exp_type,
    str(args.param_a)
]

os.makedirs(args.log_dir, exist_ok=True)

# 重定向输出到日志文件
# log_path = os.path.join(
#     args.log_dir,
#     f"{Path(args.dataset).stem}_exp{args.exp_num}_type{args.exp_type}_a{args.param_a}.log"
# )
# sys.stdout = open(log_path, 'w')
# sys.stderr = sys.stdout

print(f"=== 开始实验 {time.ctime()} ===")
print(f"参数: {vars(args)}")

try:
    # 原代码的主要执行逻辑
    # （这里需要将原代码的sys.argv逻辑替换为参数化处理）
    # --------------------------------------------
    # 原代码的全局设置
    exp_name = get_exp_name(args.dataset)
    print(f"解析得到exp_name: {exp_name}")
    exp_num = int(sys.argv[2])

    # 加载数据（修改后的智能加载方法）
    df = load_dataset(args.dataset, args.data_dir)


    warnings.filterwarnings("ignore")
    item_dict = {'aifb':'person','bgs':'rock','mutag':'bond','am':'proxy', 'movies': 'DBpedia_URL'}
    item_name = item_dict[exp_name]
    diff_name = None
    if exp_name == 'aifb' :
        diff_name = 'aifb_centrality_nodes'
        df_diff = pd.read_csv(os.path.join(args.data_dir, diff_name + ".csv"))

    elif exp_name == 'bgs' :
        diff_name = 'bgs_centrality_nodes'
        df_diff = pd.read_csv(os.path.join(args.data_dir, diff_name + ".csv"))
    elif exp_name == 'mutag' :
        diff_name = 'mutag_centrality_nodes'
        df_diff = pd.read_csv(os.path.join(args.data_dir, diff_name + ".csv"))
    elif exp_name == 'am' :
        diff_name = 'am_centrality_nodes'
        df_diff = pd.read_csv(os.path.join(args.data_dir, diff_name + ".csv"))
    elif exp_name == 'movies' :
        diff_name = 'movies_centrality_nodes'
        df_diff = pd.read_csv(os.path.join(args.data_dir, diff_name + ".csv"))

    # 对 difficult 列进行归一化处理
    d_min = df_diff['difficult'].min()
    d_max = df_diff['difficult'].max()

    df_diff['difficult'] = (df_diff['difficult'] - d_min) / (d_max - d_min)

    df = df.merge(df_diff[[item_name, 'difficult']], on=item_name, how='left')
    traindata = df
    if (exp_name == 'mutag' or 'aifb' or 'bgs' or 'am' or 'movies'):
        traindata = traindata.rename(columns={'label': 'sign'})
        traindata = traindata.rename(columns={'Unnamed: 0': 'index'})

    classes = list(traindata['sign'].unique())

    traindata = traindata.sort_values(by=['difficult'])  # 按照difficult列排序
    traindata = traindata.reset_index(drop=True)  # 去除原本的index列，按照difficult列排序重新加上Index


    ###############################shuffle的代码#################################################
    #   @Author  : hancheng wang
    #   @Time    : 2020-01-18
    #   @Desc    : 为了保证五次十折交叉验证输入数据的多样性，在不影响难度分布的前提下，我十个十个打乱数据，之后再十折拆分数据
    #              exp_num ： 随机种子，只要种子变了，就会产生不同的打乱结果，如果种子不变，打乱结果不变
    ############################################################################################
    traindata_index = []
    np.random.seed()
    # 十个十个shuffle数据
    for i in range(0, len(traindata), 10):
        traindata_index.extend(np.random.permutation(list(range(i, min(i + 10, len(traindata))))))  # 每10个数据shuffle
    traindata['traindata_index'] = traindata_index
    traindata = traindata.sort_values(by=['traindata_index'])
    traindata = traindata.reset_index(drop=True)

    traindata = traindata.drop(['traindata_index'], axis=1)

    ############################################################################################

    traindata['id'] = traindata.index
    train_number = len(traindata)

    avg_error = 0.0
    avg_loss = 0.0

    if sys.argv[3] == "0":
        check_a(float(sys.argv[4]), float(sys.argv[4]), 0.5, traindata.copy(), args.clf, 1, 30)  ######
    elif sys.argv[3] == "1":
        check__b('SVM', traindata.copy(), 10, 1, 1, 30)
    elif sys.argv[3] == "2":
        check__b('RF', traindata.copy(), 10, 1, 1, 30)
    elif sys.argv[3] == "3":
        check__b('DT', traindata.copy(), 10, 1, 1, 30)
    elif sys.argv[3] == "4":
        check__b('NB', traindata.copy(), 10, 1, 1, 30)
    elif sys.argv[3] == "5":
        check__b('KNN', traindata.copy(), 10, 1, 1, 30)
    elif sys.argv[3] == "6":
        check__b('BP', traindata.copy(), 10, 1, 1, 30)
    else:
        print("当前实验类型错误")

    # 原代码的后续处理...
    # --------------------------------------------

    print("=== 实验成功完成 ===")
except Exception as e:
    print(f"!!! 实验失败: {str(e)}")
    raise








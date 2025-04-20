from ink.base.connectors import RDFLibConnector
from ink.base.structure import InkExtractor
from ink.miner.rulemining import RuleSetMiner
import pandas as pd

if __name__ == '__main__':
    ##创建connector
    connector = RDFLibConnector('ink/datasets/animals.owl', 'xml')
    connector.query("Select ?s where {?s a <http://dl-learner.org/benchmark/dataset/animals/T-Rex>.}") #通过

    ##创建extractor
    prefix = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf#",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs#",
        "http://www.w3.org/2002/07/owl#": "owl#",
        "http://dl-learner.org/benchmark/dataset/animals/": "animals/"
    }
    extractor = InkExtractor(connector, prefixes=prefix)

    ##创建rule miner
    miner = RuleSetMiner(chains=100, max_len_rule_set=3, forest_size=10)

    ##rule mining
    pos = set(["http://dl-learner.org/benchmark/dataset/animals#dog01",
               "http://dl-learner.org/benchmark/dataset/animals#dolphin01",
               "http://dl-learner.org/benchmark/dataset/animals#platypus01",
               "http://dl-learner.org/benchmark/dataset/animals#bat01"])

    neg = set(["http://dl-learner.org/benchmark/dataset/animals#trout01",
               "http://dl-learner.org/benchmark/dataset/animals#herring01",
               "http://dl-learner.org/benchmark/dataset/animals#shark01",
               "http://dl-learner.org/benchmark/dataset/animals#lizard01",
               "http://dl-learner.org/benchmark/dataset/animals#croco01",
               "http://dl-learner.org/benchmark/dataset/animals#trex01",
               "http://dl-learner.org/benchmark/dataset/animals#turtle01",
               "http://dl-learner.org/benchmark/dataset/animals#eagle01",
               "http://dl-learner.org/benchmark/dataset/animals#ostrich01",
               "http://dl-learner.org/benchmark/dataset/animals#penguin01"])
    ##从邻居中迭代提取信息
    X_train, y_train = extractor.create_dataset(4, pos, neg, jobs=4)
    print("animal:", X_train[0][0])
    # for x in X_train[0][1]:
    #     print("relation:", x)
    #     print("objects:", X_train[0][1][x])

    ##构建INK表示
    X_train = extractor.fit_transform(X_train, counts=True, levels=True)
    df_train = pd.DataFrame.sparse.from_spmatrix(X_train[0], index=X_train[1], columns=X_train[2])
    print(df_train.head())
    print(df_train)
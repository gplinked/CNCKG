import time

from ink.base.structure import InkExtractor
from ink.base.connectors import RDFLibConnector
import sys
from sklearn import preprocessing
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class INKModel:

    def __init__(self, connector):
        self.connector = connector
        self.kg = self.connector.g  # Assuming the graph is stored in self.g
        self.verbose = True  # Set to True for verbose output
        self.prefixes = []
        self.extract_inverse = False

    # 原始方法，用于构建初始数据集，包括正例和负例节点的邻居信息。
    def create_dataset(self, depth=4, pos=None, neg=None, skip_list=None, jobs=1):
        if skip_list is None:
            skip_list = []

        def _acquire_set(val):
            v_set = set()
            if isinstance(val, str):
                res = self.connector.query(val)
                for s in res:
                    v_set.add(s['s']['value'])
            else:
                if val is not None and not isinstance(val, set):
                    with open(val) as file:
                        v_set = set(['<' + r.rstrip("\n") + '>' for r in file.readlines()])
                else:
                    if isinstance(val, set):
                        v_set = val
            return v_set

        pos_set = _acquire_set(pos)
        neg_set = _acquire_set(neg)

        if self.verbose:
            print("#Process: get neighbourhood")

        all_noi = list(pos_set.union(neg_set))
        noi_neighbours = self.extract_neighborhoods(all_noi, depth, skip_list, verbose=self.verbose, jobs=jobs)
        # update order
        all_noi = [n[0] for n in noi_neighbours]

        a = []
        if len(pos_set) > 0 or len(neg_set) > 0:
            for v in all_noi:
                if v in pos_set:
                    a.append(1)
                else:
                    a.append(0)

        return noi_neighbours, np.array(a)

    def _replace_pref(self, r):
        """
        Internal function to strip the prefix from the given URI
        :param r: URI string
        :type r: str
        :return: URI string, with the prefix replaced.
        :rtype str
        """
        for x in self.prefixes:
            r = r.replace(x, self.prefixes[x])
            #return r
        return r
    # 提取给定节点列表的邻居信息，并以指定深度查询。
    def extract_neighborhoods(self, data, depth, skip_list=None, verbose=False, jobs=1):

        if skip_list is None:
            skip_list = []

        seq = [(r, depth, skip_list) for r in data]
        if jobs > 1:
            with Pool(jobs) as pool:
                res = list(tqdm(pool.imap_unordered(self._create_neighbour_paths, seq, chunksize=100),
                                total=len(data), disable=not verbose))
                pool.terminate()
                pool.close()
                pool.join()
        else:
            res = []
            for s in tqdm(seq, disable=not verbose, total=len(data)):
                res.append(self._create_neighbour_paths(s))
        return res

    # 辅助方法，用于递归提取节点的邻居路径。
    def _create_neighbour_paths(self, t):

        noi, depth, avoid_lst = t
        value = noi, ""
        total_parts = {}
        all_done = []
        res = self._define_neighborhood(value, depth, avoid_lst, total_parts, all_done)
        return noi, res

    def _define_neighborhood(self, value, depth, avoid_lst, total_parts, all_done):
        n_e, prop = value
        if depth > 0 and n_e not in all_done:
            res = self.neighborhood_request(n_e)
            next_noi = []
            for row in res:
                p = self._replace_pref(row['p']['value'])
                os = row['o']['value']
                if 'dt' in row:
                    dt = True
                else:
                    dt = False

                if not dt:
                    os = os.split(' ')
                else:
                    os = [os]

                for o in os:
                    if p not in avoid_lst and o not in avoid_lst:
                        if not dt:
                            if o.startswith('bnode'):
                                if prop == "":
                                    next_noi.append(('<_:' + o + '>', p))
                                else:
                                    next_noi.append(('<_:' + o + '>', prop + '.' + p))
                            else:
                                if prop == "":
                                    next_noi.append(('<' + o + '>', p))
                                    if p not in total_parts:
                                        total_parts[p] = list()
                                    total_parts[p].append(self._replace_pref(o))
                                else:
                                    next_noi.append(('<' + o + '>', prop + '.' + p))
                                    if prop + "." + p not in total_parts:
                                        total_parts[prop + "." + p] = list()
                                    total_parts[prop + "." + p].append(self._replace_pref(o))
                        else:
                            if prop == "":
                                if p not in total_parts:
                                    total_parts[p] = list()
                                total_parts[p].append(self._replace_pref(o))
                            else:
                                if prop + "." + p not in total_parts:
                                    total_parts[prop + "." + p] = list()
                                total_parts[prop + "." + p].append(self._replace_pref(o))
            if depth - 1 > 0:
                [total_parts.update(self._define_neighborhood(value, depth - 1, avoid_lst, total_parts, all_done))
                 for value in next_noi]
            return total_parts

    def neighborhood_request(self, noi):
        """
        Function to make a neighborhood request of a certain instance.

        :param noi: URI of Node Of Interest (noi).
        :type noi: str
        :return: Dictionary with all values specified as in the connector string.
        :rtype: dict
        """
        try:
            if noi[0] == '<':
                noi = noi[1:]
            if noi[-1] == '>':
                noi = noi[:-1]

            q = 'SELECT ?p ?o ?dt WHERE { BIND( IRI("' + noi + '") AS ?s ) ?s ?p ?o. BIND (datatype(?o) AS ?dt) }'
            res = self.connector.query(q)
            if True:
                q = 'SELECT ?p ?o ?dt ?dt WHERE { BIND( IRI("' + noi + '") AS ?s ) ?o ?p ?s. BIND (datatype(?o) AS ?dt) }'
                res += self.connector.inv_query(q)
            #q = 'SELECT ?p ?o WHERE { <'+noi+'> ?p ?o. }'
            return res
        except Exception as e:
            #print(e)
            return []

    def add_node_and_update(self, new_node, depth, avoid_lst=None):
        """
        添加一个新节点并更新知识图谱表示
        """
        if avoid_lst is None:
            avoid_lst = []

        # 插入新节点到知识图谱中，使用 RDFLibConnector 类的 add_entity 方法
        triples = self.connector.get_triples(new_node)  # 假设新节点的三元组已获取
        self.connector.add_entity(new_node, triples)

        # 提取新节点的邻居信息并更新二进制表示
        new_neigh = self.extract_neighborhoods([new_node], depth, avoid_lst, verbose=self.verbose)
        self.update_existing_nodes(new_node, depth, avoid_lst)

        return new_neigh

    def update_existing_nodes(self, new_node, depth, avoid_lst=None):
        """
        更新受新节点影响的已有节点
        """
        if avoid_lst is None:
            avoid_lst = []

        affected_nodes = self.connector.get_affected_nodes(new_node, depth)  # 假设该方法存在
        affected_neigh = self.extract_neighborhoods(affected_nodes, depth, avoid_lst, verbose=self.verbose)
        return affected_neigh

    def get_affected_nodes(self, new_node, depth):
        """
        Get all nodes affected by the addition of a new node within a given depth.
        :param new_node: The new node added to the knowledge graph.
        :type new_node: str
        :param depth: The depth to search for affected nodes.
        :type depth: int
        :return: List of affected nodes.
        :rtype: list
        """
        visited = set()
        queue = [(URIRef(new_node), 0)]
        affected_nodes = set()

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth > depth:
                break
            if current_node not in visited:
                visited.add(current_node)
                affected_nodes.add(current_node)
                # Traverse neighbors
                for p, o in self.g.predicate_objects(subject=current_node):
                    if o not in visited:
                        queue.append((o, current_depth + 1))
                for s, p in self.g.subject_predicates(object=current_node):
                    if s not in visited:
                        queue.append((s, current_depth + 1))

        return list(affected_nodes)

def compare_dataframes(df1, df2, items_name):
    """
    Compare two dataframes and print the similarity and differences for each row.
    :param df1: First dataframe
    :param df2: Second dataframe
    """
    # Ensure the dataframes have the same columns
    print("df1.columns:", len(df1.columns))
    print("df2.columns:", len(df2.columns))

    assert list(df1.columns) == list(df2.columns), "Dataframes must have the same columns"

    # Sort dataframes by items_name column
    df1 = df1.sort_values(by=items_name).reset_index(drop=True)
    df2 = df2.sort_values(by=items_name).reset_index(drop=True)

    # Compute cosine similarity
    similarity = cosine_similarity(df1.drop(columns=['label', items_name]), df2.drop(columns=['label', items_name]))
    diagonal_similarity = np.diag(similarity)
    avg_diagonal_similarity = np.mean(diagonal_similarity)
    print(f'Average similarity: {avg_diagonal_similarity}')

    # Find differences for each row
    for i, (index1, row1) in enumerate(df1.iterrows()):
        index2 = df2.index[i]
        row2 = df2.loc[index2]

        differences = []
        for col in df1.columns:
            if col not in ['label', items_name] and row1[col] != row2[col]:
                differences.append((col, row1[col], row2[col]))

        if differences:
            #print(f'Differences for row {index1}:')
            for diff in differences:
                print(f'  Column: {diff[0]}, df1: {diff[1]}, df2: {diff[2]}')



def compare_dataframes_columns(df1, df2):
    """
    Compare the columns of two dataframes and print the differences.
    :param df1: First dataframe
    :param df2: Second dataframe
    """
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)

    print("columns_df1:",len(columns_df1) ,"columns_df2", len(columns_df2))
    # Columns in df1 but not in df2
    only_in_df1 = columns_df1 - columns_df2
    # Columns in df2 but not in df1
    only_in_df2 = columns_df2 - columns_df1

    print("Columns only in df_train_extr:")
    for col in only_in_df1:
        print(f"  {col}")

    print("Columns only in df_train_extr_new_2:")
    for col in only_in_df2:
        print(f"  {col}")

def convert_to_pos_format(affected_nodes):
    pos_format_nodes = set()
    for node in affected_nodes:
        if isinstance(node, URIRef):
            pos_format_nodes.add(f'<{str(node)}>')
        elif isinstance(node, Literal):
            pos_format_nodes.add(f'"{str(node)}"')
    return pos_format_nodes

def check_missing_values(df, items_name):
    """
    Check for missing values in the dataframe and print the columns with missing values.
    :param df: Dataframe to check
    """
    missing_values = df.isna().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        print("Columns with missing values:")
        print(missing_columns)
    else:
        print("No missing values found.")


def compare_row_features(df1, df2, items_name):
    """
    Compare the row features of two dataframes and print the differences.
    :param df1: First dataframe
    :param df2: Second dataframe
    """
    for i, (index1, row1) in enumerate(df1.iterrows()):
        index2 = df2.index[i]
        row2 = df2.loc[index2]

        differences = []
        for col in df1.columns:
            if col not in ['label', items_name] and row1[col] != row2[col]:
                differences.append((col, row1[col], row2[col]))

        if differences:
            print(f'Differences for row {index1}:')
            # for diff in differences:
            #     print(f'  Column: {diff[0]}, df1: {diff[1]}, df2: {diff[2]}')
            print("different_len:",len(differences))

def main():
    dataset = sys.argv[1]  # 'BGS'#'BGS'
    depth = int(sys.argv[2])
    method = sys.argv[3]

    dir_kb = './data_node_class/' + dataset
    files = {'AIFB': 'aifb_stripped.nt', 'BGS': 'bgs_stripped_2.nt', 'MUTAG': 'mutag.owl', 'AM': 'am_stripped.nt',
             'FOOTBALL': 'FOOTBALL-2.nt'}
    file = files[dataset]  # 'AIFB.n3'#'rdf_am-data.ttl'
    print(file)
    formats = {'AIFB': 'nt', 'BGS': 'ttl', 'MUTAG': 'xml', 'AM': 'ttl', 'FOOTBALL': 'n3','movies':'nt'}

    format = formats[dataset]

    train = './data_node_class/' + dataset + '/' + dataset + '_train.tsv'
    test = './data_node_class/' + dataset + '/' + dataset + '_test.tsv'

    excludes_dict = {
        'AIFB': ['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation'],
        'BGS': ['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'],
        'MUTAG': ['http://dl-learner.org/carcinogenesis#isMutagenic'],
        'AM': ['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material'],
        'Movies': []
    }
    excludes = excludes_dict[dataset]

    labels_dict = {'AIFB': 'label_affiliation', 'BGS': 'label_lithogenesis', 'MUTAG': 'label_mutagenic',
                   'AM': 'label_cateogory', 'FOOTBALL': 'result'}
    label_name = labels_dict[dataset]

    items_dict = {'AIFB': 'person', 'BGS': 'rock', 'MUTAG': 'bond', 'AM': 'proxy', 'FOOTBALL': 'match_id'}
    items_name = items_dict[dataset]

    try:
        df_train = pd.read_csv(train, delimiter='\t', encoding='utf-8')
        df_test = pd.read_csv(test, delimiter='\t', encoding='utf-8')
    except UnicodeDecodeError:
        df_train = pd.read_csv(train, delimiter='\t', encoding='ISO-8859-1')
        df_test = pd.read_csv(test, delimiter='\t', encoding='ISO-8859-1')

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])

    t0 = time.time()

    connector = RDFLibConnector(dir_kb + '/' + file, format)
    extractor = InkExtractor(connector, verbose=True)
    #
    pos_file = set(['<' + x + '>' for x in data[items_name].values])
    #
    #
    X_train, _ = extractor.create_dataset(depth, pos_file, set(), excludes, jobs=4)
    extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=False)


    df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
    df_data.index = [x[1:-1] for x in extracted_data[1]]
    df_data.columns = extracted_data[2]

    df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]
    df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]

    df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
    df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)


    t1 = time.time()
    print("全部计算时间：", t1-t0)


    # # 步骤1：从知识图谱中删除5个实体，并保存其三元组
    # deleted_entities = df_train[items_name].sample(1).values
    # deleted_triples = {}
    # for entity in deleted_entities:
    #     triples = connector.get_triples(entity)  # 确保 entity 格式正确
    #     deleted_triples[entity] = triples
    #     connector.delete_entity(entity)

    # 步骤1：从知识图谱中删除5个实体，并保存其三元组
    deleted_entities = df_train[items_name].sample(5).values
    deleted_triples = {}
    new_pos_file = pos_file.copy()  # 新建一个 pos_file 的副本

    for entity in deleted_entities:
        triples = connector.get_triples(entity)  # 确保 entity 格式正确
        deleted_triples[entity] = triples
        triples_deleted = connector.delete_entity(entity)
        # 从 new_pos_file 中删除实体
        entity_str = f'<{entity}>'
        if entity_str in new_pos_file:
            new_pos_file.remove(entity_str)



    # 步骤2：获得新知识图谱的嵌入
    X_train_new, _ = extractor.create_dataset(depth, new_pos_file, set(), excludes, jobs=4)
    extracted_data_new = extractor.fit_transform(X_train_new, counts=False, levels=False, float_rpr=False)

    df_data_new = pd.DataFrame.sparse.from_spmatrix(extracted_data_new[0])
    df_data_new.index = [x[1:-1] for x in extracted_data_new[1]]
    df_data_new.columns = extracted_data_new[2]

    df_train_extr_new = df_data_new[df_data_new.index.isin(df_train[items_name].values)]
    df_test_extr_new = df_data_new[df_data_new.index.isin(df_test[items_name].values)]

    df_train_extr_new = df_train_extr_new.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
    df_test_extr_new = df_test_extr_new.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

    # 步骤3：一个个添加删除的实体，获得增加后的实体嵌入

    X_train_new_2 = X_train_new.copy()
    # 将 df_data_new 转换为非稀疏格式
    df_data_new = df_data_new.sparse.to_dense()


    totaltime=0
    totaltime1=0
    for entity, triples in deleted_triples.items():
        connector.add_entity(entity, triples)
        t2 = time.time()
        affected_nodes = connector.get_affected_nodes(entity, depth)

        affected_nodes = convert_to_pos_format(affected_nodes)

        pos_affected_nodes = set(affected_nodes).intersection(pos_file)
        # print(f"pos nodes for {entity}: {pos_file}")
        # print(f"Affected nodes for {entity}: {affected_nodes}")

        # 为每个受影响的pos节点重新计算特征
        print("pos_affected_nodes len:",len(pos_affected_nodes))
        updated_features, _ = extractor.create_dataset(depth, pos_affected_nodes, set(), excludes, jobs=4)
        # updated_features = extractor.kg.extract_neighborhoods(pos_file, depth, excludes, verbose=True)

        t3 = time.time()
        totaltime = totaltime + t3 - t2
        # for node, features in updated_features:
        #     # 假设 features 是一个字典 {feature_name: feature_value}
        #     for feature_name, feature_value in features.items():
        #         if feature_name in df_data_new.columns:
        #             df_data_new.at[node, feature_name] = feature_value
        #         else:
        #             df_data_new[feature_name] = 0
        #             df_data_new.at[node, feature_name] = feature_value

        for node, features in updated_features:
            # 查找 node 在 X_train_new_2 中对应的 dict
            node_str = f'{node}'
            node_index = None
            for i, (n, f) in enumerate(X_train_new_2):
                if n == node_str:
                    node_index = i
                    break

            if node_index is not None:
                X_train_new_2[node_index] = (node_str, features)
            else:
                # 添加新的节点
                X_train_new_2.append((node_str, features))
            if node_str in pos_affected_nodes:
                new_pos_file.add(node_str)  # 重新添加回 new_pos_file
        t4 = time.time()
        totaltime1 = totaltime1 + t4 - t2



    # t3 = time.time()
    print("动态更新时间:",totaltime)
    print("totaltime1:",totaltime1)
    # 将 df_data_new 转换回稀疏格式
    # df_data_new = df_data_new.sparse.from_spmatrix(df_data_new)


    # 步骤4：对实体嵌入进行训练，验证动态更新有效性
    # 初始模型训练
    model_initial = RandomForestClassifier()  # 使用适当的模型
    X_origin_train = df_train_extr.drop(['label', items_name], axis=1).values
    y_origin_train = df_train_extr['label'].values
    X_origin_test = df_test_extr.drop(['label', items_name], axis=1).values
    y_origin_test = df_test_extr['label'].values
    model_initial.fit(X_origin_train, y_origin_train)
    initial_score = model_initial.score(X_origin_test, y_origin_test)

    # 更新后的模型训练
    extracted_data_new = extractor.fit_transform(X_train_new_2, counts=False, levels=False, float_rpr=False)

    df_data_new = pd.DataFrame.sparse.from_spmatrix(extracted_data_new[0])
    df_data_new.index = [x[1:-1] for x in extracted_data_new[1]]
    df_data_new.columns = extracted_data_new[2]

    df_train_extr_new_2 = df_data_new[df_data_new.index.isin(df_train[items_name].values)]
    df_train_extr_new_2 = df_train_extr_new_2.merge(df_train[[items_name, 'label']], left_index=True,
                                                    right_on=items_name)

    df_test_extr_new_2 = df_data_new[df_data_new.index.isin(df_test[items_name].values)]
    df_test_extr_new_2 = df_test_extr_new_2.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

    X_new_train = df_train_extr_new_2.drop(['label', items_name], axis=1).values
    y_new_train = df_train_extr_new_2['label'].values
    X_new_test = df_test_extr_new_2.drop(['label', items_name], axis=1).values
    y_new_test = df_test_extr_new_2['label'].values
    model_updated = RandomForestClassifier()
    model_updated.fit(X_new_train, y_new_train)
    updated_score = model_updated.score(X_new_test, y_new_test)

    print(f'Updated model score after adding nodes: {updated_score}')

    print(f'Initial model score: {initial_score}')
    print(f'Updated model score: {updated_score}')

    # Example usage:
    compare_dataframes_columns(df_train_extr, df_train_extr_new_2)


    # 检查缺失值
    check_missing_values(df_train_extr_new_2, items_name)

    # 比较行特征
    # compare_row_features(df_train_extr, df_train_extr_new_2, items_name)

    print("df_train_extr sample数量：", len(df_train_extr))
    print("df_train_extr_new_2 sample数量：", len(df_train_extr_new_2))
    compare_dataframes(df_train_extr, df_train_extr_new_2, items_name)
if __name__ == '__main__':
    main()
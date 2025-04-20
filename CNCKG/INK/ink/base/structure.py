"""
structure.py file.
Defines the functions and classes to construct the binary INK representation.
"""

import numpy as np
from ink.base.graph import KnowledgeGraph
from ink.base.transform.counts import create_counts
from ink.base.transform.levels import create_levels
from ink.base.transform.binarize import create_representation

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class InkExtractor:
    """
    The INK extractor.
    Constructs the binary representation from a given knowledge graph.

    :param connector: Connector instance.
    :type connector: :py:class:`ink.base.connectors.AbstractConnector`
    :param prefixes: Optional dictionary of prefixes which should be mapped.
    :type prefixes: list
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    """
    def __init__(self, connector, prefixes=None, extract_inverse=False, verbose=False):
        if prefixes is None:
            prefixes = []
        self.connector = connector
        self.kg = KnowledgeGraph(connector, prefixes, extract_inverse)
        self.levels = {}
        self.verbose = verbose
        self.train_data = None

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
        noi_neighbours = self.kg.extract_neighborhoods(all_noi, depth, skip_list, verbose=self.verbose, jobs=jobs)
        # update order
        all_noi = [n[0] for n in noi_neighbours]

        a = []
        pos_related_relations = []  # 新增列表用于存储与pos相关的关系
        if len(pos_set) > 0 or len(neg_set) > 0:
            for index, v in enumerate(all_noi):
                if v in pos_set:
                    a.append(1)
                    # 获取与pos相关的节点的邻居关系
                    _, node_neighbors = noi_neighbours[index]
                    for relation, values in node_neighbors.items():
                        pos_related_relations.extend([relation] * len(values))
                else:
                    a.append(0)

        # 统计关系的种类数
        relation_types = set(pos_related_relations)
        num_relation_types = len(relation_types)

        print(f"与pos有关的关系一共有 {num_relation_types} 类")

        # 保存与pos有关的关系（这里简单打印，你可以修改为保存到文件等操作）
        print("与pos有关的关系如下：")
        for relation in relation_types:
            print(relation)

        return noi_neighbours, np.array(a)

    def fit_transform(self, dct, counts=False, levels=False, float_rpr=False):

        if self.verbose:
            print('# Transform')
        if counts:
            dct = create_counts(dct, verbose=self.verbose)

        self.train_data = dct

        if levels:
            dct = create_levels(dct, dct, verbose=self.verbose)

        cat_df = create_representation(dct, float_rpr=float_rpr, verbose=self.verbose)

        return cat_df

    def transform(self, dct, counts=False, levels=False):

        if self.verbose:
            print('# Transform')
        if counts:
            dct = create_counts(dct, verbose=self.verbose)

        if levels:
            dct = create_levels(self.train_data, dct, verbose=self.verbose)

        cat_df = create_representation(dct, verbose=self.verbose)

        return cat_df

"""
graph.py file.
Defines all required functions to extract the neighborhoods within a knowledge graph.
"""

from tqdm import tqdm
import multiprocessing as mp
from functools import lru_cache
from multiprocessing import Pool
#import gc

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class KnowledgeGraph:
    """
    Knowledge graph class representation

    This graph builds and stores the internal knowledge graph representations.
    It stores and builds the neighborhoods of the nodes of interest through the initialized connector.

    :param connector: Connector instance.
    :type connector: :py:class:`ink.base.connectors.AbstractConnector`
    :param prefixes: Optional dictionary of prefixes which should be mapped.
    :type prefixes: list
    """
    def __init__(self, connector, prefixes=None, extract_inverse=False):
        if prefixes is None:
            prefixes = []
        self.connector = connector
        self.ind_instances = {}
        self.predicates = set()
        self.total_parts = {}
        self.neighbour_parts = {}
        self.prefixes = prefixes
        self.extract_inverse = extract_inverse

    def neighborhood_request(self, noi):

        try:
            if noi[0] == '<':
                noi = noi[1:]
            if noi[-1] == '>':
                noi = noi[:-1]

            q = 'SELECT ?p ?o ?dt WHERE { BIND( IRI("' + noi + '") AS ?s ) ?s ?p ?o. BIND (datatype(?o) AS ?dt) }'
            res = self.connector.query(q)
            if True:
                # 这里是认为指向noi的也是其邻居？
                q = 'SELECT ?p ?o ?dt ?dt WHERE { BIND( IRI("' + noi + '") AS ?s ) ?o ?p ?s. BIND (datatype(?o) AS ?dt) }'
                res += self.connector.inv_query(q)
            #q = 'SELECT ?p ?o WHERE { <'+noi+'> ?p ?o. }'
            return res
        except Exception as e:
            #print(e)
            return []

    def extract_neighborhoods(self, data, depth, skip_list=None, verbose=False, jobs=1):


        if skip_list is None:
            skip_list = []

        seq =[(r, depth, skip_list) for r in data]
        if jobs > 1:
            with Pool(jobs) as pool:
                res = list(tqdm(pool.imap_unordered(self._create_neighbour_paths, seq, chunksize=100),
                                total=len(data), disable=not verbose))
                pool.close()
                pool.join()
        else:
            res = []
            for s in tqdm(seq, disable=not verbose, total=len(data)):
                res.append(self._create_neighbour_paths(s))
        return res

    def _create_neighbour_paths(self, t):

        noi, depth, avoid_lst = t
        value = noi, ""
        total_parts = {}
        all_done = []
        res = self._define_neighborhood(value, depth, avoid_lst, total_parts, all_done)
        #gc.collect()
        return noi, res

    def _replace_pref(self, r):

        for x in self.prefixes:
            r = r.replace(x, self.prefixes[x])
            #return r
        return r

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
            if depth-1 > 0:
                #self.connector.close()
                [total_parts.update(self._define_neighborhood(value, depth - 1, avoid_lst, total_parts, all_done))
                 for value in next_noi]
            return total_parts

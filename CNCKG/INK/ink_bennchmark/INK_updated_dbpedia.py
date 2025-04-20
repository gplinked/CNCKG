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
    prefs = {'http://www.w3.org/2005/Atom': 'a',
             'http://schemas.talis.com/2005/address/schema#': 'address',
             'http://webns.net/mvcb/': 'admin',
             'http://www.w3.org/ns/activitystreams#': 'as',
             'http://atomowl.org/ontologies/atomrdf#': 'atom',
             'http://soap.amazon.com/': 'aws',
             'http://b3s.openlinksw.com/': 'b3s',
             'http://schemas.google.com/gdata/batch': 'batch',
             'http://purl.org/ontology/bibo/': 'bibo',
             'bif:': 'bif',
             'http://www.openlinksw.com/schemas/bugzilla#': 'bugzilla',
             'http://www.w3.org/2002/12/cal/icaltzd#': 'c',
             'http://www.openlinksw.com/campsites/schema#': 'campsite',
             'http://www.crunchbase.com/': 'cb',
             'http://web.resource.org/cc/': 'cc',
             'http://purl.org/rss/1.0/modules/content/': 'content',
             'http://purl.org/captsolo/resume-rdf/0.2/cv#': 'cv',
             'http://purl.org/captsolo/resume-rdf/0.2/base#': 'cvbase',
             'http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#': 'dawgt',
             'http://dbpedia.org/resource/Category:': 'dbc',
             'http://dbpedia.org/ontology/': 'dbo',
             'http://dbpedia.org/property/': 'dbp',
             'http://af.dbpedia.org/resource/': 'dbpedia-af',
             'http://als.dbpedia.org/resource/': 'dbpedia-als',
             'http://an.dbpedia.org/resource/': 'dbpedia-an',
             'http://ar.dbpedia.org/resource/': 'dbpedia-ar',
             'http://az.dbpedia.org/resource/': 'dbpedia-az',
             'http://bar.dbpedia.org/resource/': 'dbpedia-bar',
             'http://be.dbpedia.org/resource/': 'dbpedia-be',
             'http://be-x-old.dbpedia.org/resource/': 'dbpedia-be-x-old',
             'http://bg.dbpedia.org/resource/': 'dbpedia-bg',
             'http://br.dbpedia.org/resource/': 'dbpedia-br',
             'http://ca.dbpedia.org/resource/': 'dbpedia-ca',
             'http://commons.dbpedia.org/resource/': 'dbpedia-commons',
             'http://cs.dbpedia.org/resource/': 'dbpedia-cs',
             'http://cy.dbpedia.org/resource/': 'dbpedia-cy',
             'http://da.dbpedia.org/resource/': 'dbpedia-da',
             'http://de.dbpedia.org/resource/': 'dbpedia-de',
             'http://dsb.dbpedia.org/resource/': 'dbpedia-dsb',
             'http://el.dbpedia.org/resource/': 'dbpedia-el',
             'http://eo.dbpedia.org/resource/': 'dbpedia-eo',
             'http://es.dbpedia.org/resource/': 'dbpedia-es',
             'http://et.dbpedia.org/resource/': 'dbpedia-et',
             'http://eu.dbpedia.org/resource/': 'dbpedia-eu',
             'http://fa.dbpedia.org/resource/': 'dbpedia-fa',
             'http://fi.dbpedia.org/resource/': 'dbpedia-fi',
             'http://fr.dbpedia.org/resource/': 'dbpedia-fr',
             'http://frr.dbpedia.org/resource/': 'dbpedia-frr',
             'http://fy.dbpedia.org/resource/': 'dbpedia-fy',
             'http://ga.dbpedia.org/resource/': 'dbpedia-ga',
             'http://gd.dbpedia.org/resource/': 'dbpedia-gd',
             'http://gl.dbpedia.org/resource/': 'dbpedia-gl',
             'http://he.dbpedia.org/resource/': 'dbpedia-he',
             'http://hr.dbpedia.org/resource/': 'dbpedia-hr',
             'http://hsb.dbpedia.org/resource/': 'dbpedia-hsb',
             'http://hu.dbpedia.org/resource/': 'dbpedia-hu',
             'http://id.dbpedia.org/resource/': 'dbpedia-id',
             'http://ie.dbpedia.org/resource/': 'dbpedia-ie',
             'http://io.dbpedia.org/resource/': 'dbpedia-io',
             'http://is.dbpedia.org/resource/': 'dbpedia-is',
             'http://it.dbpedia.org/resource/': 'dbpedia-it',
             'http://ja.dbpedia.org/resource/': 'dbpedia-ja',
             'http://ka.dbpedia.org/resource/': 'dbpedia-ka',
             'http://kk.dbpedia.org/resource/': 'dbpedia-kk',
             'http://ko.dbpedia.org/resource/': 'dbpedia-ko',
             'http://ku.dbpedia.org/resource/': 'dbpedia-ku',
             'http://la.dbpedia.org/resource/': 'dbpedia-la',
             'http://lb.dbpedia.org/resource/': 'dbpedia-lb',
             'http://lmo.dbpedia.org/resource/': 'dbpedia-lmo',
             'http://lt.dbpedia.org/resource/as': 'dbpedia-lt',
             'http://lv.dbpedia.org/resource/a': 'dbpedia-lv',
             'http://mk.dbpedia.org/resource/': 'dbpedia-mk',
             'http://mr.dbpedia.org/resource/': 'dbpedia-mr',
             'http://ms.dbpedia.org/resource/': 'dbpedia-ms',
             'http://nah.dbpedia.org/resource/': 'dbpedia-nah',
             'http://nds.dbpedia.org/resource/': 'dbpedia-nds',
             'http://nl.dbpedia.org/resource/': 'dbpedia-nl',
             'http://nn.dbpedia.org/resource/': 'dbpedia-nn',
             'http://no.dbpedia.org/resource/': 'dbpedia-no',
             'http://nov.dbpedia.org/resource/': 'dbpedia-nov',
             'http://oc.dbpedia.org/resource/': 'dbpedia-oc',
             'http://os.dbpedia.org/resource/': 'dbpedia-os',
             'http://pam.dbpedia.org/resource/': 'dbpedia-pam',
             'http://pl.dbpedia.org/resource/': 'dbpedia-pl',
             'http://pms.dbpedia.org/resource/': 'dbpedia-pms',
             'http://pnb.dbpedia.org/resource/': 'dbpedia-pnb',
             'http://pt.dbpedia.org/resource/': 'dbpedia-pt',
             'http://ro.dbpedia.org/resource/': 'dbpedia-ro',
             'http://ru.dbpedia.org/resource/': 'dbpedia-ru',
             'http://sh.dbpedia.org/resource/': 'dbpedia-sh',
             'http://simple.dbpedia.org/resource/': 'dbpedia-simple',
             'http://sk.dbpedia.org/resource/': 'dbpedia-sk',
             'http://sl.dbpedia.org/resource/': 'dbpedia-sl',
             'http://sq.dbpedia.org/resource/': 'dbpedia-sq',
             'http://sr.dbpedia.org/resource/': 'dbpedia-sr',
             'http://sv.dbpedia.org/resource/': 'dbpedia-sv',
             'http://sw.dbpedia.org/resource/': 'dbpedia-sw',
             'http://th.dbpedia.org/resource/': 'dbpedia-th',
             'http://tr.dbpedia.org/resource/': 'dbpedia-tr',
             'http://ug.dbpedia.org/resource/': 'dbpedia-ug',
             'http://uk.dbpedia.org/resource/': 'dbpedia-uk',
             'http://vi.dbpedia.org/resource/': 'dbpedia-vi',
             'http://vo.dbpedia.org/resource/': 'dbpedia-vo',
             'http://war.dbpedia.org/resource/': 'dbpedia-war',
             'http://dbpedia.openlinksw.com/wikicompany/': 'dbpedia-wikicompany',
             'http://wikidata.dbpedia.org/resource/': 'dbpedia-wikidata',
             'http://yo.dbpedia.org/resource/': 'dbpedia-yo',
             'http://zh.dbpedia.org/resource/': 'dbpedia-zh',
             'http://zh-min-nan.dbpedia.org/resource/': 'dbpedia-zh-min-nan',
             'http://dbpedia.org/resource/': 'dbr',
             'http://dbpedia.org/resource/Template:': 'dbt',
             'http://purl.org/dc/elements/1.1/': 'dc',
             'http://purl.org/dc/terms/': 'dct',
             'http://digg.com/docs/diggrss/': 'digg',
             'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl': 'dul',
             'urn:ebay:apis:eBLBaseComponents': 'ebay',
             'http://purl.oclc.org/net/rss_2.0/enc#': 'enc',
             'http://www.w3.org/2003/12/exif/ns/': 'exif',
             'http://api.facebook.com/1.0/': 'fb',
             'http://api.friendfeed.com/2008/03': 'ff',
             'http://www.w3.org/2005/xpath-functions/#': 'fn',
             'http://xmlns.com/foaf/0.1/': 'foaf',
             'http://rdf.freebase.com/ns/': 'freebase',
             'http://base.google.com/ns/1.0': 'g',
             'http://www.openlinksw.com/schemas/google-base#': 'gb',
             'http://schemas.google.com/g/2005': 'gd',
             'http://www.w3.org/2003/01/geo/wgs84_pos#': 'geo',
             'http://sws.geonames.org/': 'geodata',
             'http://www.geonames.org/ontology#': 'geonames',
             'http://www.georss.org/georss/': 'georss',
             'http://www.opengis.net/gml': 'gml',
             'http://purl.org/obo/owl/GO#': 'go',
             'http://www.openlinksw.com/schemas/hlisting/': 'hlisting',
             'http://wwww.hoovers.com/': 'hoovers',
             'http://purl.org/stuff/hrev#': 'hrev',
             'http://www.w3.org/2002/12/cal/ical#': 'ical',
             'http://web-semantics.org/ns/image-regions': 'ir',
             'http://www.itunes.com/DTDs/Podcast-1.0.dtd': 'itunes',
             'http://www.w3.org/ns/ldp#': 'ldp',
             'http://linkedgeodata.org/triplify/': 'lgdt',
             'http://linkedgeodata.org/vocabulary#': 'lgv',
             'http://www.xbrl.org/2003/linkbase': 'link',
             'http://lod.openlinksw.com/': 'lod',
             'http://www.w3.org/2000/10/swap/math#': 'math',
             'http://search.yahoo.com/mrss/': 'media',
             'http://purl.org/commons/record/mesh/': 'mesh',
             'urn:oasis:names:tc:opendocument:xmlns:meta:1.0': 'meta',
             'http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#': 'mf',
             'http://musicbrainz.org/ns/mmd-1.0#': 'mmd',
             'http://purl.org/ontology/mo/': 'mo',
             'http://www.freebase.com/': 'mql',
             'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#': 'nci',
             'http://www.semanticdesktop.org/ontologies/nfo/#': 'nfo',
             'http://www.openlinksw.com/schemas/ning#': 'ng',
             'http://data.nytimes.com/': 'nyt',
             'http://www.openarchives.org/OAI/2.0/': 'oai',
             'http://www.openarchives.org/OAI/2.0/oai_dc/': 'oai_dc',
             'http://www.geneontology.org/formats/oboInOwl#': 'obo',
             'urn:oasis:names:tc:opendocument:xmlns:office:1.0': 'office',
             'http://www.opengis.net/': 'ogc',
             'http://www.opengis.net/ont/gml#': 'ogcgml',
             'http://www.opengis.net/ont/geosparql#': 'ogcgs',
             'http://www.opengis.net/def/function/geosparql/': 'ogcgsf',
             'http://www.opengis.net/def/rule/geosparql/': 'ogcgsr',
             'http://www.opengis.net/ont/sf#': 'ogcsf',
             'urn:oasis:names:tc:opendocument:xmlns:meta:1.0:': 'oo',
             'http://a9.com/-/spec/opensearchrss/1.0/': 'openSearch',
             'http://sw.opencyc.org/concept/': 'opencyc',
             'http://www.openlinksw.com/schema/attribution#': 'opl',
             'http://www.openlinksw.com/schemas/getsatisfaction/': 'opl-gs',
             'http://www.openlinksw.com/schemas/meetup/': 'opl-meetup',
             'http://www.openlinksw.com/schemas/xbrl/': 'opl-xbrl',
             'http://www.openlinksw.com/schemas/oplweb#': 'oplweb',
             'http://www.openarchives.org/ore/terms/': 'ore',
             'http://www.w3.org/2002/07/owl#': 'owl',
             'http://www.buy.com/rss/module/productV2/': 'product',
             'http://purl.org/science/protein/bysequence/': 'protseq',
             'http://www.w3.org/ns/prov#': 'prov',
             'http://backend.userland.com/rss2': 'r',
             'http://www.radiopop.co.uk/': 'radio',
             'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
             'http://www.w3.org/ns/rdfa#': 'rdfa',
             'http://www.openlinksw.com/virtrdf-data-formats#': 'rdfdf',
             'http://www.w3.org/2000/01/rdf-schema#': 'rdfs',
             'http://purl.org/stuff/rev#': 'rev',
             'http://purl.org/rss/1.0/': 'rss',
             'http://purl.org/science/owl/sciencecommons/': 'sc',
             'http://schema.org/': 'schema',
             'http://purl.org/NET/scovo#': 'scovo',
             'http://www.w3.org/ns/sparql-service-description#': 'sd',
             'urn:sobject.enterprise.soap.sforce.com': 'sf',
             'http://www.w3.org/ns/shacl#': 'sh',
             'http://www.w3.org/ns/shacl-shacl#': 'shsh',
             'http://rdfs.org/sioc/ns#': 'sioc',
             'http://rdfs.org/sioc/types#': 'sioct',
             'http://www.openlinksw.com/ski_resorts/schema#': 'skiresort',
             'http://www.w3.org/2004/02/skos/core#': 'skos',
             'http://purl.org/rss/1.0/modules/slash/': 'slash',
             'http://spinrdf.org/sp#': 'sp',
             'http://spinrdf.org/spin#': 'spin',
             'http://spinrdf.org/spl#': 'spl',
             'sql:': 'sql',
             'http://xbrlontology.com/ontology/finance/stock_market#': 'stock',
             'http://www.openlinksw.com/schemas/twfy#': 'twfy',
             'http://umbel.org/umbel#': 'umbel',
             'http://umbel.org/umbel/ac/': 'umbel-ac',
             'http://umbel.org/umbel/rc/': 'umbel-rc',
             'http://umbel.org/umbel/sc/': 'umbel-sc',
             'http://purl.uniprot.org/': 'uniprot',
             'http://dbpedia.org/units/': 'units',
             'http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/': 'usc',
             'http://www.openlinksw.com/xsltext/': 'v',
             'http://www.w3.org/2001/vcard-rdf/3.0#': 'vcard',
             'http://www.w3.org/2006/vcard/ns#': 'vcard2006',
             'http://www.openlinksw.com/virtuoso/xslt/': 'vi',
             'http://www.openlinksw.com/virtuoso/xslt': 'virt',
             'http://www.openlinksw.com/schemas/virtcxml#': 'virtcxml',
             'http://www.openlinksw.com/schemas/virtpivot#': 'virtpivot',
             'http://www.openlinksw.com/schemas/virtrdf#': 'virtrdf',
             'http://rdfs.org/ns/void#': 'void',
             'http://www.worldbank.org/': 'wb',
             'http://www.w3.org/2007/05/powder-s#': 'wdrs',
             'http://www.w3.org/2005/01/wf/flow#': 'wf',
             'http://wellformedweb.org/CommentAPI/': 'wfw',
             'http://commons.wikimedia.org/wiki/': 'wiki-commons',
             'http://www.wikidata.org/entity/': 'wikidata',
             'http://en.wikipedia.org/wiki/': 'wikipedia-en',
             'http://www.w3.org/2004/07/xpath-functions': 'xf',
             'http://gmpg.org/xfn/11#': 'xfn',
             'http://www.w3.org/1999/xhtml': 'xhtml',
             'http://www.w3.org/1999/xhtml/vocab#': 'xhv',
             'http://www.xbrl.org/2003/instance': 'xi',
             'http://www.w3.org/XML/1998/namespace': 'xml',
             'http://www.ning.com/atom/1.0': 'xn',
             'http://www.w3.org/2001/XMLSchema#': 'xsd',
             'http://www.w3.org/XSL/Transform/1.0': 'xsl10',
             'http://www.w3.org/1999/XSL/Transform': 'xsl1999',
             'http://www.w3.org/TR/WD-xsl': 'xslwd',
             'urn:yahoo:maps': 'y',
             'http://dbpedia.org/class/yago/': 'yago',
             'http://yago-knowledge.org/resource/': 'yago-res',
             'http://gdata.youtube.com/schemas/2007': 'yt',
             'http://s.zemanta.com/ns#': 'zem'}

    rel = False

    dataset = sys.argv[1]  # 'BGS'#'BGS'
    depth = int(sys.argv[2])
    method = sys.argv[3]
    dir_kb = './' + dataset
    train = './' + dataset + '/TrainingSet-2.tsv'
    test = './' + dataset + '/TestSet-2.tsv'

    excludes = []  # excludes_dict[dataset]#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']

    labels_dict = {'AAUP': 'label_salary', 'albums': 'label', 'cities': 'label', 'forbes': 'label', 'movies': 'Label'}
    add_cols = {"AAUP": ['DBpedia_URL', 'Number_of_full_professors', 'Number_of_associate_professors',
                         'Number_of_assistant_professors', 'Number_of_instructors', 'Number_of_faculty_all_ranks'],
                "forbes": ["DBpedia_URL", "Market_Value", "Sales", "Profits", "Assets"],
                "movies": ["DBpedia_URL"],
                "albums": ["DBpedia_URL"],
                "cities": ["DBpedia_URL"]}
    label_name = labels_dict[dataset]
    items_name = 'DBpedia_URL'


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

    connector = RDFLibConnector(dir_kb+'/'+dataset+".nt", 'n3')
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
    for i in range(5):
        model_initial = RandomForestClassifier()  # 使用适当的模型
        model_initial.fit(X_origin_train, y_origin_train)
        initial_score = model_initial.score(X_origin_test, y_origin_test)

        model_updated = RandomForestClassifier()
        model_updated.fit(X_new_train, y_new_train)
        updated_score = model_updated.score(X_new_test, y_new_test)


        print(f'Initial model score:'+str(i)+ f'{initial_score}')
        print(f'Updated model score:'+str(i)+ f' {updated_score}')

    # Example usage:
    compare_dataframes_columns(df_train_extr, df_train_extr_new_2)


    # 检查缺失值
    check_missing_values(df_train_extr_new_2, items_name)

    # 比较行特征
    # compare_row_features(df_train_extr, df_train_extr_new_2, items_name)

    print("df_train_extr sample数量：", len(df_train_extr))
    print("df_train_extr_new_2 sample数量：", len(df_train_extr_new_2))
    compare_dataframes(df_train_extr, df_train_extr_new_2, items_name)

    # 保存 OKG 特征
    np.savez(f"OKG_{dataset}_depth{depth}_train.npz", X=X_origin_train, y=y_origin_train)
    np.savez(f"OKG_{dataset}_depth{depth}_test.npz", X=X_origin_test, y=y_origin_test)

    # 保存 UKG 特征
    np.savez(f"UKG_{dataset}_depth{depth}_train.npz", X=X_new_train, y=y_new_train)
    np.savez(f"UKG_{dataset}_depth{depth}_test.npz", X=X_new_test, y=y_new_test)


if __name__ == '__main__':
    main()
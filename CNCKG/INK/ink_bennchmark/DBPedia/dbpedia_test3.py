import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer

import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
from pyrdf2vec.graphs.vertex import Vertex
from pyrdf2vec.graphs import KG
import pandas as pd
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from multiprocessing import Pool
from hashlib import md5
from typing import List,Set, Tuple, Any
from tqdm import tqdm
import rdflib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold

from pympler import asizeof
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from ink.base.connectors import RDFLibConnector
def compute_Cr(df):
    # 计算错误率分布e(diff)
    error_rate = 1 - df.groupby('difficult')['detail'].mean()

    # 计算实例分布f(diff)
    instance_distribution = df['difficult'].value_counts(normalize=True)

    # 合并错误率和实例分布到一个DataFrame
    error_instance_df = pd.concat([error_rate, instance_distribution], axis=1)
    error_instance_df.columns = ['error_rate', 'instance_distribution']
    error_instance_df.reset_index(inplace=True)

    # 计算每个 diff 的结果 (1 - diff) * e(diff) * f(diff)
    result = (1 - error_instance_df['difficult']) * error_instance_df['error_rate'] * error_instance_df[
        'instance_distribution']

    # 对不同的 diff 值进行求和
    sum_by_diff = result.sum()

    # 输出结果
    print(sum_by_diff)
    print("置信度", 1 - sum_by_diff)


import os.path
import csv


""" parameters """
if __name__ == "__main__":

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

    dataset = sys.argv[1]#'BGS'#'BGS'
    depth = int(sys.argv[2])
    method = sys.argv[3]
    count = sys.argv[4]
    level = sys.argv[5]
    float_rpr = sys.argv[6]

    base_dir = os.path.dirname(__file__)  # DBPedia目录
    dir_kb = os.path.join(base_dir, dataset)
    train = os.path.join(base_dir, dataset, 'TrainingSet.tsv')
    test = os.path.join(base_dir, dataset, 'TestSet.tsv')

    excludes = []

    labels_dict = { 'movies': 'Label'}
    add_cols = {"movies":["DBpedia_URL"]}
    label_name = labels_dict[dataset]
    items_name = 'DBpedia_URL'


    df_train = pd.read_csv(train, delimiter='\t')
    df_train = df_train.dropna(subset=[label_name])
    df_test = pd.read_csv(test, delimiter='\t')
    df_test = df_test.dropna(subset=[label_name])

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])


    pos_file = set(['<' + x + '>' for x in data[items_name].values])


    connector = RDFLibConnector(dir_kb+'/'+dataset+".nt", 'n3')

    #for _ in tqdm(range(5)):

    skip = ["http://dbpedia.org/ontology/abstract", "http://dbpedia.org/ontology/wikiPageExternalLink",
           "http://www.w3.org/2002/07/owl/sameAs", "http://purl.org/dc/terms/subject",
           "http://www.w3.org/2000/01/rdf-schema#comment","http://www.w3.org/2000/01/rdf-schema#label",
           "http://dbpedia.org/ontology/wikiPageWikiLinkText"]

    ## INK exrtact


    t0 = time.time()
    extractor = InkExtractor(connector, prefixes=prefs, verbose=True)
    X_train, _ = extractor.create_dataset(depth, pos_file, set(),skip_list=skip, jobs=4)
    extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=True)

    print("INK create time:"+str(time.time()-t0))

    df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
    df_data.index = [x[1:-1] for x in extracted_data[1]]
    df_data.columns = extracted_data[2]

    df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]  # df_data.loc[[df_train['proxy']],:]
    df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]  # df_data.loc[[df_test['proxy']],:]

    df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
    df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)


    df_Data = pd.concat([df_train_extr, df_test_extr])

    #
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




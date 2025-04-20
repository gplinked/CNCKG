"""
connector.py file.
Defines all required functions to make a connection to a knowledge graph.
"""

import json
# import stardog
from urllib import parse
from rdflib import Graph
from abc import ABC, abstractmethod
# import xmltodict
import time
from rdflib import Graph, URIRef, Literal

try:
    import faster_than_requests as ftr
    ftr.set_headers(headers=[("Accept", "application/sparql-results+json")])
    fast = True
except ImportError as e:
    import requests
    from requests.adapters import HTTPAdapter
    from requests import Session
    fast = False

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class AbstractConnector(ABC):
    """
    Abstract Connector class
    Can be used to implement new types of connectors.

    The only function which have to be implemented is the query(self, str) function.
    This function is used to query the neighborhood.
    Makes sure this function return a dictionary in the correct format.
    """
    @abstractmethod
    def query(self, q_str):
        """
        Abstract query function.
        :param q_str: Query string.
        :type q_str: str
        :rtype: dict
        """
        pass

    @abstractmethod
    def inv_query(self, q_str):
        """
        Abstract inverse query function.
        :param q_str: Query string.
        :type q_str: str
        :rtype: dict
        """
        pass




class StardogConnector(AbstractConnector):
    """
    A Stardog connector

    This Stardog connector class sets up the connection to a Stardog database.

    :param conn_details: a dictionary containing all the connection details.
    :type conn_details: dict
    :param database: database of interest.
    :type database: str

    Example::
        details = {'endpoint': 'http://localhost:5820'}
        connector = StardogConnector(details, "example_database")
    """
    def __init__(self, conn_details, database, reason=False):
        self.details = conn_details
        self.host = conn_details['endpoint']
        self.db = database
        self.reason = reason
        #self.connection = stardog.Connection(self.db, **conn_details)

        #if not fast:
        #    self.session = Session()
        #    adapter = HTTPAdapter(pool_connections=10000, pool_maxsize=10000)
        #    self.session.mount('http://', adapter)
        #else:
        #    ftr.set_headers(headers=[("Accept", "application/sparql-results+json")])

    def delete_db(self):
        """
        Function to delete the delete the database when it is available
        :return: None
        """
        try:
            with stardog.Admin(**self.details) as admin:
                admin.database(self.db).drop()
        except Exception as ex:
            print(ex)
            print("no database to drop")

    def upload_kg(self, filename):
        """
        Uploads the knowledge graph to the previously initialized database.
        :param filename: The filename of the knowledge graph.
        :type filename: str
        :return: None
        """
        with stardog.Admin(**self.details) as admin:
            try:
                admin.new_database(self.db)
                with stardog.Connection(self.db, **self.details) as conn:
                    conn.begin()
                    conn.add(stardog.content.File(filename))
                    conn.commit()
            except Exception as ex:
                print(ex)

    #def close(self):
    #    self.connection.close()

    def query(self, q_str):
        """
        Execute a query on the initialized Stardog database
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        with stardog.Connection(self.db, **self.details) as conn:
            r = conn.select(q_str)
            time.sleep(0.1)
        return r['results']['bindings']

    def inv_query(self, q_str):
        """
        Execute a query on the initialized Stardog database
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        with stardog.Connection(self.db, **self.details) as conn:
            r = conn.select(q_str)
            time.sleep(0.1)
        return r['results']['bindings']

    def old_query(self, q_str):
        """
        Execute a query on the initialized Stardog database
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """

        query = parse.quote(q_str)
        if fast:
            #a = time.time()
            r = ftr.get2str(self.host + '/' + self.db + '/query?query=' + query+'&reasoning='+str(self.reason))
            ftr.close_client()

            #print(time.time()-a)
        else:
            r = self.session.get(self.host + '/' + self.db + '/query?query=' + query+'&reasoning='+str(self.reason),
                                 headers={"Accept": "application/sparql-results+json"}).text
        #conn = stardog.Connection(self.db, **self.details)
        #r = conn.select(q_str)
        return json.loads(r)['results']['bindings']


class RDFLibConnector(AbstractConnector):
    """
    A RDFLib connector.

    This RDFLib connector class stores the knowledge graph directly within memory.

    :param filename: The filename of the knowledge graph.
    :type filename: str
    :param dataformat: Dataformat of the knowledge graph. Use XML for OWL files.
    :type dataformat: str

    Example::
        connector = RDFLibConnector('example.owl', 'xml')
    """
    def __init__(self, filename, dataformat):
        self.db_type = 'rdflib'
        self.g = Graph()
        self.g.parse(filename, format=dataformat)

    def query(self, q_str):
        """
        Execute a query through RDFLib
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        res = self.g.query(q_str)
        return json.loads(res.serialize(format="json"))['results']['bindings']

    def inv_query(self, q_str):
        """
        Execute a query through RDFLib
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        res = self.g.query(q_str)
        return json.loads(res.serialize(format="json"))['results']['bindings']

    def get_triples(self, entity):
        #获得图上entities所有triples，保存删除的triples
        """
        Get all triples for a given entity.
        :param entity: The entity to get triples for.
        :type entity: str
        :return: List of triples (subject, predicate, object) for the entity.
        :rtype: list of tuples
        """
        entity_ref = URIRef(entity)
        triples = []
        for p, o in self.g.predicate_objects(subject=entity_ref):
            object_type = 'URI' if isinstance(o, URIRef) else 'Literal'
            triples.append((str(entity_ref), str(p), str(o), object_type))  # 修改：包含主体实体和对象类型
        for s, p in self.g.subject_predicates(object=entity_ref):
            object_type = 'URI' if isinstance(s, URIRef) else 'Literal'
            triples.append((str(s), str(p), str(entity_ref), object_type))  # 修改：包含对象实体和对象类型
        return triples

    def delete_entity(self, entity):
        """
        Delete an entity from the knowledge graph.
        :param entity: The entity to be deleted.
        :type entity: str
        :return: List of triples that were deleted.
        :rtype: list of tuples
        """
        entity_ref = URIRef(entity)
        triples_to_delete = []

        # 查询与该实体相关的三元组
        query_subject = f'SELECT ?p ?o WHERE {{ <{entity_ref}> ?p ?o. }}'
        query_object = f'SELECT ?s ?p WHERE {{ ?s ?p <{entity_ref}>. }}'

        try:
            subject_triples = self.g.query(query_subject)
            object_triples = self.g.query(query_object)

            for row in subject_triples:
                object_type = 'URI' if isinstance(row.o, URIRef) else 'Literal'
                triples_to_delete.append((str(entity_ref), str(row.p), str(row.o), object_type))

            for row in object_triples:
                object_type = 'URI' if isinstance(row.s, URIRef) else 'Literal'
                triples_to_delete.append((str(row.s), str(row.p), str(entity_ref), object_type))

            # 删除操作
            delete_query_subject = f'DELETE WHERE {{ <{entity_ref}> ?p ?o. }}'
            delete_query_object = f'DELETE WHERE {{ ?s ?p <{entity_ref}>. }}'

            self.g.update(delete_query_subject)
            self.g.update(delete_query_object)

            # print(f"Entity {entity_ref} with triples {triples_to_delete} has been deleted.")
            return triples_to_delete
        except Exception as e:
            print(f"Error in deletion: {e}")
            return []

    def add_entity(self, entity, triples):
        try:
            for s, p, o, object_type in triples:
                subject = URIRef(s)
                predicate = URIRef(p)
                if object_type == 'URI':
                    obj = URIRef(o)
                else:
                    obj = Literal(o)
                self.g.add((subject, predicate, obj))
                # print(f"Triple ({subject}, {predicate}, {obj}) has been added.")
        except Exception as e:
             print(f"Error in BIND clause addition: {e}")
        #     # 确保增加操作的完整性，包括 BIND 子句
        #     noi = str(entity)
        #     if noi[0] == '<':
        #         noi = noi[1:]
        #     if noi[-1] == '>':
        #         noi = noi[:-1]
        #
        #     for s, p, o in triples:
        #         query_subject = f'INSERT DATA {{ BIND(IRI("{noi}") AS ?s) ?s <{p}> <{o}>. }}'
        #         query_object = f'INSERT DATA {{ BIND(IRI("{noi}") AS ?s) <{s}> <{p}> ?s. }}'
        #
        #         self.g.update(query_subject)
        #         self.g.update(query_object)
        #         print(f"Triple with BIND clauses ({s}, {p}, {o}) has been added.")

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
        new_node_ref = URIRef(new_node)
        visited = set()
        queue = [(new_node_ref, 0)]
        affected_nodes = set()

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth > depth:
                continue
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
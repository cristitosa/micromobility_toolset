import sqlite3
import numpy as np
import pandas as pd
import igraph as ig


PREPROCESSORS = []

def preprocessor():

    def decorator(func):

        global PREPROCESSORS

        PREPROCESSORS.append(func)

    return decorator


def read_nodes(file_path, table_name, node_name, attributes):
    """read links from sqlite database into network data structure, void

    file_path : name of link file
    node_name : column name of node id
    attributes : dictionary of { name in network data structure : name in database } """

    columns = list(attributes.values()) + [node_name]

    if file_path.endswith('.csv'):
        node_df = pd.read_csv(
            file_path,
            index_col=node_name,
            usecols=columns)

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        node_df = pd.read_sql(
            f'select * from {table_name}',
            db_connection,
            index_col=node_name,
            columns=columns)

        db_connection.close()

    else:
        raise TypeError(f"cannot read nodes from filetype {file_path}")

    name_map = {v: k for k, v in attributes.items()}
    node_df.rename(name_map, inplace=True)

    return node_df


def read_links(file_path,
               table_name,
               from_name,
               to_name,
               attributes_by_direction):

    """read links from sqlite database into network data structure, void

    file_path : path to csv
    from_name : column name of from node
    to_name : column name of to node
    attributes_by_direction : dictionary of
        { name in network data structure : ( column name for ab direction,
                                             column name for ba direction) }

    """

    ab_columns = []
    ba_columns = []
    for ab, ba in attributes_by_direction.values():
        ab_columns.append(ab)
        ba_columns.append(ba)

    columns = ab_columns + ba_columns + [from_name, to_name]

    if file_path.endswith('.csv'):
        link_df = pd.read_csv(file_path, usecols=columns)

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        link_df = pd.read_sql(f'select * from {table_name}',
                              db_connection,
                              columns=columns)

        db_connection.close()

    else:
        raise TypeError(f'cannot read nodes from filetype {file_path}')

    ab_df = link_df[[from_name, to_name]].copy()
    ba_df = link_df[[from_name, to_name]].copy()

    # set directional column values
    for k, v in attributes_by_direction.items():
        ab_df[k] = link_df[v[0]]
        ba_df[k] = link_df[v[1]]

    # TODO: add a two_way network property
    ba_df.rename(columns={from_name: to_name, to_name: from_name}, inplace=True)

    return pd.concat([ab_df, ba_df], sort=True).set_index([from_name, to_name]).sort_index()


class Network():

    def __init__(self, **kwargs):
        """initialize network data structure, void"""

        self.node_x_name = kwargs.get('node_x_name')
        self.node_y_name = kwargs.get('node_y_name')

        self.node_df = read_nodes(
            kwargs.get('node_file'),
            kwargs.get('node_table_name'),
            kwargs.get('node_name'),
            kwargs.get('node_attributes')
        )

        self.link_df = read_links(
            kwargs.get('link_file'),
            kwargs.get('link_table_name'),
            kwargs.get('from_name'),
            kwargs.get('to_name'),
            kwargs.get('link_attributes_by_direction')
        )

        self.check_network_completeness()

        self.graph = self.create_igraph(kwargs.get('saved_graph'))

        if PREPROCESSORS:
            for func in PREPROCESSORS:
                print(f'running {func.__name__}')
                func(self)
                print('done.')

    def check_network_completeness(self):
        """check to see that all nodes have links and nodes for all links have defined attributes

        """

        node_nodes = set(self.node_df.index.values)
        link_nodes = set(
            list(self.link_df.index.get_level_values(0)) +
            list(self.link_df.index.get_level_values(1)))

        stray_nodes = node_nodes - link_nodes
        missing_nodes = link_nodes - node_nodes

        if stray_nodes:
            self.node_df = self.node_df[~self.node_df.index.isin(list(stray_nodes))]
            print(f'removed {len(stray_nodes)} stray nodes from network')

        if missing_nodes:
            raise Exception(f'missing {len(missing_nodes)} nodes from network: {missing_nodes}')

    def create_igraph(self, graph_file=None):
        """build graph representation of network
        """

        if graph_file:

            return ig.Graph.Read(graph_file)

        # igraph expects vertex ids to be strings
        link_df = self.link_df.reset_index()
        node_df = self.node_df.reset_index()

        link_df.iloc[:, [0,1]] = link_df.iloc[:, [0,1]].astype(str)
        node_df.iloc[:, 0] = node_df.iloc[:, 0].astype(str)

        return ig.Graph.DataFrame(
            edges=link_df,
            vertices=node_df,
            directed=True)

    def get_skim_matrix(self, node_ids, weights, max_cost=None):
        """skim network net starting from node_id to node_id, using specified
        edge weights. Zero-out entries above max_cost, return matrix
        """

        # remove duplicate node_ids
        nodes_uniq = list(set(list(map(int, node_ids))))

        dists = self.graph.shortest_paths(
            source=nodes_uniq,
            target=nodes_uniq,
            weights=weights)

        # expand skim to match original node_ids
        node_map = [nodes_uniq.index(int(n)) for n in node_ids]
        skim_matrix = np.array(dists)[:, node_map][node_map, :]

        if max_cost:
            skim_matrix[skim_matrix > max_cost] = 0

        return skim_matrix

    # TODO: update this to igraph
    def get_nearby_pois(self, poi_ids, source_ids, weights, max_cost=None):
        """
        Gets list of nearby nodes for each source node.

        poi_ids: point-of-interest node ids to include in output values
        source_ids: output dictionary keys

        """

        nearby_pois = {}
        poi_set = set(poi_ids)

        for source in source_ids:
            paths = self.single_source_dijkstra(
                source,
                varcoef,
                max_cost=max_cost)[1]

            nearby_nodes = []
            for nodes in paths.values():
                nearby_nodes.extend(nodes)

            nearby_pois[source] = list(set(nearby_nodes) & poi_set)

        return nearby_pois

    def load_attribute_matrix(self, matrix, load_name, centroid_ids, varcoef, max_cost=None):
        """
        Add attribute values to a set of network links (links) given a list of node ids.

        Calculates every path between every node pair (until max cost) and adds attribute
        name/value to each intermediate link.
        """
        
        self.add_link_attribute(load_name)

        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] == len(centroid_ids)

        for i, centroid in enumerate(centroid_ids):

            paths = self.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[1]

            for j, target in enumerate(centroid_ids):

                if target in paths and matrix[i,j] > 0:
                    for k in range(len(paths[target])-1):
                        link = (paths[target][k],paths[target][k+1])
                        prev = self.get_link_attribute_value(link,load_name) or 0
                        self.set_link_attribute_value(link,load_name,prev+matrix[i,j])
        
        # save final values to link_df
        attributes = []
        for anode, bnode in list(self.link_df.index):
            attributes.append(self.get_link_attribute_value((anode, bnode), load_name))

        self.link_df[load_name] = attributes

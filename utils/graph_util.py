import itertools
import networkx as nx
import numpy as np
# import nltk


def get_node_info(vobs, tokens, POS_filter):
    nodes_dict = {}
    nodes_list = []
    index = 0
    # tokens = nltk.pos_tag(tokens)
    for word in tokens:
        tag = vobs[word]['pos'] if word in vobs else 'NN'
        if tag in POS_filter and word not in nodes_dict:
            nodes_dict[word] = index
            nodes_list.append(index)
            index += 1
    return nodes_dict, nodes_list


def get_edges(sentences, Windsize, node_dict, stride):
    edges_list = []
    for i in range(0, len(sentences)-Windsize+1, stride):
        co_list = []
        for word in sentences[i:i+Windsize]:
            if word in node_dict.keys():
                co_list.append(node_dict[word])
        for item in itertools.product(co_list, co_list):
            edges_list.append(item)
    return edges_list


def get_Amatrix(nodes_list, edges_list):
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)
    nodes = G.nodes()
    matrix = nx.to_numpy_matrix(G, nodelist=nodes)
    return matrix


def get_X(Vobs, node_dict, nF):
    if not node_dict:
        return None
    node = sorted(node_dict.items(), key=lambda x: x[1])
    words = [w[0] for w in node]
    X = []
    for word in words:
        cur_x = Vobs[word]['vec'] if word in Vobs else np.zeros(nF)
        X.append(cur_x)
    return np.array(X)

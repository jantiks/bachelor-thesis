import networkx as nx
import itertools
from copy import deepcopy
from minorminer import find_embedding
import matplotlib.pyplot as plt
import sys
import resource, sys
import random
import numpy as np
import igraph as ig
from collections import defaultdict
import logging
import math
from itertools import combinations_with_replacement
import copy
import random


def generate_graphs_base_plus_two_edges(G):
    vertices = list(G.nodes())
    
    all_possible_edges = list(itertools.combinations(vertices, 2))
    
    base_edges = set(G.edges())
    missing_edges = [e for e in all_possible_edges if e not in base_edges and (e[1], e[0]) not in base_edges]
    
    print("Total missing edges:", len(missing_edges))
    
    graphs = []
    for extra_edges in itertools.combinations(missing_edges, 2):
        G_new = G.copy()
        G_new.add_edges_from(extra_edges)
        graphs.append(G_new)
    
    return graphs

def get_all_graphs_K7_minus_one_edge():
    V = list(range(7))
    all_edges = list(itertools.combinations(V, 2))
    print("Total number of edges in K7:", len(all_edges))
    
    graphs = []
    for missing_edge in all_edges:
        G = nx.Graph()
        G.add_nodes_from(V)
        edges = [e for e in all_edges if e != missing_edge]
        G.add_edges_from(edges)
        graphs.append(G)
    
    return graphs

def get_all_graphs_K7_minus_two_edges():
    V = list(range(7))
    all_edges = list(itertools.combinations(V, 2))
    
    graphs = []

    for missing_edges in itertools.combinations(all_edges, 1):
        G = nx.Graph()
        G.add_nodes_from(V)
        edges = [e for e in all_edges if e not in missing_edges]
        G.add_edges_from(edges)
        graphs.append(G)
    
    return graphs


def zg(G):
    H = nx.Graph()
    for node in G.nodes():
        H.add_node(node)

    for u, v in G.edges():
        H.add_edge(f"{u.removesuffix('a')}b", f"{v.removesuffix('a')}b")

    for u, v in G.edges():
        H.add_edge(u, f"{v.removesuffix('a')}b")
        H.add_edge(v, f"{u.removesuffix('a')}b")

    return H

def color(vertex):
    return vertex[:-1]

def label(vertex):
    return vertex[-1]

def exists_kempe(G, start, end, original_begining, looked_at=set()):
    for key in G[start]:
        if key == end:
            return True
        if color(key) == color(start):
            continue
        if key in looked_at:
            continue
        if ((color(start) != color(end) and color(key) == color(end)) or (color(start) == color(end) and color(key) == color(original_begining))) and exists_kempe(G, key, end, original_begining, looked_at.union({key})):
            return True
    return False

def build_kempe_chain(g1, begining, end, set_of_all_filled_g, original_begining, level=0):
    if exists_kempe(g1, begining, end, original_begining):
        return
    begining_neighbours = g1[begining]

    already_built = False
    for vertex in g1.keys():
        if (label(begining) == "a" and label(vertex) == "a"):
            continue
        if (vertex in begining_neighbours):
            continue
    
        if (color(begining) == color(end) and color(vertex) == color(original_begining)) or (color(begining) == color(original_begining) and color(vertex) == color(end)):
            if already_built: ## already extended the inital graph, so let's create a new variation of it, for a different Kempe chain.
                newG = copy.deepcopy(g1)
                newG[begining].add(vertex)
                newG[vertex].add(begining)
                set_of_all_filled_g.append(newG)
                build_kempe_chain(newG, vertex, end, set_of_all_filled_g, original_begining, level=level+1)
            else:
                g1[begining].add(vertex)
                g1[vertex].add(begining)
                build_kempe_chain(g1, vertex, end, set_of_all_filled_g, original_begining, level=level+1)
                already_built = True

def get_edges_from_adjacency_list(adjacency_list):
    edges = set()
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            edges.add(tuple(sorted((node, neighbor))))
    return list(edges)  

def showGraph(G, title = "Graph"):
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=15, font_color='black')
    print(G)
    plt.suptitle(title)
    plt.show()

def getNetworkxGraph(dict):
    G = nx.Graph()
    for node, neighbors in dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    return G

if __name__ == "__main__":
    n = 15 ## build graphs on n vertices
    noMinorMinimalGraph = None
    noMinorGraph = None
    terminate = False
    # graphs = get_all_graphs_K7_minus_one_edge()
    # graphs = generate_graphs_base_plus_two_edges(nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')]))  ## counterexample from paper + 2edges
    graphs = [nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')])] ## counterexample from paper
    num_vertices_of_minor = 7
    index = 0
    foundedMinor = 0
    noMinor = 0

    for minor in graphs:
        if terminate:
            print("Terminating early found minor")
            break
        # builidng the graph H in the format suitable for the minorminer
        H = {
            f"{node}a": set([f"{neighbor}a" for neighbor in neighbors])
            for node, neighbors in minor.adj.items()
        }

        small = get_edges_from_adjacency_list(H)

        G = {}
        for i in range(2):
            suffix = chr(ord('a') + i)
            for key in H:
                new_key = key[:-1] + suffix
                G[new_key] = set()


        all_empty_g = []
        comb_size = n % (num_vertices_of_minor * 2) ## vertices with labels a and b are always existent, here we add the left over vertices with label c to cover the number of vertices
        if (comb_size != 0):
            all_combs = list(combinations_with_replacement(H.keys(), comb_size))
            for comb in all_combs:
                new_g = copy.deepcopy(G)
                for item in comb:
                    new_g[item[:-1] + "c"] = set()
                all_empty_g.append(new_g)
        else:
            all_empty_g.append(G)

        ## Now the list all_g is the set of all empty graphs that we want to check for the minor
        for g1 in all_empty_g:
            set_of_all_filled_g = [copy.deepcopy(g1)]
            for begining in H: 
                for end in H[begining]:
                    for graph in set_of_all_filled_g:
                        build_kempe_chain(graph, begining, end, set_of_all_filled_g, begining)
            
            for kempe_graph in set_of_all_filled_g:
                big = get_edges_from_adjacency_list(kempe_graph)
                minor_embd = find_embedding(small, big, random_seed=10, suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]], "6a": [["6a"]]})
                if minor_embd:
                    foundedMinor += 1
                    pass
                else:
                    ## Rechecking with different random seeds, to make sure that the minor is not there
                    no_minor = True
                    for i in range(10):
                        minor_embd = find_embedding(small, big, random_seed=random.randint(1, 1000), suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]], "6a": [["6a"]]})

                        if minor_embd:
                            no_minor = False
                            
                    if no_minor:
                        ## Keeping the minimal counter example
                        if noMinorMinimalGraph is None:
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        elif len(get_edges_from_adjacency_list(kempe_graph)) < len(get_edges_from_adjacency_list(noMinorMinimalGraph)):
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        print("No minor found")
                        print(small)
                        print(kempe_graph)
                        # terminate = True
                        # noMinor += 1
                        # break
            set_of_all_filled_g = []
        index += 1
        print("NEXT STEP", index, len(graphs))

    if noMinorMinimalGraph is not None and noMinorGraph is not None:
        print(noMinorGraph.edges())
        print(noMinorMinimalGraph)
        showGraph(noMinorGraph, "The minor we couldn't find")
        showGraph(getNetworkxGraph(noMinorMinimalGraph), "The minimal counter example")
    print("STATISTICS", foundedMinor, noMinor)
    

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


def generate_graphs_base_plus_edges(G, n):
    vertices = list(G.nodes())
    
    all_possible_edges = list(itertools.combinations(vertices, 2))
    
    base_edges = set(G.edges())
    missing_edges = [e for e in all_possible_edges if e not in base_edges and (e[1], e[0]) not in base_edges]
    
    print("Total missing edges:", len(missing_edges))
    
    graphs = []
    for extra_edges in itertools.combinations(missing_edges, n):
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
    if (start == end):
        return True
    
    for key in G[start]:
        if color(key) == color(start):
            continue
        if key in looked_at:
            continue
        if ((color(start) != color(end) and color(key) == color(end)) or (color(start) == color(end) and color(key) == color(original_begining))) and exists_kempe(G, key, end, original_begining, looked_at.union({key})):
            return True
    return False

def degree(g, vertex):
    return len(g[vertex])

def build_kempe_chain(g1, minor, current, end, set_of_all_filled_g, original_begining, current_chain=None, level=0, index=0):
    if current_chain is None:
        current_chain = [current]
        
    if current == end:
        return

    color_to_look_for = color(original_begining) if color(current) == color(end) else color(end)
    available_vertices = [v for v in g1.keys() if v not in current_chain and color(v) == color_to_look_for and 
                         ((label(v) != 'a' and degree(g1, v) < 20) or (label(v) == 'a' and degree(g1, v) < len(minor[v])))]

    if label(current) == "a":
        available_vertices = [v for v in available_vertices if label(v) != "a"]
    begining_neighbours = g1[current]
    already_built = False
    
    # Sort vertices: non-a vertices by degree (lowest first), then a vertices
    def vertex_sort_key(vertex):
        if label(vertex) == 'a':
            return (1, 0)  # a vertices go last
        d = degree(g1, vertex)
        if d > 3:
            return (2, 0)  # vertices with degree > 3 go to the end
        return (0, d)  # non-a vertices with degree < 3 sorted by degree
    
    available_vertices.sort(key=vertex_sort_key)

    if len(available_vertices) == 0:
        if (exists_kempe(g1, original_begining, end, original_begining, looked_at=set([original_begining]))):
            return
        
        if g1 in set_of_all_filled_g:
            set_of_all_filled_g.remove(g1)
        return

    for vertex in available_vertices:
        if already_built and len(set_of_all_filled_g): ## already extended the inital graph, so let's create a new variation of it, for a different Kempe chain.
            newG = copy.deepcopy(g1)
            newG[current].add(vertex)
            newG[vertex].add(current)
            set_of_all_filled_g.append(newG)
            new_chain = current_chain + [vertex]
            build_kempe_chain(newG, minor, vertex, end, set_of_all_filled_g, original_begining, new_chain, level=level+1, index=index)
        else:
            g1[current].add(vertex)
            g1[vertex].add(current)
            new_chain = current_chain + [vertex]
            build_kempe_chain(g1, minor, vertex, end, set_of_all_filled_g, original_begining, new_chain, level=level+1, index=index)
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

def get_all_graphs_H(n):
    V = [i for i in range(n)]
    E_C6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    all_edges = list(itertools.combinations(V, 2))
    E_rest = [e for e in all_edges if e not in E_C6]

    subsets = []
    for k in range(len(E_rest) + 1):  # From 0 to 9 additional edges
        subsets.extend(itertools.combinations(E_rest, k))
    print("Total number of edge subsets to consider:", len(subsets))

    degree_seq_dict = {}

    non_iso_graphs = []
    for S in subsets:
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_edges_from(E_C6)
        G.add_edges_from(S)
 
        deg_seq = tuple(sorted([d for n, d in G.degree()], reverse=True))
 
        if deg_seq not in degree_seq_dict:
            degree_seq_dict[deg_seq] = [G]
            non_iso_graphs.append(G)
        else:
            is_new = True
            for G_existing in degree_seq_dict[deg_seq]:
                if nx.is_isomorphic(G, G_existing):
                    is_new = False 
                    break

            if is_new:
                degree_seq_dict[deg_seq].append(G) 
                non_iso_graphs.append(G)

    return non_iso_graphs
 

if __name__ == "__main__":
    n = 15 ## build graphs on n vertices
    noMinorMinimalGraph = None
    noMinorGraph = None
    terminate = False
    nominors = []
    # graphs = 
    # graphs = get_all_graphs_K7_minus_one_edge()
    # graphs = generate_graphs_base_plus_two_edges(nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')]))  ## counterexample from paper + 2edges
    # graphs = [nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')])] ## counterexample from paper
    # graphs = [nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '3'), ('0', '4'), ('1', '4')])]
    num_vertices_of_minor = 6
    graphs = get_all_graphs_H(num_vertices_of_minor)
    # graphs = generate_graphs_base_plus_edges(nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '6'), ('6', '0')]), 1) ## counter example
    # graphs = [nx.Graph([('0', '1'), ('0', '2'), ('0', '5'), ('1', '4'), ('1', '6'), ('2', '3'), ('3', '4'), ('6', '4'), ('6', '5'), ('5', '2')])] ## counter example 2
    # graphs = [nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('2', '5')])] ## counter example
    random.shuffle(graphs)
    
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
        suspend_chains = {f"{i}a": [[f"{i}a"]] for i in minor.nodes}
        # number_of_layers = n // num_vertices_of_minor
        number_of_layers = 1
        G = {}
        for i in range(number_of_layers):
            suffix = chr(ord('a') + i)
            for key in H:
                new_key = key[:-1] + suffix
                G[new_key] = set()


        colorings = []
        number_of_colors_left = n - num_vertices_of_minor

        if number_of_colors_left != 0:
            all_combs = list(combinations_with_replacement(H.keys(), number_of_colors_left))
            
            for comb in all_combs:
                new_g = copy.deepcopy(G)
                
                # combinations_with_replacement produces sorted tuples,
                # so grouping will correctly group identical elements.
                for value, group in itertools.groupby(comb):
                    current_layer = number_of_layers
                    # Iterate through all appearances (instances) of the current value
                    for _ in list(group):
                        # Create a new key using the value (excluding its last character) 
                        # and a character representing the current layer.
                        layer_char = chr(ord('a') + current_layer)
                        new_key = value[:-1] + layer_char
                        new_g[new_key] = set()
                        # Move to the next layer for the next instance, even within the same group.
                        current_layer += 1
                
                print(new_g)
                print(comb)
                colorings.append(new_g)
        else:
            colorings.append(G)

        print(len(colorings))
        # all_empty_g = all_empty_g[:1000] ## limit the number of empty graphs to check for the minor
        # all_empty_g = random.sample(all_empty_g, int(len(all_empty_g) * 0.2))
        # print("ASD ", len(all_empty_g))        

        edges = {(min(node, neighbor), max(node, neighbor)) for node, neighbors in H.items() for neighbor in neighbors}
        edges = list(edges)

        coloring_index = 0
        for colroing in colorings:
            print("Coloring:", coloring_index, index)
            set_of_all_filled_g = [copy.deepcopy(colroing)]
            random.shuffle(edges)  # Shuffle edge order for randomness
            
            for beginning, end in edges:
                current_graphs = list(set_of_all_filled_g)
                for graph in current_graphs:
                    build_kempe_chain(
                        graph, H, beginning, end, set_of_all_filled_g, beginning,
                        current_chain=[beginning], level=0, index=index
                    )
            for kempe_graph in set_of_all_filled_g:
                big = get_edges_from_adjacency_list(kempe_graph)
                minor_embd = find_embedding(small, big, random_seed=10, suspend_chains=suspend_chains)
                if minor_embd:
                    foundedMinor += 1
                else:
                    ## Rechecking with different random seeds, to make sure that the minor is not there
                    no_minor = True
                    for i in range(10):
                        minor_embd = find_embedding(small, big, random_seed=random.randint(1, 1000), suspend_chains=suspend_chains)

                        if minor_embd:
                            no_minor = False
                            
                    if no_minor:
                        ## Keeping the minimal counter example
                        nominors.append(minor)
                        if noMinorMinimalGraph is None:
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        elif len(get_edges_from_adjacency_list(kempe_graph)) < len(get_edges_from_adjacency_list(noMinorMinimalGraph)):
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        print("No minor found")
                        print(small)
                        print(kempe_graph)
                        terminate = True
                        noMinor += 1
                        break
            coloring_index += 1
            set_of_all_filled_g = []
        index += 1
        print("NEXT STEP", index, len(graphs))

    if noMinorMinimalGraph is not None and noMinorGraph is not None:
        print(noMinorGraph.edges())
        print(noMinorMinimalGraph)
        showGraph(noMinorGraph, "The minor we couldn't find")
        showGraph(getNetworkxGraph(noMinorMinimalGraph), "The minimal counter example")

    for m in nominors:
        showGraph(m, "The minor we couldn't find")
    print("STATISTICS", foundedMinor, noMinor)
    

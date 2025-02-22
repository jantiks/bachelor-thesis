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
    # Get the list of vertices from the base graph
    vertices = list(G.nodes())
    
    # Compute all possible edges in a complete graph on these vertices.
    all_possible_edges = list(itertools.combinations(vertices, 2))
    
    # The base graph already has some edges; compute the missing ones.
    # (Since the graph is undirected, the tuple order doesn't matter.)
    base_edges = set(G.edges())
    missing_edges = [e for e in all_possible_edges if e not in base_edges and (e[1], e[0]) not in base_edges]
    
    print("Total missing edges:", len(missing_edges))
    
    # Generate new graphs by adding two edges from the missing set to the base graph.
    graphs = []
    for extra_edges in itertools.combinations(missing_edges, 4):
        # Create a copy of the base graph
        G_new = G.copy()
        G_new.add_edges_from(extra_edges)
        graphs.append(G_new)
    
    print("Total graphs generated (base + 2 extra edges):", len(graphs))
    return graphs

def get_all_graphs_K7_minus_one_edge():
    V = list(range(7))  # vertices 0 through 6
    all_edges = list(itertools.combinations(V, 2))  # all 21 edges of K7
    print("Total number of edges in K7:", len(all_edges))
    
    graphs = []
    # For each edge, remove it from the complete graph
    for missing_edge in all_edges:
        G = nx.Graph()
        G.add_nodes_from(V)
        # Add all edges except the one being removed
        edges = [e for e in all_edges if e != missing_edge]
        G.add_edges_from(edges)
        graphs.append(G)
    
    print("Total number of graphs generated (K7 minus one edge):", len(graphs))
    return graphs

def get_all_graphs_K7_minus_two_edges():
    V = list(range(7))  # vertices 0 through 6
    all_edges = list(itertools.combinations(V, 2))  # all 21 edges of K7
    print("Total number of edges in K7:", len(all_edges))
    
    graphs = []
    # Iterate over all possible pairs of edges to remove
    for missing_edges in itertools.combinations(all_edges, 1):
        G = nx.Graph()
        G.add_nodes_from(V)
        # Add all edges except the two that are being removed
        edges = [e for e in all_edges if e not in missing_edges]
        G.add_edges_from(edges)
        graphs.append(G)
    
    print("Total number of graphs generated (K7 minus two edges):", len(graphs))
    return graphs

def get_all_graphs_H(n):
    V = [i for i in range(n)]
    E_C6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    all_edges = list(itertools.combinations(V, 2))
    E_rest = [e for e in all_edges if e not in E_C6]

    # Generate all subsets of edges to add to C6
    subsets = []
    for k in range(len(E_rest) + 1):  # From 0 to 9 additional edges
        subsets.extend(itertools.combinations(E_rest, k))
    print("Total number of edge subsets to consider:", len(subsets))

    # Dictionary to group graphs by degree sequence for efficient isomorphism checks
    degree_seq_dict = {}
    # List to store the final non-isomorphic graphs
    non_iso_graphs = []

    for S in subsets:
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_edges_from(E_C6)
        G.add_edges_from(S)
        # Compute degree sequence as a graph invariant
        deg_seq = tuple(sorted([d for n, d in G.degree()], reverse=True))
        if deg_seq not in degree_seq_dict:
            degree_seq_dict[deg_seq] = [G]
            non_iso_graphs.append(G)
        else:
            is_new = True
            for G_existing in degree_seq_dict[deg_seq]:
                # Use VF2 algorithm for isomorphism checking
                if nx.is_isomorphic(G, G_existing):
                    is_new = False
                    break
            if is_new:
                degree_seq_dict[deg_seq].append(G)
                non_iso_graphs.append(G)

    return non_iso_graphs


def zg(G):
    H = nx.Graph()

    # Step 1: Add vertices with label 'a' (without edges)
    for node in G.nodes():
        H.add_node(node)

    # Step 2: Add vertices with label 'b' and copy edges from G
    for u, v in G.edges():
        
        H.add_edge(f"{u.removesuffix('a')}b", f"{v.removesuffix('a')}b")

    # Step 3: Add edges between 'a' and 'b' vertices based on adjacency in G
    for u, v in G.edges():
        H.add_edge(u, f"{v.removesuffix('a')}b")
        H.add_edge(v, f"{u.removesuffix('a')}b")  # Ensure symmetry for undirected graph

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

def build_kempe_chain(g1, begining, end, set_of_all_g1, original_begining, level=0):
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
            # print('begining: ', begining, 'vertex: ', vertex, 'end: ', end, 'original_begining: ', original_begining)
            if already_built:
                newG = copy.deepcopy(g1)
                newG[begining].add(vertex)
                newG[vertex].add(begining)
                # showGraph(getNetworkxGraph(newG))
                set_of_all_g1.append(newG)
                build_kempe_chain(newG, vertex, end, set_of_all_g1, original_begining, level=level+1)
            else:
                g1[begining].add(vertex)
                g1[vertex].add(begining)
                # showGraph(getNetworkxGraph(g1))
                build_kempe_chain(g1, vertex, end, set_of_all_g1, original_begining, level=level+1)
                already_built = True

def get_edges_from_adjacency_list(adjacency_list):
    edges = set()  # Use a set to avoid duplicate edges
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            # Add each edge as a sorted tuple (smallest first) to avoid duplicates
            edges.add(tuple(sorted((node, neighbor))))
    return list(edges)  

def showGraph(G):
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=15, font_color='black')
    print(G)
    plt.title("Graph G: Cycle of 10 Nodes")
    plt.show()

def getNetworkxGraph(dict):
    G = nx.Graph()

    # Add edges to the graph by iterating over the dictionary
    for node, neighbors in dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    return G

if __name__ == "__main__":
    noMinorMinimalGraph = None
    terminate = False
    graphs = get_all_graphs_K7_minus_one_edge()
    print(len(graphs), graphs)
    index = 0
    foundedMinor = 0
    noMinor = 0
    # graphs = generate_graphs_base_plus_two_edges(nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')]))
    # graphs = [nx.Graph([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '6'), ('6', '3')])]
    for minor in graphs:
        if terminate:
            print("Terminating found minor")
            break
        n = 15
        H = {
            f"{node}a": set([f"{neighbor}a" for neighbor in neighbors])
            for node, neighbors in minor.adj.items()
        }
        # Create a mapping from node to index

        # Initialize the adjacency matrix

        G = {}
        small = get_edges_from_adjacency_list(H)

        for i in range(2):
            suffix = chr(ord('a') + i)
            for key in H:
                new_key = key[:-1] + suffix
                G[new_key] = set()


        all_g = []
        comb_size = n % 14
        if (comb_size != 0):
            all_combs = list(combinations_with_replacement(H.keys(), comb_size))
            for comb in all_combs:
                new_g = copy.deepcopy(G)
                for item in comb:
                    new_g[item[:-1] + "c"] = set()
                all_g.append(new_g)
        else:
            all_g.append(G)

        # final_results = []
        
        all_g_index = 0
        for g1 in all_g:
            all_g_index += 1
            set_of_all_g1 = [copy.deepcopy(g1)]
            for begining in H: 
                for end in H[begining]:
                    for graph in set_of_all_g1:
                        build_kempe_chain(graph, begining, end, set_of_all_g1, begining)
            
            minorCheckingIndex = 0
            for kempe_graph in set_of_all_g1:
                print("KEMPE GRAPH", minorCheckingIndex, index, all_g_index, len(all_g), len(set_of_all_g1))
                # showGraph(getNetworkxGraph(kempe_graph))
                minorCheckingIndex += 1
                big = get_edges_from_adjacency_list(kempe_graph)
                minor_embd = find_embedding(small, big, random_seed=10, suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]], "6a": [["6a"]]})
                if minor_embd:
                    foundedMinor += 1
                    pass
                else:
                    no_minor = True
                    for i in range(10):
                        minor_embd = find_embedding(small, big, random_seed=random.randint(1, 1000), suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]], "6a": [["6a"]]})

                        if minor_embd:
                            no_minor = False
                            
                    
                    if no_minor:
                        if noMinorMinimalGraph is None:
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        elif len(get_edges_from_adjacency_list(kempe_graph)) < len(get_edges_from_adjacency_list(noMinorMinimalGraph)):
                            noMinorMinimalGraph = kempe_graph
                            noMinorGraph = minor
                        print("No minor found")
                        print(small)
                        print(kempe_graph)
                        print("No minor")
                        terminate = True
                        noMinor += 1
                        break
            # final_results.extend(set_of_all_g1)
            set_of_all_g1 = []
        index += 1
        print("NEXT STEP", index, len(graphs))

    print(noMinorGraph.edges())
    print(noMinorMinimalGraph)
    showGraph(noMinorGraph)
    showGraph(getNetworkxGraph(noMinorMinimalGraph))
    print("ASD STATISTICS", foundedMinor, noMinor)
    

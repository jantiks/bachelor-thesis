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

def build_kempe_chain(g1, begining, end, set_of_all_g1, original_begining):
    if exists_kempe(g1, begining, end, original_begining):
        return

    begining_neighbours = g1[begining]

    already_built = False
    for vertex in g1.keys():
        if (label(begining) == "a" and label(vertex) == "a"):
            continue
        if (vertex in begining_neighbours):
            continue

    
        if (color(begining) == color(end) and color(vertex) == color(original_begining)):
            if already_built:
                newG = copy.deepcopy(g1)
                newG[begining].add(vertex)
                newG[vertex].add(begining)
                set_of_all_g1.append(newG)
                build_kempe_chain(newG, vertex, end, set_of_all_g1, original_begining)
            else:
                g1[begining].add(vertex)
                g1[vertex].add(begining)
                build_kempe_chain(g1, vertex, end, set_of_all_g1, original_begining)
                already_built = True
        elif (color(begining) == color(original_begining) and color(vertex) == color(end)):
            if already_built:
                newG = copy.deepcopy(g1)
                newG[begining].add(vertex)
                newG[vertex].add(begining)
                set_of_all_g1.append(newG)
                build_kempe_chain(newG, vertex, end, set_of_all_g1, original_begining)
            else:
                g1[begining].add(vertex)
                g1[vertex].add(begining)
                build_kempe_chain(g1, vertex, end, set_of_all_g1, original_begining)
                already_built = True

def get_edges_from_adjacency_list(adjacency_list):
    edges = set()  # Use a set to avoid duplicate edges
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            # Add each edge as a sorted tuple (smallest first) to avoid duplicates
            edges.add(tuple(sorted((node, neighbor))))
    return list(edges)  

if __name__ == "__main__":
    # big = nx.Graph([('0b', '1b'), ('1b', '2b'), ('2b', '3b'), ('3b', '0b'), ('2b', '0b'), ('1b', '3b'), ('3b', '230d')])
    # small = nx.Graph([('0b', '1b'), ('1b', '2b'), ('2b', '3b'), ('3b', '0b')])


    # isomatcher = nx.isomorphism.GraphMatcher(big, small)
    # if isomatcher.subgraph_is_isomorphic():
    #     print("Isomorphic")
    # else:
    #     print("NON-Isomorphic")
    # mode = sys.argv[1] if len(sys.argv) > 1 else "automated"
    # main(mode)

    # small = [('0a', '1a'), ('1a', '2a'), ('2a', '3a'), ('3a', '4a'), ('4a', '5a'), ('5a', '0a'), ('1a', '4a')]
    small = [('a1', 'b1'), ('b1', 'c1'), ('c1', 'd1'), ('d1', 'e1'), ('e1', 'f1'), ('a1', 'f1'), ('b1', 'f1'), ('a1', 'd1')]

    big = [('a1', 'b3'), ('b3', 'a3'), ('a3', 'b1'), # a1-b1 kempe chain
           ('f1', 'a2'), ('a2', 'f2'), ('f2', 'a1'), # a1-f1 kempe chain
           ('f1', 'b2'), ('b2', 'f2'), ('f2', 'b3'), # b1-f1 kempe chain
           ('f1', 'e2'), ('e2', 'f2'), ('f2', 'e1'), # e1-f1 kempe chain
        #    ('f3', 'd1'), ('f3', 'd2'), ('d2', 'f1'), # d1-f1 kempe chain
           ('e1', 'd2'), ('d2', 'e2'), ('e2', 'd1'), # d1-e1 kempe chain
           ('c1', 'd2'), ('d2', 'c2'), ('c2', 'd1'), # c1-d1 kempe chain
           ('c1', 'b3'), ('b3', 'c2'), ('c2', 'b1'), # c1-b1 kempe chain
           ('d1', 'a2'), ('a2', 'd3'), ('d3', 'a1'), # a1-d1 kempe chain
        #    ('e1', 'b3'), ('b3', 'e2'), ('e2', 'b1'), # b1-e1 kempe chain
        #    ('e1', 'a2'), ('a2', 'e2'), ('e2', 'a1'), # a1-e1 kempe chain
        #    ('e1', 'c3'), ('c3', 'e2'), ('e2', 'c1'), # c1-e1 kempe chain
        #    ('d1', 'b2'), ('b2', 'd2'), ('d2', 'b1'), # b1-d1 kempe chain
           ]

    # tSmall = nx.Graph(small)
    # big = zg(tSmall).edges()

    # Find an assignment of sets of square variables to the triangle variables
    embedding = find_embedding(small, big, random_seed=10, suspend_chains={"a1": [['a1']], "b1": [["b1"]], "c1": [["c1"]], "d1": [["d1"]], "e1": [["e1"]], "f1": [["f1"]]})
    print(embedding)
    terminate = False
    graphs = get_all_graphs_H(6)
    index = 0
    print("FAILING MINOR", graphs[46].edges())
    graphs = [graphs[47]]
    for minor in graphs:
        if terminate:
            print("Terminating found minor")
            break
        n = 16
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
        comb_size = n % 12
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
        
        for g1 in all_g:
            set_of_all_g1 = [copy.deepcopy(g1)]
            for begining in H: 
                for end in H[begining]:
                    for graph in set_of_all_g1:
                        build_kempe_chain(graph, begining, end, set_of_all_g1, begining)
            
            for kempe_graph in set_of_all_g1:
                big = get_edges_from_adjacency_list(kempe_graph)
                minor_embd = find_embedding(small, big, random_seed=10, suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]]})
                if minor_embd:
                    pass
                    # print("Found minor")
                else:
                    no_minor = True
                    for i in range(10):
                        minor_embd = find_embedding(small, big, random_seed=random.randint(1, 1000), suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]]})
                        if minor_embd:
                            no_minor = False
                            
                    
                    if no_minor:
                        print("No minor found")
                        print(small)
                        print(kempe_graph)
                        print("No minor")
                        terminate = True
                        break
            # final_results.extend(set_of_all_g1)
            set_of_all_g1 = []
        index += 1
        print("NEXT STEP", index, len(graphs))

            

        # print("ASD SETOF ALL G", len(final_results), len(all_g))
        # small = get_edges_from_adjacency_list(H)
        # for graph in final_results:
        #     big = get_edges_from_adjacency_list(graph)
        #     minor_embd = find_embedding(small, big, random_seed=10, suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]]})
        #     if minor_embd:
        #         pass
        #         # print("Found minor")
        #     else:
        #         no_minor = True
        #         for i in range(10):
        #             minor_embd = find_embedding(small, big, random_seed=random.randint(1, 1000), suspend_chains={"0a": [['0a']], "1a": [["1a"]], "2a": [["2a"]], "3a": [["3a"]], "4a": [["4a"]], "5a": [["5a"]]})
        #             if minor_embd:
        #                 no_minor = False
                        
                
        #         if no_minor:
        #             print("No minor found")
        #             print(small)
        #             print(graph)
        #             print("No minor")
        #             terminate = True
        #             break
    

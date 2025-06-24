import networkx as nx
import itertools
from copy import deepcopy
from minorminer import find_embedding
import matplotlib.pyplot as plt
import random
import numpy as np
import igraph as ig
from itertools import combinations_with_replacement
import copy
import random
import subprocess


def even_weight_bitstrings(n):
    return [
        format(i, f"0{n}b")
        for i in range(2**n)
        if format(i, f"0{n}b").count("1") % 2 == 0
    ]


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def generate_graphs_base_plus_edges(G, n):
    vertices = list(G.nodes())

    all_possible_edges = list(itertools.combinations(vertices, 2))

    base_edges = set(G.edges())
    missing_edges = [
        e
        for e in all_possible_edges
        if e not in base_edges and (e[1], e[0]) not in base_edges
    ]

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
    if start == end:
        return True
    for key in G[start]:
        if key == end:
            return True
        if color(key) == color(start):
            continue
        if key in looked_at:
            continue
        if (color(start) != color(end) and color(key) == color(end)) or (
            color(start) == color(end) and color(key) == color(original_begining)
        ):
            return exists_kempe(G, key, end, original_begining, looked_at.union({key}))
    return False


def degree(g, vertex):
    return len(g[vertex])


def get_edges_from_adjacency_list(adjacency_list):
    edges = set()
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            edges.add(tuple(sorted((node, neighbor))))
    return list(edges)


def showGraph(G, title="Graph"):
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=15,
        font_color="black",
    )
    print(G)
    plt.suptitle(title)
    plt.show()


def getNetworkxGraph(dict):
    G = nx.Graph()
    for node, neighbors in dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    return G

def get_graphs_from_geng(n):
    proc = subprocess.run(['geng', str(n), "-c", "-d2"], capture_output=True, text=True)
    graphs = []
    for line in proc.stdout.strip().splitlines():
        G = nx.from_graph6_bytes(line.encode())
        graphs.append(G)
    return graphs

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


def testIfMinorExists(minor, parent, suspend_chains):
    minor_embd = find_embedding(
        minor, parent, random_seed=10, suspend_chains=suspend_chains
    )

    if minor_embd:
        return True

    for i in range(10):
        minor_embd = find_embedding(
            minor,
            parent,
            random_seed=random.randint(1, 1000),
            suspend_chains=suspend_chains,
        )

        if minor_embd:
            return True

    return False

## This function is helper function to manually validate again the possible counter examples the program found
def validate_result():
    counter_example = {
        "0a": {"1a", "3a", "4b", "5a", "2a"},
        "1a": {"3b", "0a", "4b", "5a", "2a"},
        "2a": {"1a", "0a", "5a", "3a"},
        "3a": {"0a", "4b", "1b", "2a"},
        "4a": {"1b", "0b", "5a", "3b"},
        "5a": {"1a", "0a", "2a", "4a"},
        "0b": {"4a", "4c"},
        "0c": {"4b", "4c"},
        "1b": {"4a", "3c", "3a", "4c"},
        "1c": {"4b", "3c", "3b", "4c"},
        "3b": {"4a", "4c", "1a", "1c", "4b"},
        "3c": {"1c", "1b", "4b", "4c"},
        "4b": {"3b", "1a", "0a", "1c", "3c", "3a", "0c"},
        "4c": {"3b", "1c", "1b", "3c", "0b", "0c"},
    }
    minor = [
        ("3a", "4a"),
        ("0a", "3a"),
        ("1a", "5a"),
        ("0a", "2a"),
        ("4a", "5a"),
        ("1a", "4a"),
        ("1a", "3a"),
        ("2a", "5a"),
        ("1a", "2a"),
        ("0a", "1a"),
        ("0a", "5a"),
        ("2a", "3a"),
        ("0a", "4a"),
    ]
    suspend_chains = {
        "0a": [["0a"]],
        "1a": [["1a"]],
        "2a": [["2a"]],
        "3a": [["3a"]],
        "4a": [["4a"]],
        "5a": [["5a"]],
    }

    vertices = even_weight_bitstrings(5)

    # Build the Clebsch graph
    G = nx.Graph()
    G.add_nodes_from(vertices)

    for u, v in itertools.combinations(vertices, 2):
        if hamming_distance(u, v) == 2:
            G.add_edge(u, v)
    minor = nx.complete_graph(5)
    for i in range(100):
        if testIfMinorExists(minor.edges(), G.edges(), suspend_chains):
            print("FASLE CALL")
            return

    input("With probability almost 1, this is a counterexample.")

def build(
    G, chains_to_build, current, end, visited, available_vertices, minor, suspend_chains
):
    if current == end:
        if not chains_to_build:
            if testIfMinorExists(
                minor, get_edges_from_adjacency_list(G), suspend_chains
            ):
                return
            else:
                if not testIfMinorExists(
                    minor, get_edges_from_adjacency_list(G), suspend_chains
                ):
                    print("FOUND COUNTEREXAMPLE")
                    print("Big graph:", G)
                    print("Minor graph:", minor)
                    input("WAIT: FOUND COUNTEREXAMPLE")
            return

        next_start, next_end = chains_to_build[0]
        remaining_chains = chains_to_build[1:]

        color_to_look_for = color(next_end)
        next_available = [v for v in G.keys() if color(v) == color_to_look_for]

        build(
            G,
            remaining_chains,
            next_start,
            next_end,
            {next_start},
            next_available,
            minor,
            suspend_chains,
        )
        return

    # Process current chain
    for vertex in available_vertices:
        newG = copy.deepcopy(G)
        newG[current].add(vertex)
        newG[vertex].add(current)

        color_to_look_for = (
            color(current) if color(vertex) == color(end) else color(end)
        )
        next_available = [
            v
            for v in G.keys()
            if v not in visited.union({vertex}) and color(v) == color_to_look_for
        ]

        build(
            newG,
            chains_to_build,
            vertex,
            end,
            visited.union({vertex}),
            next_available,
            minor,
            suspend_chains,
        )

def generateColorings(H, n, num_vertices_of_minor):
    number_of_layers = 1
    G = {}
    for i in range(number_of_layers):
            suffix = chr(ord("a") + i)
            for key in H:
                new_key = key[:-1] + suffix
                G[new_key] = set()

    colorings = []
    number_of_colors_left = n - num_vertices_of_minor

    if number_of_colors_left != 0:
            all_combs = list(
                combinations_with_replacement(H.keys(), number_of_colors_left)
            )

    for comb in all_combs:
                new_g = copy.deepcopy(G)
                for value, group in itertools.groupby(comb):
                    current_layer = number_of_layers
                    for _ in list(group):
                        layer_char = chr(ord("a") + current_layer)
                        new_key = value[:-1] + layer_char
                        new_g[new_key] = set()
                        current_layer += 1

                print(new_g)
                print(comb)
                colorings.append(new_g)
    else:
            colorings.append(G)

    print(len(colorings))
    return colorings

def main(n, num_vertices_of_minor, minor_graphs):
    index = 0

    for minor in minor_graphs:
        # builidng the graph H in the format suitable for the minorminer
        H = {
            f"{node}a": set([f"{neighbor}a" for neighbor in neighbors])
            for node, neighbors in minor.adj.items()
        }

        small = get_edges_from_adjacency_list(H)
        suspend_chains = {f"{i}a": [[f"{i}a"]] for i in minor.nodes}

        colorings = generateColorings(H, n, num_vertices_of_minor)

        edges = {
            (min(node, neighbor), max(node, neighbor))
            for node, neighbors in H.items()
            for neighbor in neighbors
        }
        edges = list(edges)

        coloring_index = 0
        for colroing in colorings:
            print("Coloring:", coloring_index, index, len(minor_graphs))
            color_to_look_for = color(edges[0][1])
            available_vertices = [
                v for v in colroing.keys() if color(v) == color_to_look_for
            ]
            build(
                colroing,
                edges,
                edges[0][0],
                edges[0][1],
                set([edges[0][0]]),
                available_vertices,
                small,
                suspend_chains,
            )
            coloring_index += 1
        index += 1
        print("NEXT STEP", index, len(minor_graphs))

if __name__ == "__main__":
    #### Uncomment each part to run the specific test case
    #### ------------------------------------------------------------####
    #### 1. the counterexample from Kriesell's and Mohr's paper for K_7.
    # minors = [
    #     nx.Graph(
    #         [
    #             ("0", "1"),
    #             ("1", "2"),
    #             ("2", "3"),
    #             ("3", "4"),
    #             ("4", "5"),
    #             ("5", "0"),
    #             ("0", "6"),
    #             ("6", "3"),
    #         ]
    #     )
    # ]
    # main(14, 7, minors) - this will find the Z(H) and it's supergraphs. 
    #### ------------------------------------------------------------####

    #### 2. Checking for all spanning subgraphs of K_6 with supergraphs with 13 vertices - will take around 5-6 hours, for smaller graphs i.e shorter time run `main(12, 6, minors)`.
    minors = get_graphs_from_geng(6)
    main(13, 6, minors)
    #### ------------------------------------------------------------####




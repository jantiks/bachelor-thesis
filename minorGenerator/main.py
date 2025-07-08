import networkx as nx
import itertools
from minorminer import find_embedding
import matplotlib.pyplot as plt
import random
from itertools import combinations_with_replacement
import copy
import random
import subprocess


# # -- HELPER FUNCTIONS -- ##
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
    proc = subprocess.run(['geng', str(n), "-c"], capture_output=True, text=True)
    graphs = []
    for line in proc.stdout.strip().splitlines():
        G = nx.from_graph6_bytes(line.encode())
        graphs.append(G)
    return graphs


# # TestMinor
def test_minor(minor, parent, suspend_chains):
    minor_embd = find_embedding(
        minor, parent, random_seed=10, suspend_chains=suspend_chains
    )

    if minor_embd:
        return True

    for _ in range(30):
        minor_embd = find_embedding(
            minor,
            parent,
            random_seed=random.randint(1, 100000),
            suspend_chains=suspend_chains,
        )

        if minor_embd:
            return True

    return False


# # This function is helper function to manually validate again the possible counter examples the program found
def __validate_result():
    counter_example = {'0a': {'4b', '3b'}, '1a': {'4b', '6b', '5b'}, '2a': {'6b', '5b'}, '3a': {'6b', '0b'}, '4a': {'1b', '0b'}, '5a': {'1b', '2b'}, '6a': {'1b', '3b', '2b'}, '0b': {'4b', '3b', '3a', '4a'}, '1b': {'4a', '5b', '4b', '6a', '6b', '5a'}, '2b': {'6a', '6b', '5b', '5a'}, '3b': {'0b', '6b', '0a', '6a'}, '4b': {'1a', '1b', '0a', '0b'}, '5b': {'1a', '1b', '2a', '2b'}, '6b': {'2a', '1a', '3b', '2b', '1b', '3a'}}
    minor = [('0a', '4a'), ('2a', '6a'), ('1a', '5a'), ('3a', '6a'), ('0a', '3a'), ('1a', '4a'), ('1a', '6a'), ('2a', '5a')]
    suspend_chains = {f"{i}": [[f"{i}"]] for i in nx.Graph(minor).nodes}

    G = nx.Graph(minor)
    print("EDGES", len(getNetworkxGraph(counter_example).edges()))

    showGraph(G, "Minor Graph")
    showGraph(getNetworkxGraph(counter_example), "Counter Example Graph")

    for i in range(100):
        if test_minor(minor, get_edges_from_adjacency_list(counter_example), suspend_chains):
            print("FALSE CALL")
            return

    input("With probability almost 1, this is a counterexample.")


# # BuildKempeChains
def build_kempe_chains(
    G, chains_to_build, current, end, visited, available_vertices, minor, suspend_chains
):
    if current == end:
        if not chains_to_build:
            if not test_minor(
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

        build_kempe_chains(
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
        G[vertex].add(current)
        G[current].add(vertex)
        
        next_available = [
            v
            for v in G.keys()
            if v not in visited.union({vertex}) and color(v) == color(current)
        ]

        build_kempe_chains(
            G,
            chains_to_build,
            vertex,
            end,
            visited.union({vertex}),
            next_available,
            minor,
            suspend_chains,
        )
        
        G[current].discard(vertex)
        G[vertex].discard(current)


# # GenerateColorings
def generate_colorings(H, n, num_vertices_of_minor):
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

        colorings.append(new_g)
    else:
        colorings.append(G)

    return colorings


def main(n, num_vertices_of_minor, minor_graphs):
    index = 0

    for minor in minor_graphs:
        # builidng the graph H in the format suitable for the minorminer library
        H = { f"{node}a": set([f"{neighbor}a" for neighbor in neighbors]) for node, neighbors in minor.adj.items()}
        suspend_chains = {f"{i}a": [[f"{i}a"]] for i in minor.nodes}
        colorings = generate_colorings(H, n, num_vertices_of_minor)
        edges = list({
            (min(node, neighbor), max(node, neighbor))
            for node, neighbors in H.items()
            for neighbor in neighbors
        })

        coloring_index = 0
        
        # # The loop below described as Algorithm 4 in the thesis.
        for colroing in colorings:
            print("Coloring:", coloring_index, index, len(minor_graphs))
            color_to_look_for = color(edges[0][1])
            available_vertices = [
                v for v in colroing.keys() if color(v) == color_to_look_for
            ]
            build_kempe_chains(
                colroing,
                edges,
                edges[0][0],
                edges[0][1],
                set([edges[0][0]]),
                available_vertices,
                get_edges_from_adjacency_list(H),
                suspend_chains,
            )
            coloring_index += 1
        index += 1
        print("NEXT STEP", index, len(minor_graphs))


if __name__ == "__main__":
    #### Uncomment each part to run the specific test case
    #### ------------------------------------------------------------####
    #### 1. the counterexample from Kriesell's and Mohr's paper for K_7. takes around 10-20 minutes to run.
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
    # minor_size = 7
    # parent_size = 14
    # main(parent_size, minor_size, minors) # - this will find the Z(H) and it's supergraphs. 
    #### ------------------------------------------------------------####

    #### 2. Checking for all spanning subgraphs of K_6 with supergraphs with 13 vertices - will take around 10-12 hours. For smaller graphs i.e shorter time decrease the parent_size.
    # minor_size = 6
    # minors = get_graphs_from_geng(minor_size)
    # parent_size = 13
    # main(parent_size, minor_size, minors)
    #### ------------------------------------------------------------####

    #### 3. Checking for all spanning subgraphs of K_7 with supergraphs with 14 vertices.
    # minor_size = 7
    # minors = get_graphs_from_geng(minor_size)
    # parent_size = 14
    # main(parent_size, minor_size, minors)
    #### ------------------------------------------------------------####
    
    #### 4. the counterexample we found for K_7. takes around 10-20 minutes to run.
    minors = [
        nx.Graph(
            [
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("3", "4"),
                ("4", "5"),
                ("5", "6"),
                ("6", "0"),
                ("6", "2"),
            ]
        )
    ]
    minor_size = 7
    parent_size = 14
    main(parent_size, minor_size, minors)  # - this will find the Z(H) and it's supergraphs. 
    #### ------------------------------------------------------------####

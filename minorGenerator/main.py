import networkx as nx
from minorminer import find_embedding
import matplotlib.pyplot as plt
import random
from itertools import combinations_with_replacement, groupby
import copy
import subprocess
import json
import logging
from typing import Dict, Set, List, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# # -- HELPER FUNCTIONS -- ##
def zg(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    for node in G.nodes():
        H.add_node(node)

    for u, v in G.edges():
        H.add_edge(f"{u.removesuffix('a')}b", f"{v.removesuffix('a')}b")

    for u, v in G.edges():
        H.add_edge(u, f"{v.removesuffix('a')}b")
        H.add_edge(v, f"{u.removesuffix('a')}b")

    return H


def color(vertex: str) -> str:
    return vertex[:-1]


def label(vertex: str) -> str:
    return vertex[-1]


def degree(g: Dict[str, Set[str]], vertex: str) -> int:
    return len(g[vertex])

def get_edges_from_adjacency_list(adjacency_list: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
    edges = set()
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            edges.add(tuple(sorted((node, neighbor))))
    return list(edges)


def show_graph(G: nx.Graph, title: str = "Graph") -> None:
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=15,
        font_color="black",
    )
    plt.suptitle(title)
    plt.show()


def get_networkx_graph(adj_dict: Dict[str, Set[str]]) -> nx.Graph:
    G = nx.Graph()
    for node, neighbors in adj_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    return G

def save_counterexample(
    parent: Dict[str, Set[str]],
    minor: List[Tuple[str, str]],
    filepath: str = "counterexamples.txt"
) -> None:
    """
    Save a counterexample to a file in JSON Lines format.

    Args:
        G (dict): The parent graph adjacency list.
        minor (dict): The minor graph adjacency list.
        filename (str): Path to the output file.
    """
    try:
        with open(filepath, "a") as f:
            f.write("=== FOUND COUNTEREXAMPLE ===\n")
            f.write("Parent graph G:\n")
            f.write(repr(parent) + "\n")
            f.write("Minor graph H:\n")
            f.write(repr(minor) + "\n")
            f.write("\n\n")
            logging.info(f"Counterexample logged to {filepath}")
    except IOError as e:
        logging.error(f"Failed to write to file {filepath}: {e}")
        return
    


def get_graphs_from_geng(n: int) -> List[nx.Graph]:
    """
    Generates all connected graphs with n vertices and minimum degree at least 2.

    Args:
        n (int): The number of vertices.

    Returns:
        list of networkx.Graph: A list of generated graphs.

    Raises:
        RuntimeError: If the geng command fails.
        FileNotFoundError: If the geng command is not found.
        ValueError: If an invalid response is returned by geng.
    """
    try:
        proc = subprocess.run(
            ['geng', str(n), "-c", "-d2"],
            capture_output=True,
            text=True,
            check=True  # This raises CalledProcessError on non-zero exit codes
        )
    except FileNotFoundError as e:
        raise FileNotFoundError("The 'geng' command was not found. Make sure nauty is installed and 'geng' is in your PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"'geng' failed with exit code {e.returncode}: {e.stderr.strip()}") from e

    graphs = []
    try:
        for line in proc.stdout.strip().splitlines():
            G = nx.from_graph6_bytes(line.encode())
            graphs.append(G)
    except Exception as e:
        raise ValueError(f"Error parsing the output of geng: {e}") from e

    return graphs


# # TestMinor
def test_minor(
    minor: List[Tuple[str, str]],
    parent: List[Tuple[str, str]],
    suspend_chains: Dict[str, List[List[str]]]
) -> bool:
    """
    Implementation of TestMinor subroutine from the thesis. 
    Attempts to find an embedding of a minor graph into a parent graph.

    This function tries to embed the `minor` graph into the `parent` graph using
    the `find_embedding` function. If the initial attempt fails, it retries multiple times with different random seeds.

    Args:
        minor (list of lists): Adjacency list of the minor graph.
        parent (list of lists): Adjacency list of the parent graph.
        suspend_chains (dict): A dictionary mapping root vertex labels (str) to lists of lists of those vertices,
            used by the minorminer library to specify root vertices of the minor for embedding.

    Returns:
        bool: True if an embedding is found in any attempt; False otherwise.
    """
    
    test_range = 30
    minor_embd = find_embedding(
        minor, parent, suspend_chains=suspend_chains
    )

    if minor_embd:
        return True

    for _ in range(test_range):
        minor_embd = find_embedding(
            minor,
            parent,
            random_seed=random.randint(1, 100000),
            suspend_chains=suspend_chains,
        )

        if minor_embd:
            return True

    return False


# # This function is helper function to manually validate again the possible counter examples the program found. 
def _validate_result():
    counter_example = {'0a': {'4b', '3b'}, '1a': {'4b', '6b', '5b'}, '2a': {'6b', '5b'}, '3a': {'6b', '0b'}, '4a': {'1b', '0b'}, '5a': {'1b', '2b'}, '6a': {'1b', '3b', '2b'}, '0b': {'4b', '3b', '3a', '4a'}, '1b': {'4a', '5b', '4b', '6a', '6b', '5a'}, '2b': {'6a', '6b', '5b', '5a'}, '3b': {'0b', '6b', '0a', '6a'}, '4b': {'1a', '1b', '0a', '0b'}, '5b': {'1a', '1b', '2a', '2b'}, '6b': {'2a', '1a', '3b', '2b', '1b', '3a'}}
    minor = [('0a', '4a'), ('2a', '6a'), ('1a', '5a'), ('3a', '6a'), ('0a', '3a'), ('1a', '4a'), ('1a', '6a'), ('2a', '5a')]
    suspend_chains = {f"{i}": [[f"{i}"]] for i in nx.Graph(minor).nodes}

    G = nx.Graph(minor)
    logging.info("EDGES", len(get_networkx_graph(counter_example).edges()))

    show_graph(G, "Minor Graph")
    show_graph(get_networkx_graph(counter_example), "Counter Example Graph")

    for i in range(100):
        if test_minor(minor, get_edges_from_adjacency_list(counter_example), suspend_chains):
            logging.info("FALSE CALL")
            return

    logging.info("With probability almost 1, this is a counterexample.")


# # BuildKempeChains
def build_kempe_chains(
    G: Dict[str, Set[str]],
    chains_to_build: List[Tuple[str, str]],
    current: str,
    end: str,
    visited: Set[str],
    available_vertices: List[str],
    minor: List[Tuple[str, str]],
    suspend_chains: Dict[str, List[List[str]]]
) -> None:
    """
    The implmentation of the BuildKempeChains subroutine from the thesis.

    Args:
        G (dict): Adjacency list of the graph.
        chains_to_build (list): List of (start, end) vertex pairs for Kempe chains.
        current (str): Current vertex in the chain.
        end (str): Target vertex to reach.
        visited (set): Visited vertices in current chain.
        available_vertices (list): Candidate vertices to extend the chain.
        minor (list): Adjacency list of the minor graph.
        suspend_chains (dict): Root vertices for minor embedding.

    Prints a counterexample if minor embedding test fails.
    """
    if current == end:
        if not chains_to_build:
            if not test_minor(
                minor, get_edges_from_adjacency_list(G), suspend_chains
            ):      
                save_counterexample(G, minor)
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
def generate_colorings(
    H: Dict[str, Set[str]],
    n: int,
    num_vertices_of_minor: int
) -> List[Dict[str, Set[str]]]:
    """
    Implmentation of the GenerateColorings subroutine from the thesis.

    Args:
        H (dict): Adjacency list of the minor graph with vertex suffixes.
        n (int): Total number of vertices of each coloring.
        num_vertices_of_minor (int): Number of vertices in the minor.

    Returns:
        list: List of adjacency lists of the colorings.
    """
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
        for value, group in groupby(comb):
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


def main(
    n: int,
    num_vertices_of_minor: int,
    minor_graphs: List[nx.Graph]
) -> None:
    """
    Run the Kempe chain algorithm on a list of minor graphs with colorings.

    Args:
        n (int): Total number of colors.
        num_vertices_of_minor (int): Number of vertices in the minor.
        minor_graphs (list): List of networkx graphs representing non-isomorphic minors.

    Returns:
        None
    """
    
    index = 0

    for minor in minor_graphs:
        # builidng the graph H in the format suitable for the minorminer library
        H = { f"{node}a": set([f"{neighbor}a" for neighbor in neighbors]) for node, neighbors in minor.adj.items()}
        suspend_chains = {f"{i}a": [[f"{i}a"]] for i in minor.nodes} ## used for the minorminer library to specify the root vertices to a minor on.
        colorings = generate_colorings(H, n, num_vertices_of_minor)
        edges = list({
            (min(node, neighbor), max(node, neighbor))
            for node, neighbors in H.items()
            for neighbor in neighbors
        })

        coloring_index = 0
        
        # # The loop below described as Algorithm 4 in the thesis.
        for coloring in colorings:
            if (coloring_index + 1) % 20 == 0 or coloring_index == 0:
                logging.info("Minor %d/%d | Coloring %d", index + 1, len(minor_graphs), coloring_index + 1)
            color_to_look_for = color(edges[0][1])
            available_vertices = [
                v for v in coloring.keys() if color(v) == color_to_look_for
            ]
            build_kempe_chains(
                coloring,
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
        logging.info("Finished minor %d of %d", index, len(minor_graphs))
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

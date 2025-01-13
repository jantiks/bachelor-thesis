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
# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(1500)


def find_rooted_minor(G, H):
    target_size = H.number_of_nodes()  # Target number of nodes in the contracted graph
    target_edges = H.number_of_edges()  # Target number of edges in H
    min_degree = min(H.degree(n) for n in H.nodes)  # Minimum degree of any node in H
    target_vertices = set(H.nodes())

    # Recursive helper function for contractions
    def contract_until_size(G_temp, bags):
        # Base case: If sizes match, return the bags
        if G_temp.number_of_nodes() < target_size:
            return None
        # Early stopping: If the current graph has fewer edges than H, stop recursion
        if G_temp.number_of_edges() < target_edges:
            return None

        for node in G_temp.nodes:
            if G_temp.degree(node) < min_degree:
                return None    
    
        if G_temp.number_of_nodes() == target_size and G_temp.number_of_edges() >= H.number_of_edges() and nx.isomorphism.GraphMatcher(G_temp, H).subgraph_is_isomorphic():
            return G_temp, bags

        for u, v in G_temp.edges():
            # Copy the graph to avoid modifying the original
            # Contract the edge
            if u in target_vertices and v in target_vertices:
                continue  # Skip contraction if both vertices are in target_vertices
            G_contracted = nx.contracted_edge(G_temp, (u, v), self_loops=False)
            new_bags = {node: bag.copy() for node, bag in bags.items()}  # Deep copy of bags

            # Update bags: Merge u and v into a single bag associated with u
            if u not in new_bags:
                new_bags[u] = {u}
            if v in new_bags:
                new_bags[u].update(new_bags[v])
                del new_bags[v]  # Remove the bag entry for v if it exists
            else:
                new_bags[u].add(v)

            # Recursive call with the contracted graph and updated bags
            result = contract_until_size(G_contracted, new_bags)
            if result:
                return result  # Return if a solution is found

        return None  # Return None if no valid contraction sequence is found

    # Initialize bags with each vertex in its own bag
    initial_bags = {node: {node} for node in G.nodes()}

    # Start the contraction process
    result = contract_until_size(G, initial_bags)
    if result:
        return result  # Return the bags if a valid contraction sequence is found
    else:
        return None  # No valid contraction sequence found
                

    # Find an embedding of H into G
    # for each vertex from H pick the correspondong vertex from G and contract an edge from that vertex
    # update the new set of edges and vertices, update the bags, check if the size of new graph equals to H, then check if an embeding exists with minorminer 
    # if not, then recursively call the function with the new graph and the same H
    
    for v in G.nodes:
        G.nodes[v]['bag'] = set([v])

# Example usage
# if __name__ == "__main__":
#     # Create graph G
#     traingle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
#     square = [('0b', '1b'), ('1b', '2b'), ('2b', '3b'), ('3b', '4b'), ('4b', '5b'), ('5b', '0b'), ('1b', '4b'),
#                ('0a', '1b'), ('0a', '5b'), ('1a', '2b'), ('1a', '0b'), ('1a', '4b'), ('2a', '1b'), ('2a', '3b'), ('3a', '2b'), ('3a', '4b'), ('4a', '3b'), ('4a', '5b'), ('4a', '1b'), ('5a', '0b'), ('5a', '4b')]
#     c4 = nx.Graph([('0a', '1a')])
#     zc4 = nx.Graph([('0b', '1b'),
#                     ('0a', '1b'), ('1a', '0b')])

#     # traingle = nx.cycle_graph(4)
#     # square = nx.cycle_graph(6)
#     # # Find an assignment of sets of square variables to the triangle variables
#     while True:
#         embedding = find_embedding(
#             c4, 
#             zc4, 
#             random_seed=random.randint(0, 2**32 - 1),
#         )    
#         print(embedding)
#         if '0a' in embedding['0a'] and '1a' in embedding['1a']:
#             break 
#     print(embedding)  # 3, one set for each variable in the triangle
#     # print(G.nodes.data())

#     G = nx.Graph()

#     # Add edges based on the embedding dictionary
#     # for node, neighbors in embedding.items():
#     #     for neighbor in square:
#     #         G.add_edge(node, neighbor)

#     # plt.figure(figsize=(6, 6))
#     # nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=15, font_color='black')
#     # print(G)
#     # plt.title("Graph G: Cycle of 10 Nodes")
#     # plt.show()

# iimport networkx as nx
# import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

def custom_circular_layout(G):
    # nodes_a = sorted([node for node in G.nodes if node.endswith('a')])
    # nodes_b = sorted([node for node in G.nodes if node.endswith('b')])
    nodes_a = sorted([node for node in G.nodes if node.endswith('a')])
    nodes_b = sorted([node for node in G.nodes if node.endswith('b')])
    nodes_c = sorted([node for node in G.nodes if node.endswith('d')])
    nodes_d = sorted([node for node in G.nodes if node.endswith('c')])

    n = len(nodes_a)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {}
    radius_a = 1
    radius_b = 2
    radius_c = 4
    radius_d = 6
    for i, node in enumerate(nodes_a):
        pos[node] = (radius_a * np.cos(angles[i]), radius_a * np.sin(angles[i]))
    for i, node in enumerate(nodes_b):
        pos[node] = (radius_b * np.cos(angles[i]), radius_b * np.sin(angles[i]))
    for i, node in enumerate(nodes_c):
        pos[node] = (radius_b * np.cos(angles[i]), radius_c * np.sin(angles[i]))
    for i, node in enumerate(nodes_d):
        pos[node] = (radius_b * np.cos(angles[i]), radius_d * np.sin(angles[i]))
    return pos

def draw_graph(G, pos, bags, title="Graph"):
    plt.clf()
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
    labels = {node: f"{node}\n{sorted(bags[node])}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title(title)
    # plt.pause(0.5)

def attempt_contraction(G, bags, pos, target_graph, contracted_size=6):
    # Check if the contracted graph has the target number of nodes
    if len(G.nodes) == contracted_size:
        # Check if all edges in target_graph exist in G
        if set(target_graph.edges()).issubset(set(G.edges())):
            print("All target_graph edges exist in G!")
            print("Final bags:", bags)
            return True
        else:
            return False

    for u, v in list(G.edges):
        if not (u.endswith("a") and v.endswith("a")):
            G_copy = G.copy()
            bags_copy = {node: bags[node].copy() for node in bags}

            primary, secondary = (u, v) if u.endswith("a") else (v, u)
            G_copy = nx.contracted_edge(G_copy, (primary, secondary), self_loops=False)
            bags_copy[primary].update(bags_copy[secondary])
            del bags_copy[secondary]

            pos_copy = {}
            # draw_graph(G_copy, pos_copy, bags_copy, title=f"Graph After Contracting Edge ({u}, {v})")
            # print(f"Contracted edge ({u}, {v})", len(G_copy.nodes), len(G_copy.edges))
            if attempt_contraction(G_copy, bags_copy, pos_copy, target_graph, contracted_size):
                return True
    return False

def interactive_contraction(G, target_graph,  bags, pos):
    while True:
        draw_graph(G, pos, bags, title="Interactive Mode - Select Edge")
        edge_input = input("Enter an edge to contract (format: u,v) or 'exit' to stop: ")

        if edge_input.lower() == "exit":
            print("Exiting interactive mode.")
            break
        elif edge_input == "check":
            if nx.is_isomorphic(G, target_graph):
                print("Isomorphic graph found!")
                print("Final bags:", bags)
            elif nx.isomorphism.GraphMatcher(target_graph, G).subgraph_is_isomorphic():
                print("Isomorphic graph found! from subgraph")
                print("Final bags:", bags)
            else: 
                print("No isomorphic graph found.")
            continue
        elif edge_input.startswith("d,"):
            # Delete edge functionality
            try:
                _, u, v = edge_input.split(",")
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    draw_graph(G, pos, bags, title=f"Graph After Deleting Edge ({u}, {v})")
                    print(f"Edge ({u}, {v}) deleted.")
                else:
                    print(f"Edge ({u}, {v}) does not exist in the graph.")
            except ValueError:
                print("Invalid delete format. Use 'd,u,v' to delete an edge.")
            continue
        try:
            u, v = map(str, edge_input.split(","))
            if not G.has_edge(u, v):
                print(f"Edge ({u}, {v}) does not exist in the graph. Try again.")
                continue
        except ValueError:
            print("Invalid input format. Please enter in the format 'u,v'.")
            continue

        # if (u.endswith("a") and v.endswith("b")) or (u.endswith("b") and v.endswith("b")):
        G = nx.contracted_edge(G, (u, v), self_loops=False)
        bags[u].update(bags[v])
        del bags[v]
        pos = {node: pos[u] if node == u else pos.get(node, (0, 0)) for node in G.nodes()}
        draw_graph(G, pos, bags, title=f"Graph After Contracting Edge ({u}, {v})")
        # else:
        #     print("Only (a,b) and (b,b) edges are allowed for contraction.")

def automated_graph_contraction(G, target_graph, contracted_size):
    pos = custom_circular_layout(G)
    bags = {node: set([node]) for node in G.nodes()}
    # draw_graph(G, pos, bags, title="Initial Graph with Empty Bags")

    if attempt_contraction(G, bags, pos, target_graph, contracted_size):
        print("Contraction successful.")
    else:
        print("No isomorphic contraction found.")
        input("Press any key to continue")
    # plt.ioff()
    # plt.show()

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

def generate_all_target_graphs():
    vertices = ['0a', '1a', '2a', '3a', '4a']
    
    all_possible_edges = list(itertools.combinations(vertices, 2))
    
    all_graphs = []
    
    for edges in itertools.chain.from_iterable(itertools.combinations(all_possible_edges, r) for r in range(len(all_possible_edges) + 1)):
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        all_graphs.append(G)
    
    return all_graphs

def main(mode="automated"):
    G = nx.Graph()
    # G.add_edges_from([('0b', '1b'), ('1b', '2b'), ('2b', '3b'), ('3b', '4b'), ('4b', '5b'), ('5b', '0b'), ('1b', '7b'), ('7b', '4b'),
    #                   ('0a', '1b'), ('0a', '5b'), ('1a', '2b'), ('1a', '0b'), ('1a', '7b'), ('2a', '1b'), ('2a', '3b'),
    #                   ('3a', '2b'), ('3a', '4b'), ('4a', '3b'), ('4a', '5b'), ('4a', '7b'), ('5a', '0b'), ('5a', '4b'), ('7a', '1b'), ('7a', '4b')])
    # target_graph = nx.Graph([('0a', '1a'), ('1a', '2a'), ('2a', '3a'), ('3a', '4a'), ('4a', '5a'), ('5a', '0a'), ('1a', '7a'), ('7a', '4a')])

    # G.add_edges_from([('0b', '1b'), ('1b', '2b'), ('2b', '3b'), ('3b', '0b'),
    #                   ('0a', '1b'), ('0a', '3b'), ('1a', '2b'), ('1a', '0b'), ('2a', '1b'), ('2a', '3b'), ('3a', '2b'), ('3a', '0b')])
    

    plt.ion()
    target_graph = nx.Graph([
            ('3a', '4a'),
            ('0a', '5a'),
            ('2a', '3a'),
            ('4a', '5a'),
            ('0a', '4a'),
            ('1a', '2a'),
            ('2a', '4a'),
            ('0a', '2a'),
            ('0a', '1a')
        ])
    G = nx.Graph([
            ('0a', '1b'),
            ('0a', '2b'),
            ('0a', '5b'),
            ('0a', '4b'),
            ('1a', '2b'),
            ('1a', '0b'),
            ('2a', '1b'),
            ('2a', '0b'),
            ('2a', '4b'),
            ('2a', '3b'),
            ('3a', '2b'),
            ('3a', '4b'),
            ('4a', '5b'),
            ('4a', '2b'),
            ('4a', '0b'),
            ('4a', '3b'),
            ('5a', '0b'),
            ('5a', '4b'),
            ('0b', '1a'),
            ('0b', '1b'),
            ('0b', '4a'),
            ('0b', '2d'),
            ('0b', '4b'),
            ('0b', '5a'),
            ('0b', '2b'),
            ('0b', '2a'),
            ('0b', '5b'),
            ('1b', '2d'),
            ('1b', '0b'),
            ('1b', '2b'),
            ('1b', '2a'),
            ('1b', '0a'),
            ('2b', '1a'),
            ('2b', '1b'),
            ('2b', '4a'),
            ('2b', '4b'),
            ('2b', '3a'),
            ('2b', '3b'),
            ('2b', '0b'),
            ('2b', '0a'),
            ('3b', '4a'),
            ('3b', '2d'),
            ('3b', '4b'),
            ('3b', '2b'),
            ('3b', '2a'),
            ('4b', '2d'),
            ('4b', '5b'),
            ('4b', '3a'),
            ('4b', '3b'),
            ('4b', '0b'),
            ('4b', '5a'),
            ('4b', '2b'),
            ('4b', '2a'),
            ('4b', '0a'),
            ('5b', '0b'),
            ('5b', '4a'),
            ('5b', '0a'),
            ('5b', '4b'),
            ('2d', '1b'),
            ('2d', '0b'),
            ('2d', '4b'),
            ('2d', '3b')
        ])

    if mode == "interactive":
        # target_graph = nx.Graph([('0a', '1a'), ('1a', '2a'), ('2a', '3a'), ('3a', '4a'), ('4a', '5a'), ('5a', '0a'), ('1a', '4a')])
        # G = nx.Graph([('a1', 'b3'), ('b3', 'a3'), ('a3', 'b1'), # a1-b1 kempe chain
        #    ('f1', 'a2'), ('a2', 'f2'), ('f2', 'a1'), # a1-f1 kempe chain
        #    ('f1', 'b2'), ('b2', 'f2'), ('f2', 'b3'), # b1-f1 kempe chain
        #    ('f1', 'e2'), ('e2', 'f2'), ('f2', 'e1'), # e1-f1 kempe chain
        # #    ('f3', 'd1'), ('f3', 'd2'), ('d2', 'f1'), # d1-f1 kempe chain
        #    ('e1', 'd2'), ('d2', 'e2'), ('e2', 'd1'), # d1-e1 kempe chain
        #    ('c1', 'd2'), ('d2', 'c2'), ('c2', 'd1'), # c1-d1 kempe chain
        #    ('c1', 'b3'), ('b3', 'c2'), ('c2', 'b1'), # c1-b1 kempe chain
        #    ('d1', 'a2'), ('a2', 'd3'), ('d3', 'a1'), # a1-d1 kempe chain
        # #    ('e1', 'b3'), ('b3', 'e2'), ('e2', 'b1'), # b1-e1 kempe chain
        # #    ('e1', 'a2'), ('a2', 'e2'), ('e2', 'a1'), # a1-e1 kempe chain
        # #    ('e1', 'c3'), ('c3', 'e2'), ('e2', 'c1'), # c1-e1 kempe chain
        # #    ('d1', 'b2'), ('b2', 'd2'), ('d2', 'b1'), # b1-d1 kempe chain
        #    ])
        pos = custom_circular_layout(G)
        bags = {node: set([node]) for node in G.nodes()}
        interactive_contraction(G, target_graph, bags, pos)
    else:
        target_graph = nx.Graph([
            ('3a', '4a'),
            ('0a', '5a'),
            ('2a', '3a'),
            ('4a', '5a'),
            ('0a', '4a'),
            ('1a', '2a'),
            ('2a', '4a'),
            ('0a', '2a'),
            ('0a', '1a')
        ])
        G = nx.Graph([
            ('0a', '1b'),
            ('0a', '2b'),
            ('0a', '5b'),
            ('0a', '4b'),
            ('1a', '2b'),
            ('1a', '0b'),
            ('2a', '1b'),
            ('2a', '0b'),
            ('2a', '4b'),
            ('2a', '3b'),
            ('3a', '2b'),
            ('3a', '4b'),
            ('4a', '5b'),
            ('4a', '2b'),
            ('4a', '0b'),
            ('4a', '3b'),
            ('5a', '0b'),
            ('5a', '4b'),
            ('0b', '1a'),
            ('0b', '1b'),
            ('0b', '4a'),
            ('0b', '2d'),
            ('0b', '4b'),
            ('0b', '5a'),
            ('0b', '2b'),
            ('0b', '2a'),
            ('0b', '5b'),
            ('1b', '2d'),
            ('1b', '0b'),
            ('1b', '2b'),
            ('1b', '2a'),
            ('1b', '0a'),
            ('2b', '1a'),
            ('2b', '1b'),
            ('2b', '4a'),
            ('2b', '4b'),
            ('2b', '3a'),
            ('2b', '3b'),
            ('2b', '0b'),
            ('2b', '0a'),
            ('3b', '4a'),
            ('3b', '2d'),
            ('3b', '4b'),
            ('3b', '2b'),
            ('3b', '2a'),
            ('4b', '2d'),
            ('4b', '5b'),
            ('4b', '3a'),
            ('4b', '3b'),
            ('4b', '0b'),
            ('4b', '5a'),
            ('4b', '2b'),
            ('4b', '2a'),
            ('4b', '0a'),
            ('5b', '0b'),
            ('5b', '4a'),
            ('5b', '0a'),
            ('5b', '4b'),
            ('2d', '1b'),
            ('2d', '0b'),
            ('2d', '4b'),
            ('2d', '3b')
        ])
        # print("Index:", index, len(all_target_graphs))
        print("Target graph:", target_graph.edges())
        automated_graph_contraction(G, target_graph, target_graph.number_of_nodes())
        print("Checked:", target_graph.edges())
        # index += 1
            

    plt.ioff()
    plt.close()



if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "automated"
    main(mode)

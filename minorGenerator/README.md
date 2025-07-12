# Enumeration code

## Prerequisites

### Python Libraries

Install the required Python libraries using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

### External Tools

This project uses **nauty's geng** to generate all connected graphs of a given size.

Install geng:
- On Debian/Ubuntu:

```
sudo apt install nauty
```

- Or build from source: https://pallini.di.uniroma1.it/

Make sure the `geng` command is available in your PATH.

---

## Usage

You can run various test cases by uncommenting the relevant sections in the `__main__` block of the Python script. If any counter-examples are found they will be logged in `counterexamples.txt` file.

### Example 1: Kriesell & Mohr Counterexample for K₇

Search for a counterexample to the minor containment conjecture for a specific graph with 7 vertices:

```
python your_script.py
```

Modify the script's last section to:

```python
minors = [
    nx.Graph(
        [
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("4", "5"),
            ("5", "0"),
            ("0", "6"),
            ("6", "3"),
        ]
    )
]
minor_size = 7
parent_size = 14
main(parent_size, minor_size, minors)
```

Expected runtime: ~10-20 minutes.

---

### Example 2: All Spanning Subgraphs of K₆ with bigger graphs of 13 Vertices

Uncomment the following block:

```python
minor_size = 6
minors = get_graphs_from_geng(minor_size)
parent_size = 13
main(parent_size, minor_size, minors)
```

Expected runtime: ~10-12 hours.

---

### Example 3: All Spanning Subgraphs of K₇ with bigger graphs of 14 Vertices

Uncomment the following block:

```python
minor_size = 7
minors = get_graphs_from_geng(minor_size)
parent_size = 14
main(parent_size, minor_size, minors)
```

Expected runtime: ~20-30 hours or longer.

---

### Example 4: Our Discovered Counterexample for K₇

```python
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
main(parent_size, minor_size, minors)
```

Expected runtime: ~10-20 minutes.

---

## Visualization

Use the `show_graph()` function to display any graph in a matplotlib window.

---
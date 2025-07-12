# Enumeration code

## Prerequisites

### Python Version
This project requires Python 3.9 or higher.

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
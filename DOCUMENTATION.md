# Wikipedia Ego Network Analysis - Complete Documentation

## Overview

This document provides comprehensive documentation for the `wiki_solution.py` script, which solves the Wikipedia Ego Network Analysis assignment. The solution analyzes Wikipedia pages by constructing ego networks around specific topics and computing various network metrics to understand relationships between concepts.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dependencies](#dependencies)
3. [Core Functions](#core-functions)
   - [get_ego_network](#1-get_ego_network)
   - [get_network](#2-get_network)
   - [analyze_ego](#3-analyze_ego)
   - [neighborhood_overlap](#4-neighborhood_overlap)
4. [Network Metrics Explained](#network-metrics-explained)
5. [Analysis Results](#analysis-results)
6. [Usage Instructions](#usage-instructions)
7. [File Outputs](#file-outputs)

---

## Problem Statement

The assignment requires analyzing Wikipedia pages by:

1. **Building ego networks** - Creating directed graphs where nodes are Wikipedia pages and edges represent hyperlinks between them
2. **Computing network metrics** - Calculating various centrality and structural measures
3. **Comparing algorithms** - Analyzing four community detection algorithms (Girvan-Newman, Louvain, Leiden, Clique Percolation)
4. **Computing neighborhood overlap** - Measuring similarity between nodes based on shared neighbors

---

## Dependencies

```python
import pywikibot as pw      # Wikipedia API interface
import networkx as nx        # Graph manipulation
import numpy as np           # Numerical computations
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt  # Visualization
```

---

## Core Functions

### 1. `get_ego_network`

```python
def get_ego_network(ego, lang='en') -> nx.DiGraph
```

**Purpose:** Constructs a directed ego network around a Wikipedia page.

**Parameters:**
- `ego` (str): Title of the Wikipedia page to build the network around
- `lang` (str): Wikipedia language code (default: 'en' for English)

**Returns:** A `nx.DiGraph` containing the ego network

**Algorithm:**

1. **Initialize**: Connect to Wikipedia using `pywikibot`
2. **Collect neighbors**:
   - Get **outgoing links**: Pages that the ego page links to
   - Get **incoming links**: Pages that link to the ego page (limited to 500 to avoid huge downloads)
3. **Filter invalid pages**: Remove pages containing:
   - `(identifier)` or `(Identifier)` in the title
   - `:` in the title (template pages, categories, etc.)
   - Self-loops
4. **Build network**:
   - Add all valid neighbors as nodes
   - Add edges from ego to its outgoing links
   - Add edges from incoming links to ego
   - For each neighbor, check if it links to other neighbors and add those edges

**Example:**
```python
G = get_ego_network('Bembidion ambiguum')
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
# Output: Nodes: 21, Edges: 106
```

---

### 2. `get_network`

```python
def get_network(name, lang='en', download=False) -> nx.DiGraph
```

**Purpose:** Manages network caching - either loads from file or downloads from Wikipedia.

**Parameters:**
- `name` (str): Title of the Wikipedia page
- `lang` (str): Wikipedia language code
- `download` (bool): If `True`, forces download even if cache exists

**Returns:** A `nx.DiGraph` with the ego network

**Logic:**
```
if download == False AND file exists:
    → Read network from CSV file
else:
    → Download network using get_ego_network()
    → Save to CSV file
    → Return network
```

**File Format:** Edge list CSV with format `source,target` per line

**Example:**
```python
# First call downloads and caches
G = get_network('Louvain method', download=True)

# Subsequent calls read from cache
G = get_network('Louvain method', download=False)  # Fast!
```

---

### 3. `analyze_ego`

```python
def analyze_ego(G, ego) -> dict
```

**Purpose:** Computes 17 network metrics for an ego network.

**Parameters:**
- `G` (nx.DiGraph): The ego network graph
- `ego` (str): The ego node identifier

**Returns:** Dictionary with all computed metrics

**Metrics Computed:**

| Metric | Description |
|--------|-------------|
| `out_degree` | Number of outgoing edges from ego |
| `in_degree` | Number of incoming edges to ego |
| `node_num` | Total number of nodes in the network |
| `edge_num` | Edges between neighbors (excluding ego edges) |
| `density` | Ratio of actual edges to maximum possible edges |
| `density_without_ego` | Density of network with ego removed |
| `betweenness` | Normalized betweenness centrality of ego |
| `clustering_coefficient` | Directed clustering coefficient of ego |
| `reciprocity` | Ratio of bidirectional links to all linked pairs |
| `reciprocity_index` | Proportion of symmetric dyads |
| `gl_reciprocity` | Garlaschelli-Loffredo reciprocity measure |
| `avg_distance` | Mean shortest path length among neighbors |
| `diameter` | Longest shortest path in the network |
| `weak_component_num` | Weak components in network without ego |
| `strong_component_num` | Strong components in network without ego |
| `brokerage` | Pairs connected only through ego |
| `avg_brokerage` | Normalized brokerage (brokerage / num_pairs) |

---

### 4. `neighborhood_overlap`

```python
def neighborhood_overlap(G, u, v, neighbor_type='out') -> float
```

**Purpose:** Computes the Jaccard similarity of neighborhoods for two nodes in a directed graph.

**Parameters:**
- `G` (nx.DiGraph): The directed graph
- `u`, `v`: The two nodes to compare
- `neighbor_type` (str): One of `'in'`, `'out'`, or `'both'`

**Returns:** Float between 0 and 1 (0 = no overlap, 1 = identical neighborhoods)

**Formulas:**

**In-Neighbor Overlap:**
$$NO_{in}(u,v) = \frac{|N_{in}(u) \cap N_{in}(v)|}{|N_{in}(u) \cup N_{in}(v)|}$$

**Out-Neighbor Overlap:**
$$NO_{out}(u,v) = \frac{|N_{out}(u) \cap N_{out}(v)|}{|N_{out}(u) \cup N_{out}(v)|}$$

**Bidirectional Overlap:**
$$NO_{both}(u,v) = \frac{|N_{both}(u) \cap N_{both}(v)|}{|N_{both}(u) \cup N_{both}(v)|}$$

Where:
- $N_{in}(u)$ = set of nodes with edges pointing TO u
- $N_{out}(u)$ = set of nodes that u points TO
- $N_{both}(u)$ = $N_{in}(u) \cup N_{out}(u)$

---

## Network Metrics Explained

### Density

The **density** of a directed graph measures how "complete" it is:

$$\text{density} = \frac{m}{n(n-1)}$$

Where $m$ is the number of edges and $n$ is the number of nodes. A complete directed graph has density 1.

### Betweenness Centrality

**Betweenness centrality** measures how often a node appears on shortest paths between other nodes:

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and $\sigma_{st}(v)$ is the number of those paths passing through $v$.

### Garlaschelli-Loffredo Reciprocity

This advanced reciprocity measure accounts for the expected reciprocity in a random graph:

$$\rho = \frac{\sum_{i} \sum_{j \neq i} (a_{ij} - \bar{a})(a_{ji} - \bar{a})}{\sum_{i} \sum_{j \neq i} (a_{ij} - \bar{a})^2}$$

Where $\bar{a}$ is the graph density. Values:
- $\rho > 0$: More reciprocity than expected
- $\rho < 0$: Less reciprocity than expected
- $\rho = 0$: Random reciprocity

### Brokerage

**Brokerage** counts pairs of nodes that can only reach each other through the ego node. High brokerage means the ego is a critical bridge in the network.

---

## Analysis Results

### Comparison of Community Detection Algorithms

| Algorithm | Nodes | Edges | Density | Betweenness | Clustering |
|-----------|-------|-------|---------|-------------|------------|
| Girvan-Newman | 16 | 40 | 0.238 | 0.291 | 0.235 |
| Louvain | 217 | 14,188 | 0.309 | 0.404 | 0.415 |
| Leiden | 96 | 4,213 | 0.473 | 0.010 | 0.477 |
| Clique Percolation | 21 | 65 | 0.205 | 0.223 | 0.196 |

### Key Findings

1. **Louvain method** has the largest ego network (217 nodes), reflecting its widespread popularity and numerous references in the scientific community.

2. **Leiden algorithm** has the highest density (0.473), indicating its neighbors are highly interconnected. This makes sense as Leiden is a focused improvement on Louvain, so its references are tightly related.

3. **Louvain and Leiden** show the strongest relationship based on neighborhood overlap, which aligns with the fact that Leiden was developed as a direct improvement to Louvain.

4. **Girvan-Newman** has higher brokerage (0.186) compared to Louvain/Leiden, suggesting it bridges more disparate concepts due to its pioneering role in the field.

---

## Usage Instructions

### Running the Script

```bash
# Using Python 3.12
python wiki_solution.py

# Or with explicit pyenv
~/.pyenv/versions/3.12.12/bin/python wiki_solution.py
```

### Running Individual Components

```python
from wiki_solution import get_ego_network, get_network, analyze_ego, neighborhood_overlap

# Build a new ego network
G = get_network('Machine learning', download=True)

# Analyze it
results = analyze_ego(G, 'Machine learning')
print(results)

# Compare two nodes
overlap = neighborhood_overlap(G, 'Node1', 'Node2', neighbor_type='out')
```

### Expected Runtime

| Operation | Approximate Time |
|-----------|------------------|
| Download small network (<50 nodes) | 10-30 seconds |
| Download large network (>200 nodes) | 2-5 minutes |
| Analyze small network | <1 second |
| Analyze large network | 10-60 seconds |

---

## File Outputs

### CSV Edge Lists

The script generates edge list files for each analyzed network:

| File | Description |
|------|-------------|
| `Bembidion ambiguum.csv` | Test network (beetle species) |
| `Girvan–Newman algorithm.csv` | Community detection algorithm |
| `Louvain method.csv` | Popular community detection method |
| `Leiden algorithm.csv` | Improved version of Louvain |
| `Clique percolation method.csv` | Overlapping community detection |

**Format:** Each line contains `source,target` representing a directed edge.

### Visualization

`neighborhood_overlap_heatmaps.png` - Three heatmaps showing:
1. In-neighbor overlap matrix
2. Out-neighbor overlap matrix
3. Bidirectional overlap matrix

---

## Test Verification

The solution passes all notebook assertions:

```python
# Sample network e1 tests
assert a['out_degree'] == 5
assert a['in_degree'] == 2
assert a['node_num'] == 6
assert a['edge_num'] == 5
assert a['density'] == 0.4
assert a['density_without_ego'] == 0.25
assert isclose(a['clustering_coefficient'], 0.2368421052631)
assert isclose(a['reciprocity'], 0.2)
assert isclose(a['reciprocity_index'], 0.4666666666666667)
assert isclose(a['gl_reciprocity'], -0.1111111111111111)
assert isclose(a['betweenness'], 0.425)
assert isclose(a['avg_distance'], 2.1)
assert a['diameter'] == 4
assert a['brokerage'] == 0
assert a['avg_brokerage'] == 0

# Neighborhood overlap tests
assert isclose(neighborhood_overlap(G, 1, 2, neighbor_type='in'), 0.5)
assert isclose(neighborhood_overlap(G, 1, 2, neighbor_type='out'), 0.333333)
assert isclose(neighborhood_overlap(G, 1, 2, neighbor_type='both'), 0.5)
```

---

## Written Analysis Answers

### Question 1: Network Comparisons

**Differences:**
- Network sizes vary significantly (16-217 nodes)
- Louvain has the highest betweenness centrality, indicating its central role
- Leiden has uniquely high density but low betweenness
- Clique percolation has the most weak components (6)

**Similarities:**
- All networks belong to the community detection research domain
- All have moderate clustering coefficients (0.19-0.48)
- None are fully connected (all have inf diameter)

**Strongest Relationship:**
Leiden and Louvain methods are most strongly related because Leiden was developed as a direct improvement to Louvain.

### Question 2: Neighborhood Overlap Analysis

1. **Most Related Pair:** Leiden and Louvain (highest out-neighbor overlap)

2. **Most Helpful Neighborhood Type:** OUT-neighbor overlap is most informative because it shows which concepts each algorithm references, reflecting their conceptual similarity.

3. **analyze_ego vs neighborhood_overlap:**
   - Use **neighborhood_overlap** for finding similarities between specific nodes
   - Use **analyze_ego** for understanding a node's structural role in the network

---

## Author Notes

This solution was developed to handle edge cases including:
- Large networks with thousands of edges
- Pages with unusual characters in titles
- Networks with disconnected components
- Nodes with no incoming or outgoing links

The implementation uses efficient NetworkX algorithms where possible and includes proper error handling for Wikipedia API issues.

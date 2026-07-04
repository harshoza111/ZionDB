# Kamradt Semantic Chunking

This document explains the algorithmic flow and mathematical equations behind Greg Kamradt's semantic similarity chunking implementation in ZionDB.

---

## The Algorithmic Goal

Traditional text chunking splits text on character length or token count, resulting in split sentences and loss of context. Semantic chunking identifies topic boundaries dynamically. A boundary is drawn whenever the semantic similarity between neighboring sentence groups drops below a statistical threshold.

---

## Mathematical Formulations

### 1. Group Representation

Given a sequence of sentences $S = [s_0, s_1, \dots, s_{N-1}]$, we smooth the context by grouping each sentence with a buffer window of size $k$:

$$G_i = \text{combine}(s_{j}) \quad \text{for } j \in [\max(0, i-k), \min(N-1, i+k)]$$

### 2. Group Embedding

Each group representation $G_i$ is converted to a normalized vector $E_i$ of size $D = 384$:

$$E_i = \text{Embed}(G_i) \quad \text{where } \|E_i\|_2 = 1.0$$

*(Optionally, if using vector averaging, $E_i$ is calculated by averaging individual sentence vectors and re-normalizing).*

### 3. Cosine Distance Calculation

We compute the cosine distance $d_i$ between consecutive group embeddings $E_i$ and $E_{i+1}$:

$$d_i = 1.0 - \text{cosine\_similarity}(E_i, E_{i+1})$$

Since $E_i$ and $E_{i+1}$ are L2-normalized, the cosine similarity is the dot product:

$$d_i = 1.0 - (E_i \cdot E_{i+1})$$

There are $N-1$ distances in total ($d_0, d_1, \dots, d_{N-2}$).

### 4. Boundary Threshold Selection

The threshold $T$ dictates how aggressively the text is split. We support two methods:

#### A. Percentile Thresholding
Given a target percentile $P$ (e.g. 95.0):

$$T = \text{Percentile}(D_{list}, P)$$

#### B. Standard Deviation Thresholding
Given a standard deviation multiplier $M$ (e.g. 1.2):

$$T = \mu + M \cdot \sigma$$

Where $\mu$ is the mean of the distances and $\sigma$ is the standard deviation.

### 5. Split Execution

For each gap $i$ from $0$ to $N-2$, a boundary is declared if:

$$d_i \ge T$$

The sentence indices are grouped between adjacent boundaries to form the final chunks.

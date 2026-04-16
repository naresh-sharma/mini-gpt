#!/usr/bin/env python3
"""
MiniGPT Embedding Visualization

Projects pretrained GloVe vectors down to 2D using PCA and plots them,
colored by category. The clustering you see is the whole point of
embeddings: "paris" lands near "london" and "tokyo", "dog" lands near
"cat" and "rabbit", because training pushed co-occurring words together
in the original 50-dimensional space.

Usage:
    python examples/visualize_embeddings.py

First run downloads ~65MB of GloVe to ~/gensim-data/ (cached).
Requires a display; pops up a matplotlib window.
"""

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np

# Curated word list grouped into categories. Small enough that every
# point is legible on the plot; diverse enough that the clustering is
# obvious.
WORDS_BY_CATEGORY = {
    "Animals": ["dog", "cat", "horse", "cow", "pig", "sheep", "rabbit", "tiger", "lion", "bear"],
    "Countries": [
        "france",
        "germany",
        "italy",
        "spain",
        "japan",
        "china",
        "india",
        "brazil",
        "canada",
        "mexico",
    ],
    "Tech": [
        "computer",
        "software",
        "internet",
        "database",
        "algorithm",
        "network",
        "server",
        "browser",
        "keyboard",
        "monitor",
    ],
    "Food": [
        "bread",
        "cheese",
        "apple",
        "banana",
        "pasta",
        "pizza",
        "chocolate",
        "coffee",
        "salad",
        "soup",
    ],
}

CATEGORY_COLORS = {
    "Animals": "#d62728",  # red
    "Countries": "#1f77b4",  # blue
    "Tech": "#2ca02c",  # green
    "Food": "#ff7f0e",  # orange
}


def load_glove():
    print("Loading pretrained GloVe (~65MB, one-time download cached to ~/gensim-data/)...")
    model = api.load("glove-wiki-gigaword-50")
    print(f"  Loaded {len(model.key_to_index):,} words, {model.vector_size}-dim vectors.")
    print()
    return model


def pca_2d(vectors):
    """
    Project to 2D using PCA. Pure NumPy — no sklearn dependency.

    PCA finds the two directions in the original 50D space along which
    the selected words vary the most, then projects every point onto
    those two directions.
    """
    # Center the data — PCA requires zero-mean input.
    centered = vectors - vectors.mean(axis=0)
    # SVD gives the principal components directly, more stably than
    # forming the covariance matrix by hand.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    # Top 2 right singular vectors are the top 2 principal directions.
    components = vt[:2]
    return centered @ components.T


def collect_vectors(model):
    """Look up every word, skipping any that aren't in the vocabulary."""
    words, vectors, categories = [], [], []
    for category, category_words in WORDS_BY_CATEGORY.items():
        for word in category_words:
            if word not in model:
                print(f"  warning: '{word}' not in vocabulary, skipping")
                continue
            words.append(word)
            vectors.append(model[word])
            categories.append(category)
    return words, np.array(vectors), categories


def plot(words, coords_2d, categories):
    """Scatter plot colored by category, with word labels."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for category, color in CATEGORY_COLORS.items():
        mask = [c == category for c in categories]
        points = coords_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], c=color, label=category, s=80, alpha=0.8)

    # Label every point.
    for word, (x, y) in zip(words, coords_2d, strict=True):
        ax.annotate(word, (x, y), fontsize=9, xytext=(5, 5), textcoords="offset points")

    ax.set_title("GloVe embeddings projected to 2D via PCA", fontsize=14)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    try:
        model = load_glove()

        print("Looking up vectors for curated word list...")
        words, vectors, categories = collect_vectors(model)
        print(f"  {len(words)} words across {len(set(categories))} categories.")
        print()

        print("Projecting 50D → 2D via PCA...")
        coords = pca_2d(vectors)
        print(f"  Done. Variance is spread across {coords.shape[1]} components.")
        print()

        print("Plotting (close the window to exit)...")
        plot(words, coords, categories)
        plt.show()

        print()
        print("What to notice:")
        print("- Words in the same category cluster together.")
        print("- Clusters are separated without anyone telling the model")
        print("  what a 'country' or 'animal' is — training figured it out")
        print("  from co-occurrence statistics alone.")
        print("- 2D is a lossy projection of 50D; small overlaps are the")
        print("  map losing detail, not the embeddings being wrong.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've installed all requirements:")
        print("    pip install -r requirements.txt")


if __name__ == "__main__":
    main()

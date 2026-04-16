#!/usr/bin/env python3
"""
MiniGPT Embedding Demo

Demonstrates what trained embeddings look like using pretrained GloVe
vectors. The Part 2 module (TokenEmbedding) implements the lookup
mechanism; GloVe shows what the matrix rows become after training on
a large corpus.

The headline moment:

    king - man + woman ≈ queen

Vector arithmetic on word embeddings produces semantically meaningful
answers — that's the payoff that makes "a dense vector per token"
worth the trouble.

Usage:
    python examples/embedding_demo.py

First run downloads ~65MB of pretrained GloVe vectors to
~/gensim-data/ (cached for all future runs).
"""

import gensim.downloader as api


def load_glove():
    """Load the 50-dimensional GloVe vectors trained on Wikipedia+Gigaword."""
    print("Loading pretrained GloVe (~65MB, one-time download cached to ~/gensim-data/)...")
    model = api.load("glove-wiki-gigaword-50")
    print(f"  Loaded {len(model.key_to_index):,} words, {model.vector_size}-dim vectors.")
    print()
    return model


def demo_lookup(model):
    """Show what a single embedding actually is: a vector of numbers."""
    print("=" * 60)
    print("WHAT IS AN EMBEDDING?")
    print("=" * 60)
    print("An embedding is a row in a big matrix, one row per word.")
    print(f"Each row has {model.vector_size} numbers.")
    print()

    for word in ["king", "queen", "computer"]:
        vec = model[word]
        # Show the first 8 numbers for readability.
        preview = ", ".join(f"{x:+.3f}" for x in vec[:8])
        print(f"  {word:>10}: [{preview}, ...]  (length {len(vec)})")
    print()
    print("These numbers are meaningless on their own. The magic is that")
    print("similar words end up with similar vectors — look at similarity:")
    print()


def demo_similarity(model):
    """Nearest-neighbor search — similar words have similar vectors."""
    print("=" * 60)
    print("NEAREST NEIGHBORS")
    print("=" * 60)
    print("Training pushes semantically similar words toward each other")
    print("in vector space. Cosine similarity picks them out:")
    print()

    for word in ["king", "computer", "paris", "happy"]:
        neighbors = model.most_similar(word, topn=5)
        neighbor_list = ", ".join(f"{w} ({s:.2f})" for w, s in neighbors)
        print(f"  near '{word}':")
        print(f"    {neighbor_list}")
        print()


def demo_analogies(model):
    """The mic-drop: vector arithmetic encodes real-world relationships."""
    print("=" * 60)
    print("ANALOGIES")
    print("=" * 60)
    print("Subtract one vector, add another — you land near a word that")
    print("completes the analogy. This emerges from training; nobody")
    print("hard-coded it.")
    print()

    analogies = [
        # (positive, negative, description)
        (["king", "woman"], ["man"], "king - man + woman = ?"),
        (["paris", "germany"], ["france"], "paris - france + germany = ?"),
        (["walking", "swim"], ["walk"], "walking - walk + swim = ?"),
        (["better", "bad"], ["good"], "better - good + bad = ?"),
        (["japan", "london"], ["tokyo"], "japan - tokyo + london = ?"),
    ]

    for positive, negative, description in analogies:
        print(f"  {description}")
        results = model.most_similar(positive=positive, negative=negative, topn=3)
        for i, (word, score) in enumerate(results, start=1):
            marker = "→" if i == 1 else " "
            print(f"    {marker} {i}. {word} ({score:.2f})")
        print()


def demo_limits(model):
    """Show where the magic breaks down — embeddings are not AGI."""
    print("=" * 60)
    print("WHERE IT BREAKS")
    print("=" * 60)
    print("GloVe is trained on word co-occurrence. It doesn't know the")
    print("world. Analogies that need actual reasoning often fail:")
    print()

    # Classic "fail" cases: subtle semantic relations beyond co-occurrence.
    tricky = [
        (["chicken", "calf"], ["cow"], "chicken is to ? as calf is to cow (want: egg)"),
        (["fire", "water"], ["ice"], "fire is to ice as ? is to water"),
    ]

    for positive, negative, description in tricky:
        print(f"  {description}")
        results = model.most_similar(positive=positive, negative=negative, topn=3)
        for i, (word, score) in enumerate(results, start=1):
            print(f"      {i}. {word} ({score:.2f})")
        print()

    print("These 'failures' are useful: they show embeddings encode")
    print("distributional statistics, not understanding. Modern LLMs")
    print("layer reasoning on top of this foundation.")
    print()


def main():
    try:
        model = load_glove()
        demo_lookup(model)
        demo_similarity(model)
        demo_analogies(model)
        demo_limits(model)

        print("=" * 60)
        print("Demo complete.")
        print()
        print("Key takeaways:")
        print("- An embedding is just a vector of numbers per word.")
        print("- Training arranges them so similar words cluster.")
        print("- Vector arithmetic captures real-world relationships.")
        print("- The MiniGPT TokenEmbedding class does the same lookup")
        print("  — random at init, meaningful after training.")
        print()
        print("Next steps:")
        print("- Read the Part 2 blog post")
        print("- Try the Part 2 Jupyter notebook")
        print("- Replace the analogy words above with your own")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've installed all requirements:")
        print("    pip install -r requirements.txt")


if __name__ == "__main__":
    main()

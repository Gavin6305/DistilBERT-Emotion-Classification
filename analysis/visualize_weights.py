import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
import umap

# Local imports
from common.labels import actual_labels, sentiment_dict, ekman_mapping, inv_sentiment_dict
from models.flat import load_flat_model


def plot_cosine_similarity(model_path="./flat_cls/checkpoint-5428", plotting=True):
    """ Plots the cosine similarity between weights of 28-dim classifier head of model"""
    # Load model from path
    eval_model, _ = load_flat_model(model_path)

    # classifier head weight matrix shape [28, hidden_dim]
    W = eval_model.classifier.weight.detach().cpu()

    # cosine similarity matrix [28, 28]
    cos_sim = F.cosine_similarity(W.unsqueeze(1), W.unsqueeze(0), dim=-1)

    # Plot
    if plotting:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cos_sim,
                    xticklabels=actual_labels,
                    yticklabels=actual_labels,
                    cmap="viridis",
                    center=0)
        plt.title("Cosine Similarity Between Emotion Classifier Weights", fontsize=16)
        plt.show()
        return None
    else:
        return cos_sim


def similarity_table(model_path="./flat_cls/checkpoint-5428", level='sentiment'):
    """ Displays a table (DataFrame) of similarities between and within groups on the sentiment or Ekman level"""
    # Convert dict to lookup table
    lvl_to_mapping = {'sentiment': sentiment_dict, 'ekman': ekman_mapping}
    lvl_to_ids = {
        group: [actual_labels.index(lbl) for lbl in labels]
        for group, labels in lvl_to_mapping[level].items()
    }

    # Get cosine similarity matrix
    cosine_sim = plot_cosine_similarity(model_path, plotting=False)

    # Compute mean similarity in set of indices
    def mean_similarity(indices):
        """Average cosine similarity among all pairs inside a group."""
        sub = cosine_sim[np.ix_(indices, indices)]
        # exclude diagonal = 1.0
        mask = ~np.eye(len(indices), dtype=bool)
        return sub[mask].mean().item()

    # Similarity in between groups
    def between_similarity(indices_A, indices_B):
        sub = cosine_sim[np.ix_(indices_A, indices_B)]
        return sub.mean().item()

    # Create table
    results = []
    groups = list(lvl_to_ids.keys())

    # Within-group
    for g in groups:
        results.append({
            "group": g,
            "type": "within",
            "similarity": mean_similarity(lvl_to_ids[g])
        })

    # Between-group
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            results.append({
                "group": f"{g1} vs {g2}",
                "type": "between",
                "similarity": between_similarity(lvl_to_ids[g1], lvl_to_ids[g2])
            })

    # Construct a df sorted by similarity and return
    sentiment_df = pd.DataFrame(results)
    return sentiment_df.sort_values("similarity", ascending=False)


def plot_dendrogram(model_path="./flat_cls/checkpoint-5428"):
    # Load model from path and get weights
    eval_model, _ = load_flat_model(model_path)
    weight_matrix = eval_model.classifier.weight.detach().cpu().numpy()

    # Normalize rows
    weight_matrix_norm = normalize(weight_matrix)

    # Compute pairwise cosine distances
    distances = pdist(weight_matrix_norm, metric="cosine")

    # Run hierarchical clustering
    Z = linkage(distances, method="ward")

    # Plot dendrogram
    plt.figure(figsize=(14, 6))
    dendrogram(Z,
               labels=actual_labels,
               leaf_rotation=90,
               leaf_font_size=10,
               color_threshold=0.6 * max(Z[:, 2]))
    plt.title("Hierarchical Clustering of Emotion Classifier Weights")
    plt.tight_layout()
    plt.show()


def plot_umap(model_path="./flat_cls/checkpoint-5428"):
    # Load model and get weights
    eval_model, _ = load_flat_model(model_path)
    weight_matrix = eval_model.classifier.weight.detach().cpu().numpy()

    # Normalize rows
    weight_matrix_norm = normalize(weight_matrix)

    # Map to colors
    def get_sentiment (emotion):
        if emotion in sentiment_dict["positive"]:
            return "positive"
        elif emotion in sentiment_dict["negative"]:
            return "negative"
        return "ambiguous"

    sentiments = [get_sentiment(e) for e in actual_labels]
    color_map = {"positive": "green", "negative": "red", "ambiguous": "blue"}
    colors = [color_map[s] for s in sentiments]

    # UMAP (Dimensionality reduction)
    reducer = umap.UMAP(n_components=2,
                        random_state=42,
                        metric="cosine",
                        min_dist=0.1)
    emb_2d = reducer.fit_transform(weight_matrix_norm)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=80)

    # Add labels
    for i, label in enumerate(actual_labels):
        plt.text(emb_2d[i, 0] + 0.01, emb_2d[i, 1] + 0.01, label, fontsize=9)

    # Legend using proxy artists
    legend_items = [
        mpatches.Patch(color="green", label="Positive"),
        mpatches.Patch(color="red", label="Negative"),
        mpatches.Patch(color="blue", label="Ambiguous")
    ]
    plt.legend(handles=legend_items, loc="best")

    plt.title("2D Emotion Embedding Map (UMAP on Classifier Weights)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

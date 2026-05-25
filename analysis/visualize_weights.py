import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from common.labels import actual_labels
from models.flat import load_flat_model


def plot_cosine_similarity(model_path="./flat_cls/checkpoint-5428"):
    """ Plots the cosine similarity between weights of 28-dim classifier head of model"""
    # Load model from path
    eval_model, _ = load_flat_model(model_path)

    # classifier head weight matrix shape [28, hidden_dim]
    W = eval_model.classifier.weight.detach().cpu()

    # cosine similarity matrix [28, 28]
    cos_sim = F.cosine_similarity(W.unsqueeze(1), W.unsqueeze(0), dim=-1)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cos_sim,
                xticklabels=actual_labels,
                yticklabels=actual_labels,
                cmap="viridis",
                center=0)
    plt.title("Cosine Similarity Between Emotion Classifier Weights", fontsize=16)
    plt.show()


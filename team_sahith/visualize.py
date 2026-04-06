import matplotlib.pyplot as plt
import os


def generate_visualizations(dataset, folder, results, cm, nulls, corr, feature_importance):

    graphs_folder = f"{folder}/graphs"
    os.makedirs(graphs_folder, exist_ok=True)

    models = [r["model"] for r in results]
    scores = [r["score"] for r in results]

    # Score graph
    plt.figure()
    plt.bar(models, scores)
    plt.title(f"Model Score - {dataset}")
    plt.xlabel("Model")
    plt.ylabel("Score (0-100)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{graphs_folder}/{dataset}_score.png")
    plt.close()

    # Confusion matrix
    if cm is not None:
        plt.figure()
        plt.imshow(cm)
        plt.colorbar()
        plt.title(f"Confusion Matrix - {dataset}")
        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.text(j, i, cm[i][j], ha='center')
        plt.tight_layout()
        plt.savefig(f"{graphs_folder}/{dataset}_cm.png")
        plt.close()

    # Missing values
    if nulls is not None:
        plt.figure()
        plt.bar(nulls.index.astype(str), nulls.values)
        plt.xticks(rotation=90)
        plt.title(f"Missing Values - {dataset}")
        plt.tight_layout()
        plt.savefig(f"{graphs_folder}/{dataset}_missing.png")
        plt.close()

    # Correlation matrix
    if corr is not None and not corr.empty:
        plt.figure()
        plt.imshow(corr)
        plt.colorbar()
        plt.title(f"Correlation - {dataset}")
        plt.tight_layout()
        plt.savefig(f"{graphs_folder}/{dataset}_corr.png")
        plt.close()

    # Feature importance
    if feature_importance is not None:
        plt.figure()
        plt.plot(feature_importance)
        plt.title(f"Feature Importance - {dataset}")
        plt.tight_layout()
        plt.savefig(f"{graphs_folder}/{dataset}_features.png")
        plt.close()
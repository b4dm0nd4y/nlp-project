import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score

def graph_metrics(model, model_name, X_valid, y_valid, decoder):
    

    models = {
        f"{model_name}": model,
    }

    # Prepare plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes = axes.ravel()

    # Class labels (human-readable)
    class_labels = [decoder[i] for i in sorted(decoder.keys())]

    for idx, (name, model) in enumerate(models.items()):
        # Predict class labels
        y_test = y_valid
        y_pred = model.predict(X_valid)

        # Compute macro F1-score and per-class F1
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        # Bar chart for per-class F1-scores
        axes[idx].bar(np.arange(len(f1_per_class)), f1_per_class, tick_label=class_labels)
        axes[idx].set_title(f'{name} - F1 Score per Class\n(Macro Avg F1: {f1:.2f})')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('F1 Score')
        axes[idx].set_ylim(0, 1)
        axes[idx].tick_params(axis='x', rotation=90)
        axes[idx].grid(True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx + 1],
                    xticklabels=class_labels, yticklabels=class_labels)
        axes[idx + 1].set_title(f'{name} - Confusion Matrix')
        axes[idx + 1].set_xlabel('Predicted')
        axes[idx + 1].set_ylabel('True')
        axes[idx + 1].tick_params(axis='x', rotation=45)
        axes[idx + 1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()
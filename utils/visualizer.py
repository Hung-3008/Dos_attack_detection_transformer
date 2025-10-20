import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, class_names, model_name):
    """
    Plots and saves a confusion matrix heatmap.

    Args:
        cm (np.ndarray): The confusion matrix.
        class_names (list): The names of the classes.
        model_name (str): The name of the model.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f'{model_name} - Confusion Matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """
    Plots and saves a feature importance bar plot.

    Args:
        model: The trained model (must have feature_importances_ attribute).
        feature_names (list): The names of the features.
        model_name (str): The name of the model.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(5)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'{model_name} - Top 5 Feature Importance')
    plt.savefig(f'{model_name} - Top 5 Feature Importance.png')
    plt.close()

def plot_classification_report(report, model_name):
    """
    Plots and saves a classification report.

    Args:
        report (dict): The classification report from sklearn.
        model_name (str): The name of the model.
    """
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    report_df = report_df.drop(index=['weighted avg', 'macro avg'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='Blues')
    plt.title(f'{model_name} - Classification Report')
    plt.savefig(f'{model_name} - Classification Report.png')
    plt.close()

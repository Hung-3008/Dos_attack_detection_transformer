from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import wandb
import os
from datetime import datetime
import joblib

def train_model(X_train, y_train):
    """
    Trains a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        RandomForestClassifier: The trained model.
    """
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    # Save trained sklearn model to checkpoints
    try:
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(ckpt_dir, f"random_forest_{ts}.joblib")
        joblib.dump(rf_classifier, model_path)
        print(f"Saved RandomForest model to {model_path}")
    except Exception as e:
        print(f"Warning: failed to save RandomForest model: {e}")

    return rf_classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print('Accuracy:', accuracy)
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    print('\nConfusion Matrix:\n', cm)

    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=model.classes_)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }


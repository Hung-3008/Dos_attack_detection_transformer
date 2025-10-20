from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import wandb

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


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import wandb

def train_model(X_train, y_train):
    """
    Trains a Decision Tree classifier.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        DecisionTreeClassifier: The trained model.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.

    Args:
        model (DecisionTreeClassifier): The trained model.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    y_pred = model.predict(X_test)

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


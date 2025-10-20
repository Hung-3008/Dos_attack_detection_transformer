import argparse
from config import config
from data_loader import data_loader
from preprocessor import preprocessor
from models import random_forest, decision_tree
from utils import visualizer
import wandb

def main():
    """
    Main function to run the pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "decision_tree"])
    args = parser.parse_args()

    model_config = {}
    if args.model == "random_forest":
        model_module = random_forest
        model_name = "Random Forest"
        model_config = {
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42
        }
    elif args.model == "decision_tree":
        model_module = decision_tree
        model_name = "Decision Tree"
        model_config = {
            "max_depth": 3,
            "random_state": 42
        }

    # Initialize wandb
    wandb.init(project="cns-dos-prediction", config={
        "model": args.model,
        "features_to_keep": config.FEATURES_TO_KEEP,
        "categorical_features": config.CATEGORICAL_FEATURES,
        **model_config
    })

    # Load data
    df_train, df_test = data_loader.load_data(
        'data/UNSW_NB15_DoS_train_data.csv',
        'data/UNSW_NB15_DoS_test_data.csv'
    )

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocessor.preprocess_data(
        df_train, 
        df_test, 
        config.FEATURES_TO_KEEP, 
        config.CATEGORICAL_FEATURES
    )

    # Train model
    model = model_module.train_model(X_train, y_train)

    # Evaluate model
    metrics = model_module.evaluate_model(model, X_test, y_test)

    try:
        # Log metrics
        wandb.log({
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"]
        })

        # Visualize results
        class_names = ['Normal', 'DoS']
        visualizer.plot_confusion_matrix(metrics['confusion_matrix'], class_names, model_name)
        visualizer.plot_feature_importance(model, X_train.columns, model_name)
        visualizer.plot_classification_report(metrics['classification_report'], model_name)
    except NameError:
        print("Could not log or visualize metrics due to a NameError. Please check the evaluate_model function.")

    wandb.finish()

if __name__ == '__main__':
    main()
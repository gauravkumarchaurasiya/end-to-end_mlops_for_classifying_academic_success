import joblib
import sys
import logging
import pandas as pd
from yaml import safe_load
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from src.logger import logging
from src.models.models_list import models
from src.visualization.plot_metric import update_metrics, plot_metrics

TARGET = 'Target'

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a CSV file."""
    return pd.read_csv(path)

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    """Split a DataFrame into feature matrix X and target vector y."""
    X = dataframe.drop(columns=target_column)
    y = dataframe[target_column]
    return X, y

def train_model(model, X_train, y_train):
    """Train a machine learning model."""
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, precision, recall

def save_model(model, save_path: Path):
    """Save a trained model to a file."""
    joblib.dump(value=model, filename=save_path)

def update_best_models(model_names, f1_scores):
    """Update the best two models based on their F1 scores."""
    model_performances = dict(zip(model_names, f1_scores))
    sorted_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_models) > 0:
        best_model1 = sorted_models[0][0]
    else:
        best_model1 = None
        
    if len(sorted_models) > 1:
        best_model2 = sorted_models[1][0]
    else:
        best_model2 = None

    return best_model1, best_model2

def save_best_models(best_model1, best_model2, file_path):
    """Save the names of the best models to the models_list.py file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.startswith('best_model1'):
                file.write(f'best_model1 = "{best_model1}"\n')
            elif line.startswith('best_model2'):
                file.write(f'best_model2 = "{best_model2}"\n')
            else:
                file.write(line)
                
def update_metrics_file(model_names, accuracys):
    """Update the plot_metic.py file with the given model name and accuracy."""
    metrics_file_path = Path(__file__).parent.parent/"visualization" / 'plot_metic.py'
    
    with open(metrics_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the lines where model_names and accuracies lists are defined and update them
    for i, line in enumerate(lines):
        if line.strip().startswith("X_model_names = ["):
            # Add the new model name
            lines.insert(i + 1, f"    '{model_names}',\n")
        elif line.strip().startswith("Y_model_accuracies = ["):
            # Add the new accuracy
            lines.insert(i + 1, f"    {accuracys:.4f},\n")
    
    with open(metrics_file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Metrics updated in {metrics_file_path}")
    logging.info(f"Metrics updated in {metrics_file_path}")


def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    training_data_path = root_path / 'data' / 'processed' / sys.argv[1]

    train_data = load_dataframe(training_data_path)
    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)
    logging.info("Train Test split successful")

    with open('params.yaml') as f:
        params = safe_load(f)

    logging.info(f"Models to be trained: {models}")

    model_names = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for model_name, model in models.items():
        logging.info(f'{model_name} is training...')
        trained_model = train_model(model=model, X_train=X_train, y_train=y_train)

        accuracy, f1, precision, recall = evaluate_model(model=trained_model, X_test=X_train, y_test=y_train)
        logging.info(f"{model_name} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        
        model_names.append(model_name)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        model_output_path = root_path / 'models' / 'models'
        model_output_path.mkdir(exist_ok=True)
        model_output_path_ = model_output_path / f'{model_name.lower()}.joblib'
        save_model(model=trained_model, save_path=model_output_path_)

    update_metrics(model_names, accuracy_scores)
    
    best_model1, best_model2 = update_best_models(model_names, f1_scores)
    logging.info(f"Best Model 1: {best_model1}")
    logging.info(f"Best Model 2: {best_model2}")

    models_list_path = current_path.parent /'models_list.py'
    save_best_models(best_model1, best_model2, models_list_path)
    logging.info("Step 4: Completed Model Training")

if __name__ == "__main__":
    main()

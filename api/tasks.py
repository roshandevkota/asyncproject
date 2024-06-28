from celery import shared_task
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

logger = logging.getLogger(__name__)


@shared_task
def add(x, y):
    return x + y

@shared_task
def process_file(file_path):
    """
    Processes a CSV file to extract metadata.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary containing the columns and delimiter of the CSV file.

    Raises:
        Exception: If there is an error processing the file.

    Example:
        >>> process_file.delay('/path/to/file.csv')
        {'columns': ['column1', 'column2'], 'delimiter': ','}
    """
    try:
        logger.info(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        delimiter = ','  # Simplified delimiter detection for illustration
        metadata = {
            'columns': df.columns.tolist(),
            'delimiter': delimiter,
        }
        logger.info(f"Metadata: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

@shared_task
def profile_data(file_path):
    """
    Generates a profiling report for a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        str: The path to the generated profiling report.

    Raises:
        Exception: If there is an error profiling the data.

    Example:
        >>> profile_data.delay('/path/to/file.csv')
        '/path/to/profile_report.html'
    """
    try:
        logger.info(f"Profiling data for file: {file_path}")
        df = pd.read_csv(file_path)
        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile_path = os.path.join(os.path.dirname(file_path), 'profile_report.html')
        profile.to_file(profile_path)
        logger.info(f"Profile report saved to: {profile_path}")
        return profile_path
    except Exception as e:
        logger.error(f"Error profiling data: {e}")
        raise

@shared_task
def train_model(file_path, target_column):
    """
    Trains a RandomForest model on the provided CSV file and target column.

    Args:
        file_path (str): The path to the CSV file.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary containing the model's training and test scores, and the path to the saved model.

    Raises:
        Exception: If there is an error training the model.

    Example:
        >>> train_model.delay('/path/to/file.csv', 'target')
        {'metadata': {'train_score': 0.95, 'test_score': 0.90}, 'model_path': '/path/to/model.pkl'}
    """
    try:
        logger.info(f"Training model with file: {file_path} and target column: {target_column}")
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        metadata = {
            "train_score": model.score(X_train, y_train),
            "test_score": model.score(X_test, y_test)
        }
        
        model_path = os.path.join(os.path.dirname(file_path), 'model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Model trained and saved to: {model_path}")
        return {'metadata': metadata, 'model_path': model_path}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

@shared_task
def predict(file_path, model_path):
    """
    Uses a trained model to make predictions on a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        model_path (str): The path to the trained model file.

    Returns:
        str: The path to the CSV file containing the predictions.

    Raises:
        Exception: If there is an error making predictions.

    Example:
        >>> predict.delay('/path/to/file.csv', '/path/to/model.pkl')
        '/path/to/predictions.csv'
    """
    try:
        logger.info(f"Making predictions with file: {file_path} and model: {model_path}")
        df = pd.read_csv(file_path)
        model = joblib.load(model_path)
        predictions = model.predict(df)
        df['predictions'] = predictions
        result_path = os.path.join(os.path.dirname(file_path), 'predictions.csv')
        df.to_csv(result_path, index=False)
        logger.info(f"Predictions saved to: {result_path}")
        return result_path
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

@shared_task
def add(x, y):
    return x + y

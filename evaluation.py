import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
class Config:
    # Paper-specific constants
    VELOCITY_THRESHOLD = 0.5  # From research paper
    FLOW_THEORY_CONSTANT = 0.75  # Example constant from Flow Theory

    # Project settings
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
    MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")
    LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

# Custom exception classes
class EvaluationError(Exception):
    """Custom exception class for evaluation-related errors."""
    pass

class ModelNotFoundError(EvaluationError):
    """Raised when a trained model is not found for evaluation."""
    pass

# Data structures/models
class AgentEvaluationMetrics:
    """
    Data model to hold evaluation metrics for an agent.

    Attributes:
        precision (float): The precision score.
        recall (float): The recall score.
        f1_score (float): The F1 score.
        accuracy (float): The accuracy of the agent's predictions.
        confusion_matrix (np.ndarray): The confusion matrix.
    """
    def __init__(self, precision: float, recall: float, f1_score: float, accuracy: float, confusion_matrix: np.ndarray):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix

# Helper classes and utilities
class AgentEvaluationDataset(Dataset):
    """
    Custom dataset class for agent evaluation.

    Args:
        data (pd.DataFrame): The evaluation data containing predictions and labels.
        transform (callable, optional): Optional transformation to apply to the data. Defaults to None.

    Attributes:
        data (pd.DataFrame): The evaluation data.
        transform (callable, optional): Optional transformation function.
    """
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.transform:
            return self.transform(self.data.iloc[idx])
        return self.data.iloc[idx]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load evaluation data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing evaluation data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        error_msg = f"Evaluation data file not found at path: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except pd.errors.EmptyDataError:
        error_msg = f"Evaluation data file is empty: {file_path}"
        logger.error(error_msg)
        raise EvaluationError(error_msg)
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing evaluation data file: {file_path}. Error: {str(e)}"
        logger.error(error_msg)
        raise EvaluationError(error_msg)

def calculate_metrics(preds: np.ndarray, labels: np.ndarray) -> AgentEvaluationMetrics:
    """
    Calculate evaluation metrics for an agent.

    Args:
        preds (np.ndarray): The predicted values.
        labels (np.ndarray): The true labels.

    Returns:
        AgentEvaluationMetrics: An object containing the evaluation metrics.

    Raises:
        EvaluationError: Raised if the shapes of predictions and labels do not match.
    """
    if preds.shape != labels.shape:
        error_msg = "Predictions and labels must have the same shape for metric calculation."
        logger.error(error_msg)
        raise EvaluationError(error_msg)

    confusion_matrix = np.confusion_matrix(labels, preds)
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix).astype(np.float64)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1).astype(np.float64)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0).astype(np.float64)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return AgentEvaluationMetrics(
        precision=np.mean(precision),
        recall=np.mean(recall),
        f1_score=np.mean(f1_score),
        accuracy=accuracy,
        confusion_matrix=confusion_matrix
    )

# Main class for evaluation
class AgentEvaluator:
    """
    Main class for evaluating an agent's performance.

    Attributes:
        data_loader (DataLoader): Data loader for evaluation data.
        device (torch.device): Device to use for computation.
        model (nn.Module): The trained agent model.

    Methods:
        load_model: Load a trained agent model.
        evaluate: Perform evaluation and calculate metrics.
    """
    def __init__(self, data_loader: DataLoader, device: torch.device):
        self.data_loader = data_loader
        self.device = device
        self.model = None

    def load_model(self, model_path: str):
        """
        Load a trained agent model.

        Args:
            model_path (str): Path to the saved model.

        Raises:
            ModelNotFoundError: Raised if the model file is not found.
        """
        try:
            self.model = torch.load(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
        except FileNotFoundError:
            error_msg = f"Trained model not found at path: {model_path}"
            logger.error(error_msg)
            raise ModelNotFoundError(error_msg)

    def evaluate(self) -> AgentEvaluationMetrics:
        """
        Perform evaluation and calculate metrics.

        Returns:
            AgentEvaluationMetrics: An object containing the evaluation metrics.

        Raises:
            EvaluationError: Raised if the model is not loaded before evaluation.
        """
        if self.model is None:
            raise EvaluationError("Model must be loaded before evaluation.")

        self.model.eval()
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in self.data_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return calculate_metrics(preds, labels)

def main():
    # Load evaluation data
    data_file = os.path.join(Config.DATA_PATH, "evaluation_data.csv")
    try:
        data = load_data(data_file)
    except FileNotFoundError:
        logger.error("Evaluation data file not found. Exiting evaluation.")
        return

    # Create DataLoader
    dataset = AgentEvaluationDataset(data)
    data_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create evaluator instance
    evaluator = AgentEvaluator(data_loader, device)

    # Load trained model for evaluation
    model_path = os.path.join(Config.MODELS_PATH, "best_model.pth")
    evaluator.load_model(model_path)

    # Perform evaluation
    try:
        metrics = evaluator.evaluate()
    except EvaluationError as e:
        logger.error(f"Evaluation error: {str(e)}")
        return

    # Log and print evaluation metrics
    logger.info("Agent Evaluation Metrics:")
    logger.info(f"Precision: {metrics.precision}")
    logger.info(f"Recall: {metrics.recall}")
    logger.info(f"F1 Score: {metrics.f1_score}")
    logger.info(f"Accuracy: {metrics.accuracy}")
    logger.info(f"Confusion Matrix:\n {metrics.confusion_matrix}")

    print("Agent Evaluation Metrics:")
    print(f"Precision: {metrics.precision}")
    print(f"Recall: {metrics.recall}")
    print(f"F1 Score: {metrics.f1_score}")
    print(f"Accuracy: {metrics.accuracy}")
    print(f"Confusion Matrix:\n {metrics.confusion_matrix}")

if __name__ == "__main__":
    main()
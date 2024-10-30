import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

@dataclass
class ModelConfig:
    """Configuration for the logistic regression model."""
    learning_rate: float = 1e-4
    max_iterations: int = 10000
    lambda_reg: float = 1.0
    epsilon: float = 1e-15
    polynomial_degree: int = 2
    
class PolynomialLogisticRegression:
    """Polynomial Logistic Regression with regularization."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model with given configuration."""
        self.config = config or ModelConfig()
        self.thetas = None
        self.cost_history = []
        self.logger = self._setup_logger()
        
    @staticmethod
    def _setup_logger():
        """Setup logger for the model."""
        logger = logging.getLogger('PolynomialLogisticRegression')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial features up to specified degree."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        features = np.ones((n_samples, self.config.polynomial_degree + 1))
        
        for degree in range(1, self.config.polynomial_degree + 1):
            features[:, degree] = X[:, 0] ** degree
            
        return features
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Apply sigmoid function with numerical stability."""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cost with L2 regularization."""
        m = len(y)
        h = self._sigmoid(X @ self.thetas)
        
        # Compute cross-entropy loss
        epsilon = self.config.epsilon
        cost = -(1/m) * np.sum(
            y * np.log(h + epsilon) + 
            (1 - y) * np.log(1 - h + epsilon)
        )
        
        # Add regularization term (excluding bias term)
        reg_term = (self.config.lambda_reg / (2 * m)) * np.sum(self.thetas[1:] ** 2)
        
        return cost + reg_term
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'PolynomialLogisticRegression':
        """Train the model using gradient descent with regularization."""
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        m = len(y)
        
        # Initialize parameters
        self.thetas = np.random.randn(X_poly.shape[1])
        self.cost_history = []
        
        # Gradient descent
        for iteration in range(self.config.max_iterations):
            # Compute predictions
            h = self._sigmoid(X_poly @ self.thetas)
            
            # Compute gradients with regularization
            gradients = np.zeros_like(self.thetas)
            gradients[0] = (1/m) * np.sum(h - y)  # Bias term
            gradients[1:] = (1/m) * (X_poly[:, 1:].T @ (h - y)) + \
                           (self.config.lambda_reg / m) * self.thetas[1:]
            
            # Update parameters
            self.thetas -= self.config.learning_rate * gradients
            
            # Compute and store cost
            current_cost = self._compute_cost(X_poly, y)
            self.cost_history.append(current_cost)
            
            # Log progress
            if verbose and (iteration + 1) % 1000 == 0:
                self.logger.info(f"Iteration {iteration + 1}, Cost: {current_cost:.6f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of class 1."""
        X_poly = self._create_polynomial_features(X)
        return self._sigmoid(X_poly @ self.thetas)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
class ModelVisualizer:
    """Class for visualizing model results."""
    
    @staticmethod
    def plot_training_history(cost_history: List[float], figsize: Tuple[int, int] = (10, 6)):
        """Plot training cost history."""
        plt.figure(figsize=figsize)
        plt.plot(cost_history, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Training Cost History')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(model: PolynomialLogisticRegression, X: np.ndarray, y: np.ndarray,
                             figsize: Tuple[int, int] = (10, 6)):
        """Plot data points and decision boundary."""
        plt.figure(figsize=figsize)
        
        # Plot original data points
        plt.scatter(X[y == 0], np.zeros_like(X[y == 0]), 
                   color='red', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1], np.ones_like(X[y == 1]), 
                   color='blue', label='Class 1', alpha=0.6)
        
        # Plot decision boundary
        X_test = np.linspace(X.min(), X.max(), 300)
        y_pred = model.predict_proba(X_test)
        
        plt.plot(X_test, y_pred, color='green', 
                label='Decision Boundary', linewidth=2)
        
        plt.xlabel('Feature Value')
        plt.ylabel('Probability')
        plt.title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364,
                  0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569,
                  0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])
    y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1])
    
    # Configure and train model
    config = ModelConfig(
        learning_rate=1e-4,
        max_iterations=10000,
        lambda_reg=1.0,
        polynomial_degree=2
    )
    
    # Create and train model
    model = PolynomialLogisticRegression(config)
    model.fit(X, y)
    
    # Visualize results
    visualizer = ModelVisualizer()
    visualizer.plot_training_history(model.cost_history)
    visualizer.plot_decision_boundary(model, X, y)
    
    # Print final parameters
    print("\nFinal Model Parameters:")
    for i, theta in enumerate(model.thetas):
        print(f"theta_{i} = {theta:.6f}")
    
    # Calculate and print accuracy
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y) * 100
    print(f"\nTraining Accuracy: {accuracy:.2f}%")
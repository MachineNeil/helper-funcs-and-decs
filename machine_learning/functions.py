from typing import Optional, List, Union
from sklearn.metrics import classification_report, roc_curve, auc
import numpy as np
from typing import Tuple, Union, Literal
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def synth_categorical(
    number_of_classes: int = 2,
    examples: int = 100,
    percentage_of_pure_chance: Union[float, Literal[0, 1]] = 0.25
) -> Tuple[np.ndarray, np.ndarray]:

    if not (0 <= percentage_of_pure_chance <= 1):
        raise ValueError("Argument must be between 0 and 1.")
    else:
        y_test = np.random.choice(list(range(number_of_classes)), examples)
        y_pred = y_test.copy()
        misclassified_examples = int(percentage_of_pure_chance * examples)
        wrong_indices = np.random.choice(
            examples, misclassified_examples, replace=False)
        new_class = np.random.uniform(
            0, number_of_classes, misclassified_examples)
        y_pred[wrong_indices] = new_class

        return y_test, y_pred


def heatmap_confusion_matrix(
    y_test: Tuple[np.ndarray],
    y_pred: Tuple[np.ndarray],
    class_labels: Optional[List[str]] = None,
    filename: str = 'synth_data_{}.png'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
    save_file: bool = False,
    title: str = 'Confusion Matrix',
) -> None:

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    if class_labels:
        class_names = class_labels
    else:
        class_names = [f'Class {i}' for i in set(y_test)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('{}\nAccuracy: {:.2f}'.format(title, acc))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_file:
        plt.savefig(filename)
    plt.show()


def synth_continuous(
    start: float = -10.0,
    end: float = 10.0,
    num_examples: int = 100,
    noise_factor: float = 0.10,
    coefficient_limit: int = 5,
    polynomial_order: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    x = np.linspace(start, end, num_examples)

    gaussian_noise = np.random.normal(0, noise_factor, num_examples)
    coefficients = np.random.uniform(-coefficient_limit,
                                     coefficient_limit, polynomial_order + 1)

    y_true = np.polyval(coefficients, x)

    scaling_factor = 10 ** polynomial_order
    y_noisy = y_true + (gaussian_noise * scaling_factor)

    return x, y_true, y_noisy


def preprocess_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.10,
    val_size: float = None,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state)

    return x_train, x_test, x_val, y_train, y_test, y_val


def plot_history(
    history: dict,
    title: str = 'Loss and Accuracy Curves',
    save_path: bool = None,
    grid: bool = False
) -> None:
    
    if not all(metric in history.history for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy']):
        raise ValueError(
            "Missing required metrics in the provided history object.")

    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history.history['loss'], label='training_loss')
    plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history.history['accuracy'], label='training_accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path.replace('.png', '_accuracy.png'))
    plt.show()


def evaluate_model(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_labels: Optional[List[str]] = None
) -> None:

    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(f'Classification report:\n{report}')

    if len(np.unique(y_true)) > 2:
        raise ValueError(
            "ROC curve is only applicable for binary classification.")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False + rate')
    plt.ylabel('True + rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

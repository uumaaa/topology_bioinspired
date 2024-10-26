import numpy as np
from keras.api import Sequential, datasets
from keras.api.layers import Flatten, InputLayer, Dense, Conv2D, MaxPooling2D
from typing import List, Optional, Tuple
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def load_data(
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    datasets = {
        "iris": load_iris,
        "breast": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }
    all_topologies = {
        "iris": [[5], [3, 3], [4]],
        "breast": [[10, 5], [12, 10, 8], [10, 10]],
        "wine": [[12, 12], [8, 8], [16, 16]],
        "digits": [[(32, 3, 3), (2, 2), (64, 3, 3), (2, 2), 16]],
    }
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized.")
    X, y = datasets[dataset_name](return_X_y=True)

    X = normalize(X)
    if dataset_name == "digits":
        temp = np.zeros((X.shape[0], 8, 8))  # pyright: ignore
        for i, image in enumerate(X):
            temp[i] = np.reshape(image, (8, 8))
        X = temp

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=47
    )
    return X_train, X_test, y_train, y_test, all_topologies[dataset_name]  # pyright: ignore


def create_model(
    neurons_per_layer: List[int | Tuple],
    num_classes: int,
    num_features: Optional[int] = None,
    data_2D: bool = False,
    dim1_size: Optional[int] = None,
    dim2_size: Optional[int] = None,
) -> Sequential:
    if data_2D and (dim1_size is None or dim2_size is None):
        raise AttributeError(
            "If data_2D is true the data must have valid size dimensions"
        )
    if not data_2D and num_features is None:
        raise AttributeError(
            "If data_2D is false the data must have valid number of features"
        )

    model = Sequential()

    if data_2D:
        model.add(InputLayer(shape=(dim1_size, dim2_size, 1)))
    else:
        model.add(InputLayer(shape=(num_features,)))

    for i, num in enumerate(neurons_per_layer):
        if isinstance(num, int):
            if i > 0 and isinstance(neurons_per_layer[i - 1], Tuple):
                model.add(Flatten())
            model.add(Dense(num, activation="relu"))
        elif len(num) == 3:
            model.add(Conv2D(num[0], (num[1], num[2])))
        elif len(num) == 2:
            model.add(MaxPooling2D((num[0], num[1])))

    model.add(Dense(num_classes, activation="softmax"))
    return model

from keras.api import Sequential
from keras.api.layers import InputLayer, Dense
from typing import List


def create_model(
    neurons_per_layer: List[int],
    num_classes: int,
    num_features: int,
) -> Sequential:
    # Create individual Dense layers from the list, but unpack them inside the Sequential constructor
    model = Sequential(
        [
            InputLayer(shape=(num_features,)),
            *[
                Dense(num, activation="relu") for num in neurons_per_layer
            ],  # Unpacking here
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model

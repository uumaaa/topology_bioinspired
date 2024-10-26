import argparse
import time
import numpy as np
from typing import Callable, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy as dp
from keras.api.models import Model
from aux import load_data, create_model


class Vector:
    def __init__(self, model: Model, position: Optional[np.ndarray] = None) -> None:
        self.model = dp(model)
        flatten_weights = self.flatten_weights()
        if position is None:
            self.position = np.random.uniform(
                low=-1.0, high=1.0, size=flatten_weights.shape
            )
        else:
            self.position = position
        self.unflatten_weights()
        self.fitness = float("-inf")

    def flatten_weights(self) -> np.ndarray:
        """Flatten the model's weights into a 1D array."""
        weights = self.model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        return flat_weights

    def unflatten_weights(self) -> None:
        """Unflatten the 1D array of weights back into the model's structure."""
        weights = self.model.get_weights()
        new_weights = []
        start = 0
        for weight in weights:
            shape = weight.shape
            size = np.prod(shape)
            new_weights.append(self.position[start : start + size].reshape(shape))
            start += size
        self.model.set_weights(new_weights)


class DEforWA:
    def __init__(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_vectors: int,
        max_iterations: int,
        mutation_factor: float = 0.8,
        crossover_probability: float = 0.9,
        evaluation_method: str = "accuracy",
        print_at_iterations: int = 10,
    ):
        self.fitness_function: Callable[[Model], float] = (
            self.accuracy_fitness_function
            if evaluation_method == "accuracy"
            else self.f1_score_fitness_function
        )
        self.model = model
        self.num_vectors = num_vectors
        self.max_iterations = max_iterations
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        self.print_at_iterations = print_at_iterations
        self.population = [Vector(model) for _ in range(num_vectors)]
        self.global_best_position = self.population[0].position
        self.global_best_fitness = float("-inf")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self) -> Tuple[np.ndarray, float]:
        for i in range(self.max_iterations):
            total_fitness = np.zeros(len(self.population))

            for index, vector in enumerate(self.population):
                fitness = self.fitness_function(vector.model)
                total_fitness[index] = fitness
                vector.fitness = fitness

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(vector.position)

            for v, vector in enumerate(self.population):
                trial_vector = self.mutate_and_crossover(vector)
                trial_vector.unflatten_weights()
                trial_vector.fitness = self.fitness_function(trial_vector.model)

                if trial_vector.fitness > vector.fitness:
                    self.population[v] = trial_vector

            if i % self.print_at_iterations == 0:
                print(
                    f"Iteration {i}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:.4f}"
                )
                print(
                    f"Mean Fitness: {np.mean(total_fitness):.4f} \t Std: {np.std(total_fitness):.4f}"
                )

        return self.global_best_position, self.global_best_fitness

    def mutate_and_crossover(self, vector: Vector) -> Vector:  # exponential
        a, b, c = np.random.choice(self.population, 3, replace=False)  # pyright: ignore
        donor_vector = (
            a.position
            + self.mutation_factor * (self.global_best_position - a.position)
            + self.mutation_factor * (b.position - c.position)
        )
        trial_vector_position = np.copy(vector.position)
        forced_index = np.random.randint(0, vector.position.shape[0] - 1)
        trial_vector_position[forced_index] = donor_vector[forced_index]
        crossover = np.random.rand(len(vector.position)) < self.crossover_probability
        trial_vector_position[crossover] = donor_vector[crossover]

        trial_vector_position = np.clip(trial_vector_position, a_min=-1, a_max=1)

        return Vector(self.model, trial_vector_position)

    def accuracy_fitness_function(self, model: Model) -> float:
        y_train_pred = np.argmax(model(self.X_train), axis=1)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        y_test_pred = np.argmax(model(self.X_test), axis=1)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        return train_accuracy + test_accuracy

    def f1_score_fitness_function(self, model: Model) -> float:
        y_train_pred = np.argmax(model(self.X_train).numpy(), axis=1)
        y_test_pred = np.argmax(model(self.X_test).numpy(), axis=1)

        is_binary_classification = len(np.unique(self.y_train)) == 2

        if is_binary_classification:
            train_f1 = f1_score(self.y_train, y_train_pred, average="binary")
            test_f1 = f1_score(self.y_test, y_test_pred, average="binary")
        else:
            train_f1 = f1_score(self.y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(self.y_test, y_test_pred, average="weighted")
        return train_f1 + test_f1

    def __str__(self) -> str:
        return f"num_vectors:{self.num_vectors}\n"


def run_de(model, X_train, y_train, X_test, y_test):
    de = DEforWA(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_vectors=X_train.shape[1] * 10,
        max_iterations=150,
        evaluation_method="f1",
        mutation_factor=0.5,
        crossover_probability=0.8,
    )
    print(de)
    position, fitness = de.optimize()
    return fitness


def execute_on_all_datasets():
    datasets = ["iris", "breast", "wine"]
    with open("test/de-v10-cr80-mf50-i150-ff1-clipped.out", mode="w") as file:
        for dataset_name in datasets:
            file.write(f"{dataset_name}\n")
            X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)

            for topology in topologies:
                start_time = time.time()
                file.write(f"topology: {topology}\n")
                model = create_model(
                    topology, np.unique(y_train).shape[0], X_train.shape[1]
                )
                fitness = run_de(model, X_train, y_train, X_test, y_test)
                file.write(f"{fitness}\n")
                end_time = time.time()

                file.write(f"{(end_time-start_time):.6f}s")


def execute_on_single_dataset(dataset_name):
    X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)

    for topology in topologies:
        print(f"topology: {topology}")

        if dataset_name == "digits":
            model = create_model(
                topology, num_classes=10, data_2D=True, dim1_size=8, dim2_size=8
            )
        else:
            model = create_model(
                topology, np.unique(y_train).shape[0], X_train.shape[1]
            )

        fitness = run_de(model, X_train, y_train, X_test, y_test)
        print(f"Fitness: {fitness}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar PSO en diferentes datasets")
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=["iris", "breast", "wine", "digits", "all"],
        default="iris",
        help="dataset to test (iris, breast, wine, digits). default: 'iris'.",
    )
    args = parser.parse_args()
    dataset_name = args.test

    if dataset_name == "all":
        execute_on_all_datasets()
    else:
        execute_on_single_dataset(dataset_name)

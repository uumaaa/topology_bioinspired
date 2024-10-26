import argparse
import numpy as np
import time
import random
from typing import Callable, Tuple
from copy import deepcopy as dp
from keras.api.models import Model
from aux import create_model, load_data
from sklearn.metrics import accuracy_score, f1_score


class Whale:
    def __init__(self, model: Model, low_limit: float, high_limit: float) -> None:
        self.model = dp(model)
        flatten_weights = self.flatten_weights()
        self.position = np.random.uniform(
            low=low_limit, high=high_limit, size=flatten_weights.shape
        )
        self.whale_position = self.position
        self.unflatten_weights()
        self.velocity = np.zeros_like(self.position)
        self.current_fitness = float("-inf")
        self.A: float = 0
        self.C: float = 0
        self.low_limit = low_limit
        self.high_limit = high_limit

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

    def update_position_as_particle(
        self,
        global_best_position: np.ndarray,
        best_fitness: float,
        max_inertia_weight: float,
        min_inertia_weight: float,
        cognitive_constant: float,
        social_constant: float,
        average_fitness: float,
    ) -> None:
        """r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        dynamic_inertia = (
            max(
                min_inertia_weight
                - (max_inertia_weight - min_inertia_weight)
                * (best_fitness - self.current_fitness)
                / (best_fitness - average_fitness),
                0,
            )
            if (self.current_fitness >= average_fitness)
            else max_inertia_weight
        )
        cognitive_velocity = (
            cognitive_constant * r1 * (self.whale_position - self.position)
        )
        social_velocity = social_constant * r2 * (global_best_position - self.position)
        self.velocity = (
            dynamic_inertia * self.velocity + cognitive_velocity + social_velocity
        )"""
        self.position = np.clip(
            self.whale_position, a_min=self.low_limit, a_max=self.high_limit
        )
        self.unflatten_weights()

    def update_position_encircling(self, global_best_position: np.ndarray) -> None:
        D = np.abs(self.C * global_best_position - self.position)
        self.whale_position = global_best_position - self.A * D

    def update_position_bubble_netting(
        self, global_best_position: np.ndarray, b
    ) -> None:
        D = np.abs(global_best_position - self.position)
        l = np.random.uniform(low=-1, high=1)
        self.whale_position = (
            D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best_position
        )

    def update_position_random(self, random_whale_position: np.ndarray) -> None:
        D = np.abs(self.C * random_whale_position - self.position)
        self.whale_position = random_whale_position - self.A * D


class APSO_WOA:
    def __init__(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_whales: int,
        max_generations: int,
        max_inertia_weight: float = 0.9,
        min_inertia_weight: float = 0.4,
        cognitive_constant: float = 1.5,
        social_constant: float = 1.5,
        low_limit: float = -1,
        high_limit: float = 1,
        b: float = 1.0,
        print_at_iterations: int = 10,
        evaluation_method: str = "accuracy",
    ):
        self.fitness_function: Callable[[Model], float] = (
            self.accuracy_fitness_function
            if evaluation_method == "accuracy"
            else self.f1_score_fitness_function
        )
        self.num_whales = num_whales
        self.max_generations = max_generations
        self.max_inertia_weight = max_inertia_weight
        self.min_inertia_weight = min_inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.b = b
        self.print_at_iterations = print_at_iterations
        self.herd = [Whale(model, low_limit, high_limit) for _ in range(num_whales)]
        self.position_len = len(self.herd[0].position)
        print(self.position_len)
        self.global_best_position = self.herd[0].position
        self.global_best_fitness = float("-inf")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self) -> Tuple[np.ndarray, float]:
        for i in range(self.max_generations):
            fitness_values = np.zeros(len(self.herd))
            for index, whale in enumerate(self.herd):
                whale.current_fitness = self.test(whale.position)

                if whale.current_fitness > self.global_best_fitness:
                    self.global_best_fitness = whale.current_fitness
                    self.global_best_position = np.copy(whale.position)
                fitness_values[index] = whale.current_fitness

            average_fitness = np.average(fitness_values)
            a = 2 * (self.max_generations - i) / (self.max_generations)

            for whale in self.herd:
                r = np.random.random()
                r1 = np.random.random()
                whale.A = 2 * a * r - a
                whale.C = 2 * r1
                p = np.random.random()
                if p < 0.5:
                    if np.abs(whale.A) < 1:
                        whale.update_position_encircling(self.global_best_position)
                    elif np.abs(whale.A) >= 1:
                        random_whale = np.random.randint(0, self.num_whales)
                        whale.update_position_random(self.herd[random_whale].position)
                elif p >= 0.5:
                    whale.update_position_bubble_netting(
                        self.global_best_position, self.b
                    )

                whale.update_position_as_particle(
                    self.global_best_position,
                    self.global_best_fitness,
                    self.max_inertia_weight,
                    self.min_inertia_weight,
                    self.cognitive_constant,
                    self.social_constant,
                    average_fitness,
                )

            if i % self.print_at_iterations == 0:
                print(
                    f"Iteration {i}/{self.max_generations}, Best Fitness: {self.global_best_fitness:4f}"
                )
                print(
                    f"Mean {np.mean(fitness_values):4f} \t Std {np.std(fitness_values):4f}"
                )

        return self.global_best_position, self.global_best_fitness

    def accuracy_fitness_function(
        self,
        model: Model,
    ) -> float:
        y_train_pred = np.argmax(model(self.X_train), axis=1)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        y_test_pred = np.argmax(model(self.X_test), axis=1)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        return train_accuracy + test_accuracy

    def test(self, position: np.ndarray):
        return np.sum(-np.square(position))

    def f1_score_fitness_function(
        self,
        model: Model,
    ) -> float:
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


def run_apso_woa(model, X_train, y_train, X_test, y_test):
    pso = APSO_WOA(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_whales=50,
        max_generations=100,
        social_constant=2,
        low_limit=-1,
        high_limit=1,
        cognitive_constant=2,
        max_inertia_weight=0.9,
        min_inertia_weight=0.4,
        evaluation_method="f1",
    )
    position, fitness = pso.optimize()
    return fitness


def execute_on_all_datasets():
    datasets = ["iris", "breast", "wine"]
    with open("test/ipso-p25-w55-i150-ff1-clipped", mode="w") as file:
        for dataset_name in datasets:
            file.write(f"{dataset_name}\n")
            X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)

            for topology in topologies:
                start_time = time.time()
                file.write(f"topology: {topology}\n")
                model = create_model(
                    topology, np.unique(y_train).shape[0], X_train.shape[1]
                )
                fitness = run_apso_woa(model, X_train, y_train, X_test, y_test)
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

        fitness = run_apso_woa(model, X_train, y_train, X_test, y_test)
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

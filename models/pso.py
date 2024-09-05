import argparse
import numpy as np
from typing import Callable, Tuple
from copy import deepcopy as dp
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from keras.api.models import Model
from sklearn.model_selection import train_test_split
from aux import create_model
from sklearn.metrics import accuracy_score


class Particle:
    def __init__(
        self,
        model: Model,
    ) -> None:
        self.model = dp(model)
        flatten_weights = self.flatten_weights(self.model)
        self.position = (
            np.random.uniform(low=-1.0, high=1, size=flatten_weights.shape) * 100
        )
        self.unflatten_weights(self.position)
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_fitness = float("-inf")

    def flatten_weights(self, model: Model) -> np.ndarray:
        """Flatten the model's weights into a 1D array."""
        weights = model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        return flat_weights

    def unflatten_weights(self, flat_weights: np.ndarray) -> None:
        """Unflatten the 1D array of weights back into the model's structure."""
        weights = self.model.get_weights()
        new_weights = []
        start = 0
        for weight in weights:
            shape = weight.shape
            size = np.prod(shape)
            new_weights.append(flat_weights[start : start + size].reshape(shape))
            start += size
        self.model.set_weights(new_weights)

    def update_velocity(
        self,
        global_best_position: np.ndarray,
        inertia_weight: float,
        cognitive_constant: float,
        social_constant: float,
    ) -> None:
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_velocity = (
            cognitive_constant * r1 * (self.best_position - self.position)
        )
        social_velocity = social_constant * r2 * (global_best_position - self.position)
        self.velocity = (
            inertia_weight * self.velocity + cognitive_velocity + social_velocity
        )

    def update_position(self) -> None:
        self.position += self.velocity
        self.unflatten_weights(self.position)


class ParticleSwarmOptimizationForWeightAdjustment:
    def __init__(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_particles: int,
        max_iterations: int,
        inertia_weight: float = 0.1,
        cognitive_constant: float = 1.5,
        social_constant: float = 1.5,
        print_at_iterations: int = 10,
    ):
        self.func: Callable[
            [Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float
        ] = self.accuracy_fitness_function
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.print_at_iterations = print_at_iterations
        self.swarm = [Particle(model) for _ in range(num_particles)]
        self.global_best_position = self.swarm[0].position
        self.global_best_fitness = float("-inf")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self) -> Tuple[np.ndarray, float]:
        for i in range(self.max_iterations):
            total_fitness = np.zeros(len(self.swarm))
            for index, particle in enumerate(self.swarm):
                fitness = self.func(
                    particle.model,
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                )
                total_fitness[index] = fitness
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_position,
                    self.inertia_weight,
                    self.cognitive_constant,
                    self.social_constant,
                )
                particle.update_position()

            if i % self.print_at_iterations == 0:
                print(
                    f"Iteration {i}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:4f}"
                )
                print(
                    f"Mean {np.mean(total_fitness):4f} \t Std {np.std(total_fitness):4f}"
                )

        return self.global_best_position, self.global_best_fitness

    def accuracy_fitness_function(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        y_train_pred = np.argmax(model(X_train), axis=1)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = np.argmax(model(X_test), axis=1)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        return (2 / 3) * train_accuracy + (1 / 3) * test_accuracy


def load_data(
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datasets = {
        "iris": load_iris,
        "breast": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }

    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized.")

    data = datasets[dataset_name]()
    X = data.data
    labels = data.target
    return train_test_split(X, labels, test_size=0.2)  # pyright: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar PSO en diferentes datasets")

    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=["iris", "breast", "wine", "digits"],
        default="iris",
        help="dataset to test (iris, breast, wine, digits). default: 'iris'.",
    )

    args = parser.parse_args()
    dataset_name = args.test

    # Cargar datos según el dataset seleccionado
    X_train, X_test, y_train, y_test = load_data(dataset_name)

    model = create_model([32, 16, 16], np.unique(y_train).shape[0], X_train.shape[1])

    num_particles = 300
    max_iterations = 1000
    pso = ParticleSwarmOptimizationForWeightAdjustment(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_particles,
        max_iterations,
        inertia_weight=0.4,
    )
    position, fitness = pso.optimize()
    # print(f"Position: {position}")
    print(f"Fitness: {fitness}")

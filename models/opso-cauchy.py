import argparse
import numpy as np
import time
from typing import Callable, Tuple
from keras.api.models import Model
from aux import create_model, load_data
from sklearn.metrics import accuracy_score, f1_score


class Particle:
    def __init__(
        self,
        particle_position_length: int,
        a: float,
        b: float,
    ) -> None:
        self.a = a
        self.b = b
        self.position = np.random.uniform(
            low=self.a, high=self.b, size=particle_position_length
        )
        self.opposite_position = self.position
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_fitness = float("-inf")

    def update_velocity(
        self,
        global_best_position: np.ndarray,
        cognitive_constant: float,
        social_constant: float,
        dynamic_inertia: float,
    ) -> None:
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_velocity = (
            cognitive_constant * r1 * (self.best_position - self.position)
        )
        social_velocity = social_constant * r2 * (global_best_position - self.position)
        self.velocity = (
            dynamic_inertia * self.velocity + cognitive_velocity + social_velocity
        )

    def update_position(self) -> None:
        self.position += self.velocity
        self.position = np.clip(self.position, a_min=self.a, a_max=self.b)

    def update_opposite_position(self, a_t, b_t) -> None:
        self.opposite_position = a_t + b_t - self.position


class OPSO:
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
        u_constant: float = 1.0025,
        cognitive_constant: float = 1.5,
        social_constant: float = 1.5,
        a_t: float = -0.5,
        b_t: float = 0.5,
        opposite_probability=0.2,
        print_at_iterations: int = 10,
        evaluation_method: str = "accuracy",
    ):
        self.fitness_function: Callable[[np.ndarray], float] = (
            self.accuracy_fitness_function
            if evaluation_method == "accuracy"
            else self.test_function
        )
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.u_constant = u_constant
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.print_at_iterations = print_at_iterations
        self.model = model
        weights = self.model.get_weights()
        self.particle_position_length = np.concatenate(
            [w.flatten() for w in weights]
        ).shape[0]
        self.swarm = [
            Particle(self.particle_position_length, a_t, b_t)
            for _ in range(num_particles)
        ]
        self.double_swarm = np.zeros(
            (self.num_particles * 2, self.particle_position_length + 1)
        )
        print(self.particle_position_length)
        for idx, particle in enumerate(self.swarm):
            self.double_swarm[idx, : self.particle_position_length] = particle.position

        self.global_best_position = self.swarm[0].position
        self.global_best_fitness = float("-inf")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.opposite_probability = opposite_probability

    def optimize(self) -> Tuple[np.ndarray, float]:
        for i in range(self.max_iterations):
            dynamic_inertia = self.inertia_weight * self.u_constant ** (-i)

            # actualizacion de gbest y pbest
            total_fitness = np.zeros(self.num_particles)
            for idx, particle in enumerate(self.swarm):
                fitness = self.fitness_function(
                    particle.position,
                )
                total_fitness[idx] = fitness
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.position)

                self.double_swarm[idx, self.particle_position_length] = fitness

            if np.random.rand() < self.opposite_probability:
                a_t = self.swarm[0].position
                b_t = self.swarm[0].position
                for particle in self.swarm:
                    a_t = np.minimum(a_t, particle.position)
                    b_t = np.maximum(b_t, particle.position)
                for idx, particle in enumerate(self.swarm):
                    particle.update_opposite_position(a_t, b_t)
                    opposite_fitness = self.fitness_function(particle.opposite_position)
                    self.double_swarm[
                        idx + self.num_particles, : self.particle_position_length
                    ] = particle.opposite_position
                    self.double_swarm[
                        idx + self.num_particles, self.particle_position_length
                    ] = opposite_fitness
                double_swarm_sorted = self.double_swarm[
                    self.double_swarm[:, self.particle_position_length].argsort()[::-1]
                ]
                for idx, particle in enumerate(self.swarm):
                    particle.position = double_swarm_sorted[
                        idx, : self.particle_position_length
                    ]

            else:
                for idx, particle in enumerate(self.swarm):
                    particle.update_velocity(
                        self.global_best_position,
                        self.cognitive_constant,
                        self.social_constant,
                        dynamic_inertia,
                    )
                    particle.update_position()
                    self.double_swarm[idx, : self.particle_position_length] = (
                        particle.position
                    )

            if i % self.print_at_iterations == 0:
                print(
                    f"Iteration {i}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:4f}"
                )
                print(
                    f"Mean {np.mean(total_fitness):4f} \t Std {np.std(total_fitness):4f}"
                )

        return self.global_best_position, self.global_best_fitness

    def update_model(self, position) -> None:
        weights = self.model.get_weights()
        new_weights = []
        start = 0
        for weight in weights:
            shape = weight.shape
            size = np.prod(shape)
            new_weights.append(position[start : start + size].reshape(shape))
            start += size
        self.model.set_weights(new_weights)

    def accuracy_fitness_function(self, position: np.ndarray) -> float:
        self.update_model(position)
        y_train_pred = np.argmax(self.model(self.X_train), axis=1)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        y_test_pred = np.argmax(self.model(self.X_test), axis=1)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        return train_accuracy + test_accuracy

    def f1_score_fitness_function(
        self,
        position: np.ndarray,
    ) -> float:
        self.update_model(position)
        y_train_pred = np.argmax(self.model(self.X_train).numpy(), axis=1)
        y_test_pred = np.argmax(self.model(self.X_test).numpy(), axis=1)

        is_binary_classification = len(np.unique(self.y_train)) == 2

        if is_binary_classification:
            train_f1 = f1_score(self.y_train, y_train_pred, average="binary")
            test_f1 = f1_score(self.y_test, y_test_pred, average="binary")
        else:
            train_f1 = f1_score(self.y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(self.y_test, y_test_pred, average="weighted")
        return train_f1 + test_f1

    def test_function(self, position: np.ndarray) -> float:
        return -np.sum(np.square(position))


def run_opso(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    num_particles,
    inertia_weight,
    u_constant,
    social_constant,
    cognitive_constant,
):
    opso = OPSO(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_particles=num_particles,
        max_iterations=150,
        inertia_weight=inertia_weight,
        u_constant=u_constant,
        social_constant=social_constant,
        cognitive_constant=cognitive_constant,
        evaluation_method="f1",
    )
    position, fitness = opso.optimize()
    return fitness


# Función para ejecutar en todos los conjuntos de datos
def execute_on_all_datasets():
    datasets = ["iris", "breast", "wine"]
    with open("ipso.out", mode="w") as file:
        for dataset_name in datasets:
            file.write(f"{dataset_name}\n")
            X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)

            for topology in topologies:
                model = create_model(
                    topology, np.unique(y_train).shape[0], X_train.shape[1]
                )

                # Ejecución del PSO con diferentes configuraciones
                for num_particles in [
                    int(X_train.shape[1] * 20 * 0.2),
                    int(X_train.shape[1] * 20 * 0.8),
                    int(X_train.shape[1] * 20 * 0.9),
                    int(X_train.shape[1] * 20),
                    int(X_train.shape[1] * 20 * 1.1),
                    int(X_train.shape[1] * 20 * 1.2),
                    int(X_train.shape[1] * 20 * 1.8),
                ]:
                    for cognitive_constant in [1.7, 1.8, 1.9, 2]:
                        for social_constant in [1.7, 1.8, 1.9, 2]:
                            for min_values in [
                                [0.1, 1.005],
                                [0.3, 1.005],
                                [0.5, 1.005],
                                [0.7, 1.005],
                                [0.1, 1.035],
                                [0.3, 1.035],
                                [0.5, 1.035],
                                [0.7, 1.035],
                            ]:
                                start_time = time.time()
                                fitness = run_opso(
                                    model,
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    num_particles,
                                    min_values[0],
                                    min_values[1],
                                    cognitive_constant,
                                    social_constant,
                                )
                                file.write(f"topology: {topology}\n")
                                file.write(f"num_particles {num_particles}\n")
                                file.write(
                                    f"cognitive_constant {cognitive_constant} \t social_constant {social_constant}\n"
                                )
                                file.write(f"weight and u_constant {min_values}\n")
                                file.write(f"fitness\t{fitness}\n")
                                end_time = time.time()
                                file.write(f"time\t{(end_time-start_time):.6f}s\n")


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
        in2 = time.time()
        fitness = run_opso(
            model, X_train, y_train, X_test, y_test, 50, 0.3, 1.005, 2, 2
        )
        out = time.time()
        print(f"{out-in2}")
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

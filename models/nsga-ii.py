import random
import argparse
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from keras.api.models import Model
from copy import deepcopy as dp
from sklearn.metrics import accuracy_score
from aux import create_model, load_data


class Individual:
    def __init__(
        self,
        model: Model,
        genes: Optional[np.ndarray] = None,
        fitness: Optional[np.ndarray] = None,
    ) -> None:
        self.model: Model = dp(model)
        flatten_weights = self.flatten_weights()
        if genes is None:
            self.genes = (
                np.random.uniform(low=-1.0, high=1, size=flatten_weights.shape) * 5
            )
        else:
            self.genes = genes
        self.unflatten_weights()
        self.fitness: Optional[np.ndarray] = fitness
        self.rank: Optional[int] = None
        self.crowding_distance: Optional[float] = None
        self.domination_set: Set[Individual] = set()
        self.dominated_count: int = 0

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
            new_weights.append(self.genes[start : start + size].reshape(shape))
            start += size
        self.model.set_weights(new_weights)


class NSGAII:
    def __init__(
        self,
        model: Model,
        population_size: int,
        generations: int,
        mutation_rate: float,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        print_at_iterations: int = 10,
    ) -> None:
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.print_at_iterations = print_at_iterations

    def non_dominated_sort(
        self, population: List[Individual]
    ) -> List[List[Individual]]:
        fronts: List[List[Individual]] = [[]]
        for p in population:
            p.domination_set = set()
            p.dominated_count = 0
            for q in population:
                if self.dominates(p, q):
                    p.domination_set.add(q)
                elif self.dominates(q, p):
                    p.dominated_count += 1
            if p.dominated_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i: int = 0
        size_of_fronts = 0
        total_fronts: List[List[Individual]] = []  # Final list to store selected fronts
        stop = False  # Control variable to break from all loops

        while len(fronts[i]) > 0 and not stop:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in p.domination_set:
                    q.dominated_count -= 1
                    if q.dominated_count == 0:
                        q.rank = i + 1
                        next_front.append(q)

            if size_of_fronts + len(fronts[i]) <= self.population_size:
                total_fronts.append(fronts[i])
                size_of_fronts += len(fronts[i])
            else:
                # Add only the necessary amount of individuals to reach self.population_size
                remaining_slots = self.population_size - size_of_fronts
                if remaining_slots != 0:
                    total_fronts.append(fronts[i][:remaining_slots])
                stop = True  # Set stop to True to break the outer loops
                break

            i += 1
            fronts.append(next_front)

        return total_fronts

    def dominates(self, p: Individual, q: Individual) -> bool:
        return all(p_f <= q_f for p_f, q_f in zip(p.fitness, q.fitness)) and any(
            p_f < q_f for p_f, q_f in zip(p.fitness, q.fitness)
        )

    def crowding_distance(self, front: List[Individual]) -> None:
        distance: Dict[Individual, float] = {p: 0 for p in front}
        num_objectives: int = len(front[0].fitness)

        for m in range(num_objectives):
            front.sort(key=lambda x: x.fitness[m])
            distance[front[0]] = distance[front[-1]] = float("inf")
            for i in range(1, len(front) - 1):
                if front[-1].fitness[m] - front[0].fitness[m] == 0:
                    pass
                else:
                    distance[front[i]] += (
                        front[i + 1].fitness[m] - front[i - 1].fitness[m]
                    ) / (front[-1].fitness[m] - front[0].fitness[m])

        for p in front:
            p.crowding_distance = distance[p]

    def tournament_selection(self, population: List[Individual], k: int) -> Individual:
        selected: List[Individual] = random.sample(population, k)
        return min(selected, key=lambda x: (x.rank, -x.crowding_distance))

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        point: int = random.randint(1, len(parent1.genes) - 1)
        child1_genes: np.ndarray = np.hstack(
            (parent1.genes[:point], parent2.genes[point:])
        )
        child2_genes: np.ndarray = np.hstack(
            (parent2.genes[:point], parent1.genes[point:])
        )
        return Individual(parent1.model, child1_genes), Individual(
            parent2.model, child2_genes
        )

    def mutation(self, individual: Individual) -> None:
        for i in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                individual.genes[i] *= random.uniform(0.1, 1.9)

    def optimize(self) -> List[Individual]:
        # Initialize the population
        population: List[Individual] = [
            Individual(self.model) for _ in range(self.population_size)
        ]
        for individual in population:
            individual.fitness = self.test_function(individual.genes)

        fronts: List[List[Individual]] = self.non_dominated_sort(population)
        for front in fronts:
            self.crowding_distance(front)
        # Evolution loop
        for generation in range(self.generations):
            # Create a new population using selection, crossover, and mutation
            new_population: List[Individual] = []
            while len(new_population) < self.population_size:
                parent1: Individual = self.tournament_selection(population, 2)
                parent2: Individual = self.tournament_selection(population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutation(child1)
                self.mutation(child2)
                child1.fitness = self.test_function(child1.genes)
                child2.fitness = self.test_function(child2.genes)
                new_population.extend([child1, child2])

            # Replace old population with the new one
            appended_population = population + new_population
            # Perform non-dominated sorting and assign crowding distances
            fronts: List[List[Individual]] = self.non_dominated_sort(
                appended_population
            )
            population = []
            total_fitness = np.zeros(
                (self.population_size, len(appended_population[0].fitness))
            )
            indiviual_number = 0
            for front in fronts:
                self.crowding_distance(front)
                for p in front:
                    total_fitness[indiviual_number, :] = p.fitness[:]
                    population.append(p)
                    indiviual_number += 1
            if generation % self.print_at_iterations == 0:
                print(f"Generation {generation}/{self.generations}\t")
                print(
                    f"Mean: {np.mean(total_fitness,axis=0)}\t Std: {np.std(total_fitness,axis=0)}"
                )
        # Return the final Pareto front (first front)
        return self.non_dominated_sort(population)[0]

    def accuracy_fitness_function(
        self,
        model: Model,
    ) -> np.ndarray:
        y_train_pred = np.argmax(model(self.X_train), axis=1)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        y_test_pred = np.argmax(model(self.X_test), axis=1)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        return np.array([1 - train_accuracy, 1 - test_accuracy])

    def test_function(self, position: np.ndarray) -> np.ndarray:
        return np.array([np.sum(np.square(position)), 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecutar NSGA-II en diferentes datasets"
    )

    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=["iris", "breast", "wine", "digits", "all"],
        default="iris",
        help="dataset to test (iris, breast, wine, digits, all). default: 'iris'.",
    )

    args = parser.parse_args()
    dataset_name = args.test
    if dataset_name == "all":
        with open("test/nsga-ii-80.out", mode="w") as file:
            datasets = ["iris", "breast", "wine"]
            for dataset_name in datasets:
                file.write(f"{dataset_name}\n")
                X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)
                for topology in topologies:
                    file.write(f"topology: {topology}\n")
                    # Crear un modelo basado en las dimensiones de los datos
                    model = create_model(
                        topology, np.unique(y_train).shape[0], X_train.shape[1]
                    )

                    nsga2 = NSGAII(
                        model=model,
                        population_size=X_train.shape[1] * 14,
                        generations=150,
                        mutation_rate=0.80,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        print_at_iterations=10,
                    )
                    pareto_front = nsga2.optimize()

                    # Imprimir los resultados del Pareto front
                    for individual in pareto_front:
                        file.write(f"{1 - individual.fitness}\n")
    else:
        X_train, X_test, y_train, y_test, topologies = load_data(dataset_name)
        for topology in topologies:
            print(f"with topology: {topology}")
            # Crear un modelo basado en las dimensiones de los datos
            model = create_model(
                topology, np.unique(y_train).shape[0], X_train.shape[1]
            )

            nsga2 = NSGAII(
                model=model,
                population_size=10,
                generations=150,
                mutation_rate=0.2,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                print_at_iterations=10,
            )
            pareto_front = nsga2.optimize()

            # Imprimir los resultados del Pareto front
            print(f"Results for topology: {topology}")
            for individual in pareto_front:
                print(f"Fitness: {1 - individual.fitness}")

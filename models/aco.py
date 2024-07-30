import numpy as np
from typing import List, Callable, Tuple


class AntColonyOptimization:
    def __init__(self, func: Callable[[List[Tuple[float, float]], List[int]], float], num_nodes: int, coordinates: List[Tuple[float, float]], num_ants: int, num_iterations: int, alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.5, pheromone_deposit: float = 100.0, optimize_for: str = "min",print_at_iteration = 10):
        self.func = func
        self.num_nodes = num_nodes
        self.coordinates = coordinates
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.optimize_for = optimize_for
        self.pheromone = np.ones((num_nodes, num_nodes)) / num_nodes
        self.best_route = None
        self.best_distance = float('inf') if optimize_for == "min" else -float('inf')
        self.print_at_iteration = print_at_iteration

    def run(self) -> Tuple[List[int], float]:
        for iteration in range(self.num_iterations):
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)
            for route in all_routes:
                distance = self.func(self.coordinates, route)
                if (self.optimize_for == "min" and distance < self.best_distance) or (self.optimize_for == "max" and distance > self.best_distance):
                    self.best_distance = distance
                    self.best_route = route
            if iteration % self.print_at_iteration == 0:
                print(f"Iteration {iteration+1}/{self.num_iterations}, Best Distance: {self.best_distance}")
        return self.best_route, self.best_distance

    def construct_solutions(self) -> List[List[int]]:
        all_routes = []
        for _ in range(self.num_ants):
            route = [np.random.randint(self.num_nodes)]
            while len(route) < self.num_nodes:
                distances = self.distance(route[-1])
                move_prob = self.pheromone[route[-1], :] ** self.alpha * (1.0 / distances) ** self.beta
                move_prob[list(route)] = 0
                if move_prob.sum() == 0:
                    move_prob = np.ones(self.num_nodes)
                    move_prob[list(route)] = 0
                move_prob /= move_prob.sum()
                next_node = np.random.choice(range(self.num_nodes), p=move_prob)
                route.append(next_node)
            all_routes.append(route)
        return all_routes

    def distance(self, node: int) -> np.ndarray:
        distances = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if i != node:
                distances[i] = np.sqrt((self.coordinates[node][0] - self.coordinates[i][0]) ** 2 + (self.coordinates[node][1] - self.coordinates[i][1]) ** 2)
        return distances + 1e-10

    def update_pheromones(self, all_routes: List[List[int]]) -> None:
        self.pheromone *= (1 - self.evaporation_rate)
        for route in all_routes:
            distance = self.func(self.coordinates, route)
            for i in range(len(route) - 1):
                if self.optimize_for == "min":
                    self.pheromone[route[i], route[i + 1]] += self.pheromone_deposit / distance
                else:
                    self.pheromone[route[i], route[i + 1]] += distance / self.pheromone_deposit


if __name__ == '__main__':
    def tsp_distance(coordinates: List[Tuple[float, float]], route: List[int]) -> float:
        distance = 0.0
        for i in range(len(route) - 1):
            distance += np.sqrt((coordinates[route[i]][0] - coordinates[route[i + 1]][0]) ** 2 + (coordinates[route[i]][1] - coordinates[route[i + 1]][1]) ** 2)
        distance += np.sqrt((coordinates[route[-1]][0] - coordinates[route[0]][0]) ** 2 + (coordinates[route[-1]][1] - coordinates[route[0]][1]) ** 2)
        return distance
    num_nodes = 5
    coordinates: List[Tuple[float, float]] = [(_,_) for _ in range(num_nodes)]
    num_ants = 20
    num_iterations = 500

    # Optimize for minimum
    aco_min = AntColonyOptimization(tsp_distance, num_nodes, coordinates, num_ants, num_iterations, optimize_for="max",print_at_iteration=50)
    best_route, best_distance = aco_min.run()
    print(f"Best Route: {best_route}, Best Distance: {best_distance}")



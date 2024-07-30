import numpy as np
from typing import Callable, List, Tuple


class Particle:
    def __init__(self, bounds: List[Tuple[float, float]], optimize_for: str = "min") -> None:
        self.position: np.ndarray = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity: np.ndarray = np.random.rand(len(bounds))  # Inicializar velocidad aleatoriamente
        self.best_position: np.ndarray = np.copy(self.position)  # Mejor posición conocida
        self.best_fitness: float = float('inf') if optimize_for == "min" else -float('inf')  # Mejor fitness conocido

    def update_velocity(self, global_best_position: np.ndarray, inertia_weight: float, cognitive_constant: float, social_constant: float) -> None:
        r1: np.ndarray = np.random.rand(len(self.position))
        r2: np.ndarray = np.random.rand(len(self.position))
        cognitive_velocity: np.ndarray = cognitive_constant * r1 * (self.best_position - self.position)
        social_velocity: np.ndarray = social_constant * r2 * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds: List[Tuple[float, float]]) -> None:
        self.position = self.position + self.velocity
        # Aplicar límites a las posiciones
        for i, (low, high) in enumerate(bounds):
            if self.position[i] < low:
                self.position[i] = low
            elif self.position[i] > high:
                self.position[i] = high

class ParticleSwarmOptimization:
    def __init__(self, func: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]], num_particles: int, max_iterations: int, optimize_for: str = "min", inertia_weight: float = 0.5, cognitive_constant: float = 1.5, social_constant: float = 1.5, print_at_iterations: int = 10):
        self.func = func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.optimize_for = optimize_for
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.print_at_iterations = print_at_iterations
        self.swarm = [Particle(bounds, optimize_for) for _ in range(num_particles)]
        self.global_best_position: np.ndarray = np.copy(self.swarm[0].position)
        self.global_best_fitness: float = float('inf') if optimize_for == "min" else -float('inf')

    def optimize(self) -> Tuple[np.ndarray, float]:
        for i in range(self.max_iterations):
            for particle in self.swarm:
                fitness = self.func(particle.position)
                if (self.optimize_for == "min" and fitness < particle.best_fitness) or (self.optimize_for == "max" and fitness > particle.best_fitness):
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                if (self.optimize_for == "min" and fitness < self.global_best_fitness) or (self.optimize_for == "max" and fitness > self.global_best_fitness):
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, self.inertia_weight, self.cognitive_constant, self.social_constant)
                particle.update_position(self.bounds)

            if i % self.print_at_iterations == 0:
                print(f"Iteration {i+1}/{self.max_iterations}, Best Fitness: {self.global_best_fitness}")

        return self.global_best_position, self.global_best_fitness


if __name__ == '__main__':
    def objective_function(x: np.ndarray) -> float:
        return np.sum(x ** 2)

    bounds = [(-10, 10), (-40, 4), (-20, 25)]  # Bounds for each dimension
    num_particles = 30
    max_iterations = 500
    pso = ParticleSwarmOptimization(objective_function, bounds, num_particles, max_iterations, optimize_for="min",print_at_iterations=50)
    best_position, best_fitness = pso.optimize()
    print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")
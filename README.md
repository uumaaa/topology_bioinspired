# Optimization Algorithms Documentation

## Overview

This repository contains implementations of two popular optimization algorithms:
- **Ant Colony Optimization (ACO)**
- **Particle Swarm Optimization (PSO)**

These algorithms are used to find optimal solutions in complex spaces. Both algorithms are implemented in Python and are demonstrated in Jupyter Notebooks.

## Algorithms

### Ant Colony Optimization (ACO)

ACO is inspired by the foraging behavior of ants. It uses a probabilistic technique to explore and exploit the solution space based on pheromone trails left by ants. The algorithm is particularly useful for discrete optimization problems like the Traveling Salesman Problem (TSP).

#### `AntColonyOptimization` Class

- **Attributes:**
  - `func` (Callable[[List[Tuple[float, float]], List[int]], float]): The objective function to be minimized or maximized.
  - `num_nodes` (int): Number of nodes in the problem.
  - `bounds` (List[Tuple[float, float]]): The bounds of each dimension.
  - `num_ants` (int): Number of ants in the colony.
  - `num_iterations` (int): Number of iterations for the optimization process.
  - `alpha` (float): Influence of pheromone on the probability of moving to a node.
  - `beta` (float): Influence of heuristic information on the probability of moving to a node.
  - `evaporation_rate` (float): Rate at which pheromones evaporate.
  - `pheromone_deposit` (float): Amount of pheromone deposited on the path.
  - `optimize_for` (str): Optimization type, either `"min"` or `"max"`.

- **Methods:**
  - `run() -> Tuple[List[int], float]`: Runs the ACO algorithm and returns the best route and its distance.
  - `construct_solutions() -> List[List[int]]`: Constructs solutions based on the pheromone matrix.
  - `distance(node: int) -> np.ndarray`: Calculates the distance from a given node to all other nodes.
  - `update_pheromones(all_routes: List[List[int]]) -> None`: Updates the pheromone matrix based on the routes and distances.

### Particle Swarm Optimization (PSO)

PSO is inspired by the social behavior of birds flocking or fish schooling. It uses a population of candidate solutions (particles) which move around in the search space, adjusting their positions based on their own experience and that of their neighbors.

#### `Particle` Class

- **Attributes:**
  - `position` (np.ndarray): Current position of the particle.
  - `velocity` (np.ndarray): Current velocity of the particle.
  - `best_position` (np.ndarray): Best position found by the particle.
  - `best_fitness` (float): Best fitness value found by the particle.

- **Methods:**
  - `update_velocity(global_best_position: np.ndarray, inertia_weight: float, cognitive_constant: float, social_constant: float) -> None`: Updates the velocity of the particle based on its best position and the global best position.
  - `update_position(bounds: List[Tuple[float, float]]) -> None`: Updates the position of the particle and ensures it stays within bounds.

#### `ParticleSwarmOptimization` Class

- **Attributes:**
  - `func` (Callable[[np.ndarray], float]): The objective function to be minimized or maximized.
  - `x0` (List[float]): Initial position for particles.
  - `bounds` (List[Tuple[float, float]]): Bounds for each dimension.
  - `num_particles` (int): Number of particles in the swarm.
  - `max_iterations` (int): Maximum number of iterations.
  - `optimize_for` (str): Optimization type, either `"min"` or `"max"`.
  - `inertia_weight` (float): Weight of the inertia in velocity update.
  - `cognitive_constant` (float): Cognitive constant in velocity update.
  - `social_constant` (float): Social constant in velocity update.
  - `print_at_iterations` (int): Frequency of printing the progress.

- **Methods:**
  - `optimize() -> Tuple[np.ndarray, float]`: Runs the PSO algorithm and returns the best position and fitness value.

## Jupyter Notebooks

- **Ant Colony Optimization Notebook**: `aco.ipynb`
  - Path: `path/to/aco.ipynb`
  - Contains implementation details, usage, and examples for the ACO algorithm.

- **Particle Swarm Optimization Notebook**: `pso.ipynb`
  - Path: `path/to/pso.ipynb`
  - Contains implementation details, usage, and examples for the PSO algorithm.

## Usage

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/usuario/repositorio.git
   cd repositorio

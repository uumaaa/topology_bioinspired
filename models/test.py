import numpy as np

# Definimos los parámetros
num_particles = 50  # Número de partículas en el enjambre
dimensions = 43  # Dimensiones del vector (variable) a optimizar
max_iterations = 150  # Máximo de iteraciones
c1, c2 = 2.0, 2.0  # Coeficientes de aceleración
w = 0.7  # Factor de inercia
v_max = 5.0  # Límite máximo de velocidad de las partículas


# Función objetivo: suma de los cuadrados de los elementos del vector
def objective_function(x):
    return np.sum(-(x**2))


# Inicialización de partículas
positions = np.random.uniform(-10, 10, (num_particles, dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
personal_best_positions = np.copy(positions)
personal_best_scores = np.apply_along_axis(
    objective_function, 1, personal_best_positions
)
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
global_best_score = np.max(personal_best_scores)

# Iteración principal de PSO
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Actualizamos la velocidad de cada partícula
        inertia = w * velocities[i]
        cognitive_component = (
            c1 * np.random.rand() * (personal_best_positions[i] - positions[i])
        )
        social_component = c2 * np.random.rand() * (global_best_position - positions[i])
        velocities[i] = inertia + cognitive_component + social_component

        # Limitamos la velocidad para evitar cambios grandes
        velocities[i] = np.clip(velocities[i], -v_max, v_max)

        # Actualizamos la posición de la partícula
        positions[i] += velocities[i]

        # Evaluamos la nueva posición
        current_score = objective_function(positions[i])

        # Actualizamos el mejor personal de cada partícula
        if current_score > personal_best_scores[i]:
            personal_best_scores[i] = current_score
            personal_best_positions[i] = positions[i]

        # Actualizamos el mejor global si encontramos una mejor solución
    if np.max(personal_best_scores) > global_best_score:
        global_best_score = np.max(personal_best_scores)
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]

    # Imprimimos el progreso
    print(
        f"Iteración {iteration+1}/{max_iterations} - Mejor puntuación global: {global_best_score}"
    )

# Resultados finales
print("\nResultado:")
print("Mejor posición encontrada:", global_best_position)
print("Mejor puntuación encontrada:", global_best_score)

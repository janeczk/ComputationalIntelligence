import random
import math
import matplotlib.pyplot as plt

NUM_CITIES = 30
POPULATION_COUNT = 100
BEST_COUNT = POPULATION_COUNT // 5
GRID_SIZE = 500
DIST_FROM_BOUND = 30
MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 5
GENERATIONS = 1000


def generate_cities(n, size=GRID_SIZE):
    cities = [(random.randint(0 + DIST_FROM_BOUND, size - DIST_FROM_BOUND),
               random.randint(0 + DIST_FROM_BOUND, size - DIST_FROM_BOUND)) for _ in range(n)]
    return cities

def plot_route(route, cities):
    ordered_cities = [cities[i] for i in route] + [cities[route[0]]]
    x, y = zip(*ordered_cities)

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o', c='green', markersize=5)
    plt.scatter(*zip(*cities), c='blue', s=40)
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.title("Najlepsza trasa po ewolucji")
    plt.grid(True)
    plt.show()

def evaluate_distance(order, cities):
    total_distance = 0
    for i in range(len(order)):
        city_a = cities[order[i]]
        city_b = cities[(order[(i + 1) % len(order)])]
        dx = city_a[0] - city_b[0]
        dy = city_a[1] - city_b[1]
        total_distance += math.hypot(dx, dy)
    return total_distance

def generate_population(population_size, num_cities):
    population = []
    base_order = list(range(num_cities))
    for _ in range(population_size):
        individual = base_order.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def select_best_individuals(population, cities, count):
    scored = [(individual, evaluate_distance(individual, cities)) for individual in population]
    scored.sort(key=lambda x: x[1])
    best_individuals = [ind for ind, dist in scored[:count]]
    return best_individuals

def tournament_selection(population, cities, k):
    """Wybiera najlepszego osobnika z losowej próbki k-osobników"""
    tournament = random.sample(population, k)
    winner = min(tournament, key=lambda ind: evaluate_distance(ind, cities))
    return winner

def crossover_ox(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]

    fill_values = [gene for gene in parent2 if gene not in child]
    fill_idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[fill_idx]
            fill_idx += 1
    return child

def mutate_swap(individual):
    """Losowo zamienia dwa miasta w chromosomie"""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def create_new_population(population, cities, elite_individuals, population_size):
    new_population = elite_individuals.copy()

    while len(new_population) < population_size:
        parent1 = tournament_selection(population, cities, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, cities, TOURNAMENT_SIZE)
        child = crossover_ox(parent1, parent2)
        mutated_child = mutate_swap(child)
        new_population.append(mutated_child)

    return new_population

def main():
    cities = generate_cities(NUM_CITIES)
    population = generate_population(POPULATION_COUNT, NUM_CITIES)

    for gen in range(GENERATIONS):
        best_individuals = select_best_individuals(population, cities, BEST_COUNT)
        best = best_individuals[0]
        distance = evaluate_distance(best, cities)

        if gen % 10 == 0:
            print(f"Generacja {gen} | Najlepszy dystans: {distance:.2f}")
            plot_route(best, cities)

        population = create_new_population(population, cities, best_individuals, POPULATION_COUNT)

    # Finalny wynik
    best_final = select_best_individuals(population, cities, 1)[0]
    print("Najlepszy osobnik:", best_final)
    print("Długość trasy:", evaluate_distance(best_final, cities))
    plot_route(best_final, cities)

if __name__ == '__main__':
    main()

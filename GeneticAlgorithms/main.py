import random
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
from config import *

def generate_cities(n, size):
    return [
        (random.randint(DIST_FROM_BOUND, size - DIST_FROM_BOUND),
         random.randint(DIST_FROM_BOUND, size - DIST_FROM_BOUND))
        for _ in range(n)
    ]


def generate_cities_circle(n, size, margin=30):
    radius = (size // 2) - margin
    center_x = size // 2
    center_y = size // 2

    cities = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        cities.append((x, y))

    return cities


def plot_route(route, cities, generation=None):
    ordered_cities = [cities[i] for i in route] + [cities[route[0]]]
    x, y = zip(*ordered_cities)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, '-o', c='green', markersize=5)
    plt.scatter(*zip(*cities), c='blue', s=40)
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)

    title = "Najlepsza trasa"
    if generation is not None:
        title += f" (Generacja {generation})"
    plt.title(title)
    plt.show()

def evaluate_distance(order, cities):
    return sum(
        math.hypot(cities[order[i]][0] - cities[order[(i + 1) % len(order)]][0],
                   cities[order[i]][1] - cities[order[(i + 1) % len(order)]][1])
        for i in range(len(order))
    )

def generate_population(pop_size, num_cities):
    base_order = list(range(num_cities))
    return [random.sample(base_order, len(base_order)) for _ in range(pop_size)]

def select_best_individuals(population, cities, count):
    return sorted(population, key=lambda ind: evaluate_distance(ind, cities))[:count]

def roulette_selection(best_individuals, cities):
    fitness = [1 / evaluate_distance(ind, cities) for ind in best_individuals]
    total = sum(fitness)
    probs = [f / total for f in fitness]
    return random.choices(best_individuals, weights=probs, k=1)[0]

def tournament_selection(population, cities, k=3):
    return min(random.sample(population, k), key=lambda ind: evaluate_distance(ind, cities))

def crossover_ox(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = p1[start:end]
    fill_values = [gene for gene in p2 if gene not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[idx]
            idx += 1
    return child

def mutate_insert(ind):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(ind)), 2)
        gene = ind.pop(i)
        ind.insert(j, gene)
    return ind

def mutate_swap(ind):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutate(individual, insert_prob):
    return mutate_insert(individual.copy()) if random.random() < insert_prob else mutate_swap(individual.copy())

def create_new_population(best_individuals, all_population, cities, pop_size, insert_prob, selection_method):
    new_population = [best_individuals[0]]
    while len(new_population) < pop_size:
        if selection_method == "roulette":
            p1 = roulette_selection(best_individuals, cities)
            p2 = roulette_selection(best_individuals, cities)
        else:
            p1 = tournament_selection(best_individuals, cities)
            p2 = tournament_selection(best_individuals, cities)
        child = crossover_ox(p1, p2)
        new_population.append(mutate(child, insert_prob))
    return new_population

def run_genetic_algorithm(
    num_cities=NUM_CITIES,
    pop_size=POPULATION_COUNT,
    best_count=BEST_COUNT,
    grid_size=GRID_SIZE,
    dist_bound=DIST_FROM_BOUND,
    mutation_rate=MUTATION_RATE,
    insert_prob=INSERT_PROB,
    generations=GENERATIONS,
    selection_method=SELECTION_METHOD,
    seed=SEED,
    show_progress=False
):
    random.seed(seed)
    global MUTATION_RATE
    MUTATION_RATE = mutation_rate


    cities = generate_cities(num_cities, grid_size)
    population = generate_population(pop_size, num_cities)

    best_distance = float('inf')
    no_improve = 0
    best_distances = []
    avg_distances = []

    for gen in range(generations):
        best_ind = select_best_individuals(population, cities, best_count)
        current_best = best_ind[0]
        current_dist = evaluate_distance(current_best, cities)

        best_distances.append(current_dist)
        avg = sum(evaluate_distance(ind, cities) for ind in population) / len(population)
        avg_distances.append(avg)

        if current_dist < best_distance:
            best_distance = current_dist
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > 0 and no_improve % 100 == 0:
            MUTATION_RATE = min(1.0, MUTATION_RATE * 1.15)

        if show_progress and gen % 50 == 0:
            clear_output(wait=True)
            print(f"Generacja {gen} | Najlepszy dystans: {current_dist:.2f}")
            plot_route(current_best, cities, gen)

        population = create_new_population(
            best_ind, population, cities,
            pop_size, insert_prob, selection_method
        )

    best_final = select_best_individuals(population, cities, 1)[0]
    if show_progress:
        print("Najlepszy osobnik:", best_final)
        print("Długość trasy:", evaluate_distance(best_final, cities))
        plot_route(best_final, cities, generation=generations)

    return {
        "cities": cities,
        "best_individual": best_final,
        "best_distance": evaluate_distance(best_final, cities),
        "best_distances": best_distances,
        "avg_distances": avg_distances,
        "final_mutation_rate": MUTATION_RATE
    }

if __name__ == '__main__':
    run_genetic_algorithm(show_progress=True)
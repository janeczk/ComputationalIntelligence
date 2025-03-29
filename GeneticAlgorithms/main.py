import random
import math
import matplotlib.pyplot as plt

NUM_CITIES = 10
POPULATION_COUNT = 50
BEST_COUNT = 5  # ilu najlepszych osobników wybieramy
GRID_SIZE = 200
DIST_FROM_BOUND = 30


def generate_cities(n, size=GRID_SIZE):
    cities = [(random.randint(0 + DIST_FROM_BOUND, size - DIST_FROM_BOUND),
               random.randint(0 + DIST_FROM_BOUND, size - DIST_FROM_BOUND)) for _ in range(n)]
    return cities

def plot_route(route, cities):
    """Rysuje trasę zgodnie z kolejnością w route"""
    ordered_cities = [cities[i] for i in route] + [cities[route[0]]]  # zamyka pętlę
    x, y = zip(*ordered_cities)

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o', c='green', markersize=5)
    plt.scatter(*zip(*cities), c='blue', s=40)
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.title("Najkrótsza wylosowana trasa")
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
    """Zwraca 'count' najlepszych osobników (o najkrótszej trasie)"""
    scored = [(individual, evaluate_distance(individual, cities)) for individual in population]
    scored.sort(key=lambda x: x[1])  # sortujemy po długości trasy
    best_individuals = [ind for ind, dist in scored[:count]]
    return best_individuals

def main():
    cities = generate_cities(NUM_CITIES)
    population = generate_population(POPULATION_COUNT, NUM_CITIES)

    best_individuals = select_best_individuals(population, cities, BEST_COUNT)

    print("Najlepszy osobnik:", best_individuals[0])
    print("Najkrótsza trasa:", evaluate_distance(best_individuals[0], cities))

    plot_route(best_individuals[0], cities)

if __name__ == '__main__':
    main()

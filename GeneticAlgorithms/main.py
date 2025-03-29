import random
import math
import matplotlib.pyplot as plt

NUM_CITIES = 10
GRID_SIZE = 200
DIST_FROM_BOUND = 30


def generate_cities(n, size=GRID_SIZE):
    """Generuje n losowych miast na planszy o rozmiarze GRID_SIZE"""
    cities = [(random.randint(0+DIST_FROM_BOUND, size-DIST_FROM_BOUND),
               random.randint(0+DIST_FROM_BOUND, size-DIST_FROM_BOUND)) for _ in range(n)]
    return cities

def plot_cities(cities):
    """Wyświetla miasta na wykresie"""
    x, y = zip(*cities)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='blue', s=40)
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.show()

def evaluate_distance(order, cities):
    """Oblicza całkowitą długość trasy dla danej kolejności odwiedzania miast"""
    total_distance = 0
    for i in range(len(order)):
        city_a = cities[order[i]]
        city_b = cities[order[(i + 1) % len(order)]]  # wraca do pierwszego miasta
        dx = city_a[0] - city_b[0]
        dy = city_a[1] - city_b[1]
        distance = math.hypot(dx, dy)
        total_distance += distance
    return total_distance

def main():
    cities = generate_cities(NUM_CITIES)
    plot_cities(cities)

    order = list(range(NUM_CITIES))
    random.shuffle(order)
    print("Kolejność:", order)

    distance = evaluate_distance(order, cities)
    print("Całkowita długość trasy:", distance)

if __name__ == '__main__':
    main()

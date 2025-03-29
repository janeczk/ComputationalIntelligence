import random
import matplotlib.pyplot as plt

NUM_CITIES = 10
GRID_SIZE = 200
DIST_FROM_BOUND = 30


def generate_cities(n, size=GRID_SIZE):
    """Generuje n losowych miast na planszy o rozmiarze GRID_SIZE"""
    cities = [(random.randint(0+DIST_FROM_BOUND, size-DIST_FROM_BOUND), random.randint(0+DIST_FROM_BOUND, size-DIST_FROM_BOUND)) for _ in range(n)]
    return cities

def plot_cities(cities):
    """Wyświetla miasta na wykresie"""
    x, y = zip(*cities)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='blue', s=40)
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.show()

def main():
    cities = generate_cities(NUM_CITIES)
    plot_cities(cities)
    print(cities)

if __name__ == '__main__':
    main()

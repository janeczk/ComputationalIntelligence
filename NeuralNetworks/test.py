import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

# --- Ustawienie stylu wykresu na bardziej profesjonalny ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- Ustawienie ziarna losowości dla powtarzalności wyników ---
np.random.seed(42)

# --- Inicjalizacja parametrów ---
ile_danych = 200  # Zwiększamy nieco ilość danych
ile_epok = 4000   # Dajemy modelom dużo czasu na naukę
a = 0
b = 12

# --- Generowanie DANYCH ---
X_dane = a + (b - a) * np.random.rand(ile_danych, 1)
y_dane = 0.5 * np.cos(0.2 * X_dane**2) + 0.5 + 0.1 * np.random.randn(ile_danych, 1)

# --- Normalizacja danych do przedziału [-1, 1] ---
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
X_dane_scaled = x_scaler.fit_transform(X_dane)
y_dane_scaled = y_scaler.fit_transform(y_dane.reshape(-1, 1))

# --- Podział danych na treningowe i testowe ---
X_train, X_test, y_train, y_test = train_test_split(
    X_dane_scaled, y_dane_scaled, test_size=0.2, random_state=42
)

# --- 1. ZNACZNIE ROZSZERZONA SIATKA PARAMETRÓW ---
param_grid = {
    'hidden_layer_sizes': [(60, 90, 60, 30)], #, (70, 100, 70)
    'activation': [ 'relu'], #'tanh',
    'learning_rate_init': [0.001], #0.01,, 0.0005
    'alpha': [0.0001],  # Parametr regularyzacji L2, zapobiega przeuczaniu , 0.001, 0.01
    'solver': ['adam']
}

# --- Konfiguracja Grid Search ---
mlp = MLPRegressor(random_state=1, max_iter=ile_epok,n_iter_no_change=50)
grid_search = GridSearchCV(
    estimator=mlp, param_grid=param_grid, cv=3,
    scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)

print("\n--- Uruchamiam rozszerzony GridSearchCV (to może potrwać)... ---")
start_time = time.time()
grid_search.fit(X_train, y_train.ravel())
end_time = time.time()
print(f"--- Zakończono! Czas trwania: {end_time - start_time:.2f} sekundy ---")


# --- 2. SZCZEGÓŁOWE WYNIKI GRID SEARCH ---
print("\n--- Najlepszy znaleziony model ---")
# Pobieramy najlepszy model
best_net = grid_search.best_estimator_

# Wyświetlamy jego parametry oraz liczbę epok, w której osiągnął zbieżność
print(f"Parametry: {grid_search.best_params_}")
print(f"Liczba iteracji do zbieżności: {best_net.n_iter_}")
print(f"Najlepszy wynik (ujemny MSE) z CV: {grid_search.best_score_:.6f}")


# --- Ocena na zbiorze testowym ---
y_pred_scaled = best_net.predict(X_test)
y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1))
final_mse = mean_squared_error(y_test_orig, y_pred_orig)
print(f"\nFinalny błąd MSE na zbiorze testowym: {final_mse:.6f}")


# --- 3. ULEPSZONY WYKRES ---
x_plot = np.arange(a, b, 0.05).reshape(-1, 1)
y_plot_true = 0.5 * np.cos(0.2 * x_plot**2) + 0.5
x_plot_scaled = x_scaler.transform(x_plot)
y_plot_pred_scaled = best_net.predict(x_plot_scaled)
y_plot_pred_orig = y_scaler.inverse_transform(y_plot_pred_scaled.reshape(-1, 1))

plt.figure(figsize=(16, 9))
ax = plt.gca() # Pobranie osi wykresu do dalszych modyfikacji

# Punkty danych jako tło
ax.scatter(X_dane, y_dane, color='skyblue', s=10, alpha=0.4, label='Dane (oryginalne)')

# Funkcja oryginalna jako solidna linia odniesienia
ax.plot(x_plot, y_plot_true, color='black', linewidth=3, linestyle='-', label='Funkcja oryginalna')

# Predykcja modelu jako wyróżniona linia
ax.plot(x_plot, y_plot_pred_orig, color='#d90429', linewidth=2.5, linestyle='--', label='Predykcja najlepszego modelu')

# Ustawienia estetyczne
ax.set_title(f'Wydajność Optymalnego Modelu (Finalne MSE: {final_mse:.4f})', fontsize=20, pad=20)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc='best', fontsize=12)
ax.set_ylim(-0.25, 1.25)
ax.set_facecolor('#f8f9fa') # Delikatne tło dla obszaru wykresu

plt.tight_layout() # Dopasowanie, aby nic nie zostało obcięte
plt.show()
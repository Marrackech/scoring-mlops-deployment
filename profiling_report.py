"""
Script de profiling — compare latence avant/après optimisation.
"""
import time
import json
import joblib
import numpy as np
import cProfile
import pstats
import io

print("Chargement des artefacts...")
model = joblib.load("model/lgbm_best.joblib")
scaler = joblib.load("model/scaler.joblib")

with open("model/features.json") as f:
    FEATURE_NAMES = json.load(f)

# Simulation input vide (cas nominal)
input_data = {}

# --- Méthode AVANT : DataFrame ---
import pandas as pd

def predict_avec_dataframe(model, scaler, input_data):
    row = {name: input_data.get(name, 0) for name in FEATURE_NAMES}
    df = pd.DataFrame([row], columns=FEATURE_NAMES).fillna(0)
    arr = scaler.transform(df)
    return model.predict_proba(arr)[0][1]

# --- Méthode APRÈS : numpy array ---
def predict_avec_numpy(model, scaler, input_data):
    arr = np.zeros(len(FEATURE_NAMES), dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return model.predict_proba(arr_scaled)[0][1]

# Benchmark
N = 100

print(f"\nBenchmark sur {N} itérations...")

# Avant
start = time.perf_counter()
for _ in range(N):
    predict_avec_dataframe(model, scaler, input_data)
t_avant = (time.perf_counter() - start) / N * 1000

# Après
start = time.perf_counter()
for _ in range(N):
    predict_avec_numpy(model, scaler, input_data)
t_apres = (time.perf_counter() - start) / N * 1000

gain = ((t_avant - t_apres) / t_avant) * 100

print(f"\n{'='*45}")
print(f"  Méthode DataFrame  : {t_avant:.3f} ms/requête")
print(f"  Méthode NumPy      : {t_apres:.3f} ms/requête")
print(f"  Gain               : {gain:.1f}%")
print(f"{'='*45}")

# Profiling détaillé
print("\nProfiling cProfile — méthode numpy :")
pr = cProfile.Profile()
pr.enable()
for _ in range(50):
    predict_avec_numpy(model, scaler, input_data)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())

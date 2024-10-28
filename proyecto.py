import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import time
import itertools
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_tpm(file_path):
    """
    Carga la Matriz de Transición de Probabilidades (TPM) desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        tuple: (TPM como ndarray, lista de estados en t, lista de estados en t+1)
    """
    df = pd.read_csv(file_path, index_col=0)
    states_t = list(df.index)
    states_t1 = list(df.columns)
    tpm = df.values
    return tpm, states_t, states_t1

def calculate_emd(distribution1, distribution2):
    """
    Calcula la Distancia del Migrador de Tierra (EMD) entre dos distribuciones.

    Args:
        distribution1 (ndarray): Primera distribución de probabilidad.
        distribution2 (ndarray): Segunda distribución de probabilidad.

    Returns:
        float: Valor de la EMD.
    """
    return wasserstein_distance(distribution1, distribution2)

def marginalize_tpm(tpm, candidate_indices, background_indices):
    """
    Marginaliza la TPM sobre los elementos de background.

    Args:
        tpm (ndarray): Matriz de transición completa.
        candidate_indices (list): Índices de los elementos seleccionados.
        background_indices (list): Índices de los elementos de background.

    Returns:
        ndarray: TPM marginalizada para el subconjunto seleccionado.
    """
    # Eliminar filas de background en t
    marginalized_tpm = np.delete(tpm, background_indices, axis=0)
    # Eliminar columnas de background en t+1
    marginalized_tpm = np.delete(marginalized_tpm, background_indices, axis=1)
    return marginalized_tpm

def get_distribution(tpm):
    """
    Obtiene la distribución de probabilidad a partir de la TPM.

    Args:
        tpm (ndarray): Matriz de transición.

    Returns:
        ndarray: Distribución de probabilidad normalizada.
    """
    distribution = tpm.sum(axis=0)
    total = distribution.sum()
    if total == 0:
        raise ValueError("La distribución de probabilidad no puede tener una suma de 0.")
    distribution /= total
    return distribution

def evaluate_divisions(V, W1, TPM, states_t, states_t1):
    """
    Evalúa todas las posibles divisiones dentro de W1 y retorna la que tiene el menor EMD.

    Args:
        V (list): Lista de índices de elementos.
        W1 (set): Conjunto de elementos seleccionados.
        TPM (ndarray): Matriz de transición completa.
        states_t (list): Lista de estados en t.
        states_t1 (list): Lista de estados en t+1.

    Returns:
        tuple: (Mejor división como tupla de conjuntos, EMD mínimo)
    """
    min_emd = float('inf')
    best_division = None

    # Generar todas las combinaciones posibles de divisiones dentro de W1 (desde 1 hasta len(W1)-1)
    for r in range(1, len(W1)):
        for subset in itertools.combinations(W1, r):
            subset_A = set(subset)
            subset_B = W1 - subset_A

            # Obtener índices para cada subconjunto
            subset_A_indices = list(subset_A)
            subset_B_indices = list(subset_B)

            # Marginalizar la TPM para cada subconjunto
            marginalized_tpm_A = marginalize_tpm(TPM, subset_A_indices, list(set(V) - subset_A))
            marginalized_tpm_B = marginalize_tpm(TPM, subset_B_indices, list(set(V) - subset_B))

            # Obtener distribuciones de probabilidad
            P_A = get_distribution(marginalized_tpm_A)
            P_B = get_distribution(marginalized_tpm_B)
            P_X = get_distribution(TPM)

            # Calcular EMD para cada subconjunto
            emd_A = calculate_emd(P_A, P_X)
            emd_B = calculate_emd(P_B, P_X)
            total_emd = emd_A + emd_B

            # Comparar con el mínimo actual
            if total_emd < min_emd:
                min_emd = total_emd
                best_division = (subset_A, subset_B)

    return best_division, min_emd

def divide_system_heuristic(V, TPM, states_t, states_t1):
    """
    Divide el sistema en dos subconjuntos utilizando una heurística mejorada.

    Args:
        V (list): Lista de índices de elementos.
        TPM (ndarray): Matriz de transición completa.
        states_t (list): Lista de estados en t.
        states_t1 (list): Lista de estados en t+1.

    Returns:
        tuple: (Mejor división como tupla de conjuntos, EMD mínimo)
    """
    W0 = set()
    W1 = set()

    # Inicializar W1 con el primer elemento
    W1.add(V[0])
    print(f"Inicializando W1 con el elemento: {V[0]}")

    for i in range(1, len(V)):
        next_element = find_min_emd(V, W1, TPM, states_t, states_t1)
        W1.add(next_element)
        print(f"Agregado elemento {next_element} a W1")

    # Evaluación final
    best_division, min_emd = evaluate_divisions(V, W1, TPM, states_t, states_t1)
    return best_division, min_emd

def find_min_emd(V, W1, TPM, states_t, states_t1):
    """
    Encuentra el siguiente elemento que minimiza la EMD al agregarlo a W1.

    Args:
        V (list): Lista de índices de elementos.
        W1 (set): Conjunto de elementos seleccionados.
        TPM (ndarray): Matriz de transición completa.
        states_t (list): Lista de estados en t.
        states_t1 (list): Lista de estados en t+1.

    Returns:
        int: Índice del mejor candidato para agregar.
    """
    min_emd = float('inf')
    best_candidate = None

    P_X = get_distribution(TPM)

    for vi in V:
        if vi not in W1:
            temp_W1 = W1.copy()
            temp_W1.add(vi)

            background = set(V) - temp_W1
            temp_W1_indices = list(temp_W1)
            background_indices = list(background)

            marginalized_tpm = marginalize_tpm(TPM, temp_W1_indices, background_indices)
            P_subset = get_distribution(marginalized_tpm)

            emd_value = calculate_emd(P_subset, P_X)
            print(f"Evaluando agregar elemento {vi}: EMD = {emd_value}")

            if emd_value < min_emd:
                min_emd = emd_value
                best_candidate = vi

    print(f"Mejor candidato para agregar: {best_candidate} con EMD = {min_emd}")
    return best_candidate

def brute_force_divide(V, TPM, states_t, states_t1):
    """
    Implementa una división por fuerza bruta para comparar resultados.

    Args:
        V (list): Lista de índices de elementos.
        TPM (ndarray): Matriz de transición completa.
        states_t (list): Lista de estados en t.
        states_t1 (list): Lista de estados en t+1.

    Returns:
        tuple: (Mejor división como tupla de conjuntos, EMD mínimo)
    """
    min_emd = float('inf')
    best_division = None

    # Generar todas las posibles divisiones (excluyendo el conjunto vacío y el conjunto completo)
    for r in range(1, len(V)):
        for subset in itertools.combinations(V, r):
            subset_A = set(subset)
            subset_B = set(V) - subset_A

            # Marginalizar la TPM para cada subconjunto
            marginalized_tpm_A = marginalize_tpm(TPM, list(subset_A), list(subset_B))
            marginalized_tpm_B = marginalize_tpm(TPM, list(subset_B), list(subset_A))

            # Obtener distribuciones de probabilidad
            P_A = get_distribution(marginalized_tpm_A)
            P_B = get_distribution(marginalized_tpm_B)
            P_X = get_distribution(TPM)

            # Calcular EMD para cada subconjunto
            emd_A = calculate_emd(P_A, P_X)
            emd_B = calculate_emd(P_B, P_X)
            total_emd = emd_A + emd_B

            # Comparar con el mínimo actual
            if total_emd < min_emd:
                min_emd = total_emd
                best_division = (subset_A, subset_B)

    return best_division, min_emd

def main():
    """
    Función principal que ejecuta el algoritmo.
    """
    # Abre ventana para seleccionar archivo
    Tk().withdraw()  # Oculta la ventana principal de tkinter
    file_path = askopenfilename(title="Selecciona el archivo TPM CSV", filetypes=[("CSV files", "*.csv")])

    if file_path:  # Si el usuario seleccionó un archivo
        TPM, states_t, states_t1 = load_tpm(file_path)
        # Aquí puedes continuar con la lógica que dependa de los datos cargados en TPM
        print("Archivo TPM cargado correctamente.")
    else:
        print("No se seleccionó ningún archivo.")

    # Cargar TPM y estados
    TPM, states_t, states_t1 = load_tpm(file_path)

    # Definir el sistema candidato
    V = list(range(len(states_t)))  # Índices de elementos en t

    # Medir tiempo de ejecución para el algoritmo heurístico mejorado
    start_time = time.time()

    # Dividir el sistema usando la heurística mejorada
    best_division_heuristic, min_emd_heuristic = divide_system_heuristic(V, TPM, states_t, states_t1)

    end_time = time.time()

    # Mostrar resultados heurísticos
    print("\nDivisión óptima del sistema (Heurístico Mejorado):")
    print(f"Subconjunto 1: {best_division_heuristic[0]}")
    print(f"Subconjunto 2: {best_division_heuristic[1]}")
    print(f"Diferencia EMD: {min_emd_heuristic}")
    print(f"Tiempo de ejecución: {end_time - start_time} segundos")

    # Medir tiempo de ejecución para fuerza bruta
    start_time_bf = time.time()

    # Dividir el sistema usando fuerza bruta
    best_division_bf, min_emd_bf = brute_force_divide(V, TPM, states_t, states_t1)

    end_time_bf = time.time()

    # Mostrar resultados de fuerza bruta
    print("\nDivisión óptima del sistema (Fuerza Bruta):")
    print(f"Subconjunto 1: {best_division_bf[0]}")
    print(f"Subconjunto 2: {best_division_bf[1]}")
    print(f"Diferencia EMD: {min_emd_bf}")
    print(f"Tiempo de ejecución: {end_time_bf - start_time_bf} segundos")

if __name__ == "__main__":
    main()

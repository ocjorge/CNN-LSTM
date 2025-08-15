# =============================================================================
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D,
                                     LSTM, Dense, Dropout, Flatten)
from tensorflow.keras.utils import to_categorical

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN Y PARÁMETROS GLOBALES
# =============================================================================
# --- Parámetros de datos ---
DATASET_PATH = 'ninapro_db1_data'  # <-- ¡RUTA A LA CARPETA CON LOS ARCHIVOS .MAT!
SUBJECT_ID = 1  # Sujeto a procesar (de 1 a 27)
EXERCISES_TO_PROCESS = [1, 2, 3]  # Ejercicios a incluir para el sujeto

# --- Parámetros de preprocesamiento ---
WINDOW_SIZE = 200  # Longitud de la ventana en muestras (aprox. 200ms si la frec. es 100Hz)
STEP = 50  # Desplazamiento de la ventana

# --- Parámetros del modelo y entrenamiento ---
NUM_CLASSES = 53  # NinaPro DB1 tiene 52 gestos + 1 de reposo
EPOCHS = 30
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# =============================================================================
# SECCIÓN 3: FUNCIONES DE CARGA Y PREPROCESAMIENTO
# =============================================================================
def load_ninapro_data(base_path, subject, exercises):
    """
    Carga los datos de sEMG y las etiquetas de los gestos para un sujeto
    y una lista de ejercicios desde archivos .mat.
    """
    all_emg = np.array([])
    all_gestures = np.array([])

    for exercise in exercises:
        file_path = os.path.join(base_path, f'S{subject}_E{exercise}_A1.mat')
        try:
            data = loadmat(file_path)
            # Concatena los datos de cada ejercicio
            if all_emg.size == 0:
                all_emg = data['emg']
                all_gestures = data['restimulus']
            else:
                all_emg = np.vstack((all_emg, data['emg']))
                all_gestures = np.vstack((all_gestures, data['restimulus']))
        except FileNotFoundError:
            print(f"¡ADVERTENCIA! No se encontró el archivo: {file_path}")
            print("Por favor, descarga los datos y colócalos en la carpeta correcta.")
            return None, None

    return all_emg, all_gestures


def create_windows(emg, gestures, window_size, step):
    """
    Crea ventanas deslizantes a partir de las señales sEMG.
    Cada ventana será una muestra para el entrenamiento.
    """
    X, y = [], []

    # El gesto 0 es de reposo, a menudo se excluye para entrenar solo en movimientos activos
    active_indices = np.where(gestures.flatten() != 0)[0]

    for i in range(0, len(active_indices) - window_size, step):
        # Obtener los índices de la ventana actual
        window_indices = active_indices[i: i + window_size]

        # Verificar que la ventana sea contigua
        if window_indices[-1] - window_indices[0] != window_size - 1:
            continue

        window_emg = emg[window_indices]

        # La etiqueta de la ventana es el gesto más frecuente
        window_gestures = gestures[window_indices]
        label = np.bincount(window_gestures.flatten()).argmax()

        # Normalización de la ventana (Z-score): crucial para un buen rendimiento
        mean = np.mean(window_emg, axis=0)
        std = np.std(window_emg, axis=0)
        window_normalized = (window_emg - mean) / (std + 1e-8)  # Se añade epsilon para evitar división por cero

        X.append(window_normalized)
        y.append(label)

    return np.array(X), np.array(y)


# =============================================================================
# SECCIÓN 4: FUNCIÓN PARA CONSTRUIR EL MODELO
# =============================================================================
def build_hybrid_cnn_lstm_model(input_shape, num_classes):
    """
    Construye un modelo híbrido CNN-LSTM para clasificación de sEMG.
    - CNN 1D para extraer características locales de las señales.
    - LSTM para aprender las dependencias temporales de esas características.
    """
    input_layer = Input(shape=input_shape)

    # Bloque Convolucional 1
    cnn = Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(input_layer)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.3)(cnn)

    # Bloque Convolucional 2
    cnn = Conv1D(filters=128, kernel_size=9, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.3)(cnn)

    # Capa Recurrente LSTM
    # La salida de la CNN (secuencia de características) se alimenta a la LSTM
    lstm = LSTM(128, return_sequences=False)(cnn)
    lstm = Dropout(0.4)(lstm)

    # Capa de Clasificación Final
    output_layer = Dense(num_classes, activation='softmax')(lstm)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Usar con etiquetas 'one-hot encoded'
                  metrics=['accuracy'])

    return model


# =============================================================================
# SECCIÓN 5: SCRIPT PRINCIPAL DE EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    # 1. Cargar datos
    print("Cargando datos...")
    emg_signals, gesture_labels = load_ninapro_data(DATASET_PATH, SUBJECT_ID, EXERCISES_TO_PROCESS)

    if emg_signals is not None:
        # 2. Preprocesar datos
        print("Preprocesando datos y creando ventanas...")
        X, y = create_windows(emg_signals, gesture_labels, WINDOW_SIZE, STEP)

        # Corregir etiquetas para que empiecen en 0 (si los gestos van de 1 a 52)
        y = y - 1

        # Convertir etiquetas a formato one-hot encoding
        y_categorical = to_categorical(y,
                                       num_classes=NUM_CLASSES - 1)  # Restamos 1 porque el gesto 0 (reposo) se excluyó

        # 3. Dividir en conjuntos de entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
        )

        print(f"Datos listos. Formas:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_val: {X_val.shape}")
        print(f"  - y_val: {y_val.shape}")

        # 4. Construir el modelo
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_hybrid_cnn_lstm_model(input_shape, num_classes=NUM_CLASSES - 1)
        model.summary()

        # 5. Entrenar el modelo
        print("\nIniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val)
        )

        # 6. Visualizar resultados
        plt.figure(figsize=(12, 5))

        # Gráfica de Precisión
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
        plt.title('Precisión del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()

        # Gráfica de Pérdida
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.title('Pérdida del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 7. Guardar el modelo entrenado
        MODEL_FILENAME = 'modelo_semg_codo_derecho.h5'
        model.save(MODEL_FILENAME)
        print(f"\nModelo entrenado guardado exitosamente como '{MODEL_FILENAME}'")

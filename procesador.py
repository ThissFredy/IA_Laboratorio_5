# procesador.py
from PIL import Image
import numpy as np
import os
import pandas as pd

# Dimensiones estándar de entrada
ANCHO_ESTANDAR = 400
ALTO_ESTANDAR = 184

def normalizar(datos, min_val=None, max_val=None):
    """Aplica normalización Min-Max a un array."""
    if min_val is None:
        min_val = np.min(datos)
    if max_val is None:
        max_val = np.max(datos)
    
    if max_val - min_val == 0:
        return np.zeros_like(datos), min_val, max_val
        
    return (datos - min_val) / (max_val - min_val), min_val, max_val

def procesar_datos_excel(ruta_excel, carpeta_imagenes, train_percent):
    """
    Procesa un archivo Excel para el entrenamiento de regresión.
    Divide los datos en train, validation y test.
    """
    try:
        df = pd.read_excel(ruta_excel)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo Excel. Error: {e}")

    if not all(col in df.columns for col in ["nombre", "px/cm", "longitud"]):
        raise ValueError("El Excel debe tener las columnas: 'nombre', 'px/cm', 'longitud'")

    vectores_x_img = []
    vectores_x_escala = []
    vectores_y_longitud = []
    errores_archivos = []

    print("Procesando filas del Excel...")
    for index, fila in df.iterrows():
        nombre_img = str(fila["nombre"])
        px_cm = float(fila["px/cm"])
        longitud = float(fila["longitud"])
        
        ruta_img = os.path.join(carpeta_imagenes, nombre_img)
        
        try:
            # ¡IMPORTANTE! El entrenamiento siempre usa la imagen en escala de grises
            # como lo definimos en el dataset (personalizada1, etc.)
            vector_img = procesar_imagen_a_vector(ruta_img)
            
            vectores_x_img.append(vector_img)
            vectores_x_escala.append(px_cm)
            vectores_y_longitud.append(longitud)
            
        except FileNotFoundError:
            errores_archivos.append(nombre_img)
        except Exception as e:
            raise ValueError(f"Error procesando la imagen {nombre_img}: {e}")

    if errores_archivos:
        raise FileNotFoundError(f"No se encontraron las siguientes imágenes en '{carpeta_imagenes}':\n" + 
                                f", ".join(errores_archivos))
    
    if not vectores_x_img:
        raise ValueError("No se procesó ninguna imagen exitosamente.")

    # --- Normalización ---
    X_img = np.array(vectores_x_img) / 255.0
    X_escala_norm, min_escala, max_escala = normalizar(np.array(vectores_x_escala))
    y_long_norm, min_longitud, max_longitud = normalizar(np.array(vectores_y_longitud))
    X_final = np.hstack((X_img, X_escala_norm.reshape(-1, 1)))
    y_final = y_long_norm.reshape(-1, 1)
    
    params_norm = {
        "min_escala": min_escala, "max_escala": max_escala,
        "min_longitud": min_longitud, "max_longitud": max_longitud
    }
    n_entradas = X_final.shape[1]
    n_salidas = y_final.shape[1]

    num_patrones = len(X_final)
    indices = np.arange(num_patrones)
    np.random.shuffle(indices)
    
    X_final = X_final[indices]
    y_final = y_final[indices]

    train_size = int(num_patrones * (train_percent / 100.0))
    remaining_size = num_patrones - train_size
    val_size = int(remaining_size / 2)
    
    X_train = X_final[0:train_size]
    y_train = y_final[0:train_size]
    X_val = X_final[train_size : train_size + val_size]
    y_val = y_final[train_size : train_size + val_size]
    X_test = X_final[train_size + val_size : ]
    y_test = y_final[train_size + val_size : ]
    
    print("División de datos completada.")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, \
           n_entradas, n_salidas, params_norm

def procesar_imagen_a_vector(ruta_o_pil):
    """
    Procesa una imagen al estándar de la RED NEURONAL (400x184, Grises).
    """
    try:
        if isinstance(ruta_o_pil, str):
            img = Image.open(ruta_o_pil)
        else:
            img = ruta_o_pil
    except Exception as e:
        raise IOError(f"No se pudo abrir la imagen: {e}")

    img_l = img.convert("L") 
            
    ancho, alto = img_l.size
    if alto > ancho:
        img_l = img_l.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
    
    img_base = img_l.resize((ANCHO_ESTANDAR, ALTO_ESTANDAR), Image.Resampling.BICUBIC)
    
    return np.array(img_base).flatten()
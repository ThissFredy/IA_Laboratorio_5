# filtros.py
import numpy as np

# --- KERNELS DE LA PRESENTACIÓN ---

KERNEL_DESENFOQUE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])
KERNEL_ENFOQUE = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
KERNEL_REALCE = np.array([
    [ 0,  0,  0],
    [ -1, 1,  0],
    [ 0,  0,  0]
])
KERNEL_REPUJADO = np.array([
    [ -2, -1,  0],
    [ -1,  1,  1],
    [  0,  1,  2]
])
KERNEL_DETECCION_BORDES = np.array([
    [ 0,  1, 0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])
KERNEL_SOBEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
KERNEL_SHARPEN = np.array([
    [ 1, -2, 1],
    [ -2, 5, -2],
    [ 1, -2, 1]
])
KERNEL_NORTE = np.array([
    [ 1,  1,  1],
    [ 1, -2,  1],
    [-1, -1, -1]
])
KERNEL_ESTE = np.array([
    [-1,  1,  1],
    [-1, -2,  1],
    [-1,  1,  1]
])
KERNEL_GAUSS = np.array([
    [1,  2,  3,  1, 1],
    [2, 7, 11, 7, 2],
    [3, 11, 17, 11, 3],
    [2, 7, 11, 7, 1],
    [1,  2,  3,  2, 1]
])

# Diccionario para acceder a los kernels por nombre
KERNELS = {
    "Desenfoque": KERNEL_DESENFOQUE,
    "Gauss (5x5)": KERNEL_GAUSS,
    "Enfoque": KERNEL_ENFOQUE,
    "Sharpen": KERNEL_SHARPEN,
    "Detección Bordes": KERNEL_DETECCION_BORDES,
    "Sobel": KERNEL_SOBEL,
    "Realce": KERNEL_REALCE,
    "Repujado": KERNEL_REPUJADO,
    "Filtro Norte": KERNEL_NORTE,
    "Filtro Este": KERNEL_ESTE,
}


def delete_red(arr_img_rgb):
    """Elimina el canal rojo de un array numpy RGB."""
    arr_out = arr_img_rgb.copy()
    arr_out[:, :, 0] = 0
    return arr_out

def delete_green(arr_img_rgb):
    """Elimina el canal verde de un array numpy RGB."""
    arr_out = arr_img_rgb.copy()
    arr_out[:, :, 1] = 0
    return arr_out

def delete_blue(arr_img_rgb):
    """Elimina el canal azul de un array numpy RGB."""
    arr_out = arr_img_rgb.copy()
    arr_out[:, :, 2] = 0
    return arr_out

def convertir_a_grises_math(arr_img_rgb):
    """
    Convierte un array RGB a escala de grises usando la fórmula de luminosidad.
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if arr_img_rgb.ndim != 3 or arr_img_rgb.shape[2] != 3:
        raise ValueError("La imagen de entrada debe ser un array RGB (3 canales).")
        
    R = arr_img_rgb[:, :, 0]
    G = arr_img_rgb[:, :, 1]
    B = arr_img_rgb[:, :, 2]
    
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    return gray.astype(np.uint8)

def aplicar_filtro_mediana_manual(arr_imagen, tamano_ventana=3):
    """
    Aplica un filtro de mediana de forma "manual" usando bucles y numpy.
    Funciona tanto para imágenes en escala de grises (2D) como RGB (3D).
    """
    pad = tamano_ventana // 2
    
    # Crear una imagen de salida vacía
    output_array = np.zeros_like(arr_imagen)
    
    if arr_imagen.ndim == 3:
        # --- Lógica para Imagen 3D (RGB) ---
        img_h, img_w, img_c = arr_imagen.shape
        # Aplicar padding (relleno) a los bordes
        img_padded = np.pad(arr_imagen, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        
        # Recorrer cada píxel de la imagen original
        for y in range(img_h):
            for x in range(img_w):
                # Recorrer cada canal (R, G, B)
                for i in range(img_c):
                    # Extraer la vecindad 3x3
                    region = img_padded[y : y + tamano_ventana, x : x + tamano_ventana, i]
                    # Calcular la mediana de los 9 píxeles y asignarla
                    output_array[y, x, i] = np.median(region)

    elif arr_imagen.ndim == 2:
        # --- Lógica para Imagen 2D (Grises) ---
        img_h, img_w = arr_imagen.shape
        # Aplicar padding (relleno) a los bordes
        img_padded = np.pad(arr_imagen, ((pad, pad), (pad, pad)), mode='edge')
        
        # Recorrer cada píxel
        for y in range(img_h):
            for x in range(img_w):
                # Extraer la vecindad 3x3
                region = img_padded[y : y + tamano_ventana, x : x + tamano_ventana]
                # Calcular la mediana y asignarla
                output_array[y, x] = np.median(region)
    
    else:
        raise ValueError("Dimensiones de array no soportadas, debe ser 2D o 3D.")
        
    return output_array.astype(np.uint8)

def aplicar_convolucion_matematica(arr_imagen, kernel, c_manual=None):
    """
    Aplica una convolución 2D de forma matemática.
    Maneja arrays 2D (Gris) y 3D (RGB).
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    if c_manual is not None:
        c = c_manual
    else:
        c = np.sum(kernel)
    if c == 0:
        c = 1

    if arr_imagen.ndim == 3:
        img_h, img_w, img_c = arr_imagen.shape
        arr_salida = np.zeros_like(arr_imagen, dtype=np.float64)
        pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
        img_padded = np.pad(arr_imagen, pad_width, mode='edge')

        for y in range(img_h):
            for x in range(img_w):
                for i in range(img_c):
                    region = img_padded[y : y + k_h, x : x + k_w, i]
                    valor_conv = np.sum(region * kernel) / c
                    arr_salida[y, x, i] = valor_conv

    elif arr_imagen.ndim == 2:
        img_h, img_w = arr_imagen.shape
        arr_salida = np.zeros_like(arr_imagen, dtype=np.float64)
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        img_padded = np.pad(arr_imagen, pad_width, mode='edge')

        for y in range(img_h):
            for x in range(img_w):
                region = img_padded[y : y + k_h, x : x + k_w]
                valor_conv = np.sum(region * kernel) / c
                arr_salida[y, x] = valor_conv
    else:
        raise ValueError(f"Dimensiones de array no soportadas: {arr_imagen.ndim}")

    arr_salida = np.clip(arr_salida, 0, 255)
    return arr_salida.astype(np.uint8)
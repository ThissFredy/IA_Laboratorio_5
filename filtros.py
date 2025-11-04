# filtros.py
import numpy as np
from PIL import Image, ImageFilter

# --- Definición de Kernels (Máscaras) de la Página 17 ---

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
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])
KERNEL_SOBEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
KERNEL_SHARPEN = np.array([
    [ 1, -2,  1],
    [-2,  5, -2],
    [ 1, -2,  1]
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
    [2,  7, 11,  7, 2],
    [3, 11, 17, 11, 3],
    [2,  7, 11,  7, 1],
    [1,  2,  3,  2, 1]
])

# Diccionario para acceder a los kernels por nombre
KERNELS = {
    "Desenfoque": KERNEL_DESENFOQUE,
    "Enfoque": KERNEL_ENFOQUE,
    "Realce": KERNEL_REALCE,
    "Repujado": KERNEL_REPUJADO,
    "Detección Bordes": KERNEL_DETECCION_BORDES,
    "Sobel": KERNEL_SOBEL,
    "Sharpen": KERNEL_SHARPEN,
    "Filtro Norte": KERNEL_NORTE,
    "Filtro Este": KERNEL_ESTE,
    "Gauss (5x5)": KERNEL_GAUSS,
}

def aplicar_filtro_mediana(imagen_pil, tamano_ventana=3):
    """Aplica un filtro de mediana a una imagen PIL."""
    return imagen_pil.filter(ImageFilter.MedianFilter(size=tamano_ventana))

def aplicar_convolucion_matematica(arr_imagen, kernel, c_manual=None):
    """
    Aplica una convolución 2D de forma matemática.
    Maneja arrays 2D (Gris) y 3D (RGB).
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Calcular la norma 'c'
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
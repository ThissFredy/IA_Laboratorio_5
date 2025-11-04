import os
import numpy as np
from PIL import Image

# --- Constantes de Configuración ---
ANCHO_DESEADO = 400
ALTO_DESEADO = 184
CARPETA_ORIGINALES = "Originales"
CARPETA_AUMENTADA = "Dataset"
TOTAL_IMAGENES = 30


# Kernel de Desenfoque (tipo "Box Blur")
# La norma 'c' es la suma de sus elementos (9)
KERNEL_DESENFOQUE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

# Kernel de Enfoque (Sharpen)
# La norma 'c' es la suma de sus elementos (1)
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
    [ -1, 1,  1],
    [ 0,  1,  2]
])

KERNEL_DETECCION_BORDES = np.array([
    [ 0,  1, 0],
    [ 1,  -4,  1],
    [0,  1,  0]
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
    [ 1,  -2,  1],
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

def _filtro_mediana(imagen_np_gray, tamano_ventana=3):
    pad = tamano_ventana // 2
    img_h, img_w = imagen_np_gray.shape
    output_array = np.zeros_like(imagen_np_gray)
    for y in range(pad, img_h - pad):
        for x in range(pad, img_w - pad):
            region = imagen_np_gray[y - pad : y + pad + 1, 
                                    x - pad : x + pad + 1]
            mediana = np.median(region)
            output_array[y, x] = mediana
    return output_array.astype('uint8')

def delete_red(arr_img):
    """
    Elimina el canal rojo de una imagen representada como un array numpy.
    """
    arr_img_sin_rojo = arr_img.copy()
    arr_img_sin_rojo[:, :, 0] = 0
    return arr_img_sin_rojo

def delete_green(arr_img):
    """
    Elimina el canal verde de una imagen representada como un array numpy.
    """
    arr_img_sin_verde = arr_img.copy()
    arr_img_sin_verde[:, :, 1] = 0
    return arr_img_sin_verde

def delete_azul(arr_img):
    """
    Elimina el canal azul de una imagen representada como un array numpy.
    """
    arr_img_sin_azul = arr_img.copy()
    arr_img_sin_azul[:, :, 2] = 0
    return arr_img_sin_azul

def aplicar_convolucion_matematica(arr_imagen, kernel):
    """
    Aplica una convolución 2D de forma matemática.
    --- CORREGIDA PARA MANEJAR 2D (Gris) y 3D (RGB) ---
    """
    # Obtener las dimensiones del kernel
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Calcular la norma 'c' (como en la diapositiva 15)
    c = np.sum(kernel)
    if c == 0:
        c = 1  # Evitar división por cero
            
    if arr_imagen.ndim == 3:
        # --- Lógica para Imagen 3D (RGB) ---
        img_h, img_w, img_c = arr_imagen.shape
        # Crear salida 3D
        arr_salida = np.zeros_like(arr_imagen, dtype=np.float64)
        # Padding 3D
        pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
        img_padded = np.pad(arr_imagen, pad_width, mode='edge')
        
        # Loop 3D
        for y in range(img_h):
            for x in range(img_w):
                for i in range(img_c):  # Aplicar a cada canal: R, G, B
                    region = img_padded[y : y + k_h, x : x + k_w, i]
                    valor_conv = np.sum(region * kernel) / c
                    arr_salida[y, x, i] = valor_conv

    elif arr_imagen.ndim == 2:
        # --- Lógica para Imagen 2D (Escala de Grises) ---
        img_h, img_w = arr_imagen.shape
        # Crear salida 2D
        arr_salida = np.zeros_like(arr_imagen, dtype=np.float64)
        # Padding 2D
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        img_padded = np.pad(arr_imagen, pad_width, mode='edge')
        
        # Loop 2D
        for y in range(img_h):
            for x in range(img_w):
                region = img_padded[y : y + k_h, x : x + k_w]
                valor_conv = np.sum(region * kernel) / c
                arr_salida[y, x] = valor_conv
    
    else:
        raise ValueError(f"Dimensiones de array no soportadas: {arr_imagen.ndim}")

    # Asegurar que los valores estén en el rango válido [0, 255]
    arr_salida = np.clip(arr_salida, 0, 255)
    
    # Convertir a tipo entero de 8 bits (formato de imagen)
    return arr_salida.astype(np.uint8)

def procesar_imagenes():
    """
    Función principal para leer, procesar y aumentar las imágenes.
    """
    if not os.path.exists(CARPETA_AUMENTADA):
        os.makedirs(CARPETA_AUMENTADA)
        print(f"Carpeta creada: {CARPETA_AUMENTADA}")


    print("Iniciando procesamiento de imágenes...")
    for i in range(1, TOTAL_IMAGENES + 1):
        nombre_archivo = f"{i}.jpg"
        ruta_archivo = os.path.join(CARPETA_ORIGINALES, nombre_archivo)

        img = Image.open(ruta_archivo)


        img = img.convert("RGB") 
            
        ancho, alto = img.size
        if alto > ancho:
            img = img.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
        
        img_base = img.resize((ANCHO_DESEADO, ALTO_DESEADO), Image.Resampling.BICUBIC)
        
        # img_base.save(os.path.join(CARPETA_AUMENTADA, f"{i}_base.jpg"))


        arr_base = np.array(img_base)

        # * 0. Grises
        img_gray = img_base.convert("L")
        img_gray.save(os.path.join(CARPETA_AUMENTADA, f"{i}_gray.jpg"))

        # A. Flip Horizontal
        img_flip = img_base.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip.save(os.path.join(CARPETA_AUMENTADA, f"{i}_flip.jpg"))

        # B. Desenfoque
        arr_blur = aplicar_convolucion_matematica(arr_base, KERNEL_DESENFOQUE)
        img_blur = Image.fromarray(arr_blur)
        img_blur.save(os.path.join(CARPETA_AUMENTADA, f"{i}_blur.jpg"))

        # C. Enfoque
        arr_sharp = aplicar_convolucion_matematica(arr_base, KERNEL_ENFOQUE)
        img_sharp = Image.fromarray(arr_sharp)
        img_sharp.save(os.path.join(CARPETA_AUMENTADA, f"{i}_sharp.jpg"))

        # D. Realce
        arr_realce = aplicar_convolucion_matematica(arr_base, KERNEL_REALCE)
        img_realce = Image.fromarray(arr_realce)
        img_realce.save(os.path.join(CARPETA_AUMENTADA, f"{i}_realce.jpg"))

        # E. Repujado
        arr_repujado = aplicar_convolucion_matematica(arr_base, KERNEL_REPUJADO)
        img_repujado = Image.fromarray(arr_repujado)
        img_repujado.save(os.path.join(CARPETA_AUMENTADA, f"{i}_repujado.jpg"))

        # F. Detección de Bordes
        arr_bordes = aplicar_convolucion_matematica(arr_base, KERNEL_DETECCION_BORDES)
        img_bordes = Image.fromarray(arr_bordes)
        img_bordes.save(os.path.join(CARPETA_AUMENTADA, f"{i}_bordes.jpg"))

        # D. Sobel
        arr_sobel = aplicar_convolucion_matematica(arr_base, KERNEL_SOBEL)
        img_sobel = Image.fromarray(arr_sobel)
        img_sobel.save(os.path.join(CARPETA_AUMENTADA, f"{i}_sobel.jpg"))

        # E. Sharpen
        arr_sharpen = aplicar_convolucion_matematica(arr_base, KERNEL_SHARPEN)
        img_sharpen = Image.fromarray(arr_sharpen)
        img_sharpen.save(os.path.join(CARPETA_AUMENTADA, f"{i}_sharpen.jpg"))

        # * F. Norte
        arr_norte = aplicar_convolucion_matematica(arr_base, KERNEL_NORTE)
        img_norte = Image.fromarray(arr_norte).convert("L")
        img_norte.save(os.path.join(CARPETA_AUMENTADA, f"{i}_norte.jpg"))

        # G. Este
        arr_este = aplicar_convolucion_matematica(arr_base, KERNEL_ESTE)
        img_este = Image.fromarray(arr_este)
        img_este.save(os.path.join(CARPETA_AUMENTADA, f"{i}_este.jpg"))

        # H. Gauss
        arr_gauss = aplicar_convolucion_matematica(arr_base, KERNEL_GAUSS)
        img_gauss = Image.fromarray(arr_gauss)
        img_gauss.save(os.path.join(CARPETA_AUMENTADA, f"{i}_gauss.jpg"))

        # * I. Personalizada
        arr_personalizada = aplicar_convolucion_matematica(arr_base, KERNEL_GAUSS)
        arr_personalizada = aplicar_convolucion_matematica(arr_base, KERNEL_NORTE)
        img_personalizada = Image.fromarray(arr_personalizada).convert("L")
        img_personalizada = Image.fromarray(arr_personalizada.astype(np.uint8))
        img_personalizada.save(os.path.join(CARPETA_AUMENTADA, f"{i}_personalizada.jpg"))

        # * J. Personalizada 2
        arr_personalizada_2 = aplicar_convolucion_matematica(arr_base, KERNEL_DESENFOQUE)
        img_personalizada_2 = Image.fromarray(arr_personalizada_2.astype(np.uint8)).convert("L")
        arr_personalizada_2 = np.array(img_personalizada_2)
        arr_personalizada_2 = _filtro_mediana(arr_personalizada_2, tamano_ventana=3)
        arr_personalizada_2 = aplicar_convolucion_matematica(arr_personalizada_2, KERNEL_DETECCION_BORDES)
        arr_personalizada_2 = (arr_personalizada_2 > 8) * 255
        img_personalizada_2 = Image.fromarray(arr_personalizada_2.astype(np.uint8))
        img_personalizada_2.save(os.path.join(CARPETA_AUMENTADA, f"{i}_personalizada2.jpg"))





        print(f"Procesada y aumentada: {nombre_archivo} -> 12 imágenes generadas.")

    print(f"¡Proceso completado! Se generaron {TOTAL_IMAGENES * 12} imágenes en '{CARPETA_AUMENTADA}'.")

# --- Ejecutar el programa ---
if __name__ == "__main__":
    # Asegúrate de que exista la carpeta de originales
    if not os.path.exists(CARPETA_ORIGINALES):
        os.makedirs(CARPETA_ORIGINALES)
        print(f"Se creó la carpeta '{CARPETA_ORIGINALES}'.")
        print("Por favor, agrega tus 30 imágenes (1.jpg, 2.jpg, ...) en esta carpeta y vuelve a ejecutar.")
    else:
        procesar_imagenes()
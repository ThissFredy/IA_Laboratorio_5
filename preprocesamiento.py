import os
import numpy as np
from PIL import Image

# --- Diccionario de Kernels (Copiado de tu código) ---
KERNELS = {
    'Enfoque': np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    'Desenfoque': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    'Realce_Bordes': np.array([
        [0, 0, 0],
        [-1, 1, 0],
        [0, 0, 0]
    ]),
    'Repujado': np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]),
    'Deteccion_Bordes': np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]),
    'Sobel': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    'Filtro_Gauss': np.array([
        [1, 2, 3, 1, 1],
        [2, 7, 11, 7, 2],
        [3, 11, 17, 11, 3],
        [2, 7, 11, 7, 2],
        [1, 2, 3, 2, 1]
    ])
}

# --- Función de Convolución Manual (Copiada de tu código) ---
def convolve_manual_rgb(imagen_np, kernel):
    """
    Aplica una convolución 2D manual a una imagen RGB (array de NumPy).
    Itera sobre cada canal de color (R, G, B) y aplica el kernel.
    """
    # Manejar el caso de una imagen en escala de grises
    if imagen_np.ndim == 2:
        # Convertir temporalmente a 3D para que el resto del código funcione
        imagen_np = np.stack([imagen_np] * 3, axis=-1)

    k_h, k_w = kernel.shape
    img_h, img_w, img_c = imagen_np.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    output_array = np.zeros_like(imagen_np, dtype=np.float64)

    for c in range(img_c):
        for y in range(pad_h, img_h - pad_h):
            for x in range(pad_w, img_w - pad_w):
                
                region = imagen_np[y - pad_h : y + pad_h + 1, 
                                   x - pad_w : x + pad_w + 1, 
                                   c]
                
                valor_conv = np.sum(region * kernel)
                output_array[y, x, c] = valor_conv
    
    return output_array

# --- Función Principal de Procesamiento ---
def procesar_y_guardar_imagen(ruta_original, carpeta_destino, kernel_nombre, kernel, ancho_deseado, alto_deseado):
    """
    Aplica la lógica completa de preprocesamiento a una sola imagen y la guarda.
    """
    try:
        # 1. Cargar la imagen
        img = Image.open(ruta_original)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # 2. Corregir Orientación (¡NUEVA LÓGICA!)
        # Si es vertical (alto > ancho), rotarla 90 grados
        ancho, alto = img.size
        if alto > ancho:
            # Usamos -90 (o 270) para rotar en sentido horario, 
            # que es lo más común para fotos verticales.
            # 'expand=True' ajusta el tamaño del lienzo a la imagen rotada.
            img = img.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)

        # 3. Escalar al tamaño objetivo
        # Ahora que la imagen es horizontal, la redimensionamos a 400x184
        img_escalada = img.resize((ancho_deseado, alto_deseado), Image.Resampling.BICUBIC)

        # 4. Aplicar Convolución (Kernel)
        img_np = np.array(img_escalada)
        img_filtrada_np = convolve_manual_rgb(img_np, kernel)

        # 5. Normalizar kernels de desenfoque/gauss
        if kernel_nombre in ['Desenfoque', 'Filtro_Gauss']:
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                img_filtrada_np = img_filtrada_np / kernel_sum
        
        # 6. Recortar valores y convertir de nuevo a PIL
        img_filtrada_np = np.clip(img_filtrada_np, 0, 255)
        img_filtrada_pil = Image.fromarray(img_filtrada_np.astype('uint8'))

        # 7. Guardar el archivo
        ruta_base, extension = os.path.splitext(os.path.basename(ruta_original))
        nuevo_nombre = f"{ruta_base}_{kernel_nombre}{extension}"
        ruta_guardado = os.path.join(carpeta_destino, nuevo_nombre)
        
        img_filtrada_pil.save(ruta_guardado)
        print(f"  Guardada como: {ruta_guardado}")

    except Exception as e:
        print(f"  ERROR al procesar {ruta_original}: {e}")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    
    CARPETA_ORIGINALES = "Originales"
    ANCHO_OBJETIVO = 400
    ALTO_OBJETIVO = 184

    # Bucle principal para crear el dataset
    while True:
        print("\n--- Creador de Dataset de Preprocesamiento ---")
        print("Kernels Disponibles:")
        for key in KERNELS.keys():
            print(f" - {key}")
        print("\nEscribe 'salir' para terminar el programa.")
        
        # 1. Preguntar qué kernel aplicar
        kernel_a_aplicar = input("Elige el nombre exacto del kernel que quieres aplicar: ")
        
        if kernel_a_aplicar.lower() == 'salir':
            break
            
        if kernel_a_aplicar not in KERNELS:
            print("Error: Kernel no válido. Inténtalo de nuevo.")
            continue
            
        kernel_seleccionado = KERNELS[kernel_a_aplicar]

        # 2. Preguntar dónde guardar
        carpeta_destino = input(f"Ingresa el nombre de la carpeta destino (ej: {kernel_a_aplicar}): ")

        if not carpeta_destino:
            print("Error: Debes ingresar un nombre para la carpeta destino.")
            continue

        # 3. Crear la carpeta de destino
        os.makedirs(carpeta_destino, exist_ok=True)
        print(f"\nProcesando imágenes con el kernel '{kernel_a_aplicar}'. Guardando en '{carpeta_destino}'...")

        # 4. Iterar y procesar todas las 30 imágenes
        for i in range(1, 31):
            nombre_archivo = f"{i}.jpg"
            ruta_imagen = os.path.join(CARPETA_ORIGINALES, nombre_archivo)
            
            if os.path.exists(ruta_imagen):
                print(f"Procesando: {nombre_archivo}...")
                procesar_y_guardar_imagen(
                    ruta_imagen, 
                    carpeta_destino, 
                    kernel_a_aplicar, 
                    kernel_seleccionado, 
                    ANCHO_OBJETIVO, 
                    ALTO_OBJETIVO
                )
            else:
                print(f"Advertencia: No se encontró {ruta_imagen}. Saltando...")
        
        print(f"\n--- Lote completado para '{kernel_a_aplicar}' ---")

    print("Proceso de preprocesamiento finalizado.")
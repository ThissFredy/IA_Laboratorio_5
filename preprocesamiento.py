from PIL import Image
import numpy as np
import os

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


def convolve_manual_rgb(imagen_np, kernel):
    """
    Aplica una convolución 2D manual a una imagen RGB (array de NumPy).
    Itera sobre cada canal de color (R, G, B) y aplica el kernel.
    """
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
                
                # [cite_start]Operación de convolución [cite: 212, 242]
                valor_conv = np.sum(region * kernel)
                
                output_array[y, x, c] = valor_conv
    
    return output_array


def aplicar_filtro_manual(ruta_imagen, nombre_kernel, carpeta_salida, ancho_deseado=400, alto_deseado=184):
        
    if nombre_kernel not in KERNELS:
        print(f"Error: El kernel '{nombre_kernel}' no está definido.")
        return

    try:
        imagen_original = Image.open(ruta_imagen)
    except Exception as e:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        print(f"Error: {e}")
        return

    if imagen_original.mode == 'RGBA':
        imagen_original = imagen_original.convert('RGB')
        
    # Tu lógica para redimensionar según la orientación
    if(imagen_original.size[1] < imagen_original.size[0]):
        imagen = imagen_original.resize((ancho_deseado, alto_deseado), Image.Resampling.BICUBIC)
    else:
        imagen = imagen_original.resize((alto_deseado, ancho_deseado), Image.Resampling.BICUBIC)

    imagen_np = np.array(imagen)
    kernel = KERNELS[nombre_kernel]
    imagen_filtrada_np = convolve_manual_rgb(imagen_np, kernel)

    if nombre_kernel in ['Desenfoque', 'Filtro_Gauss']:
        kernel_sum = np.sum(kernel)
        if kernel_sum != 0:
            imagen_filtrada_np = imagen_filtrada_np / kernel_sum
    
    imagen_filtrada_np = np.clip(imagen_filtrada_np, 0, 255)
    imagen_filtrada_pil = Image.fromarray(imagen_filtrada_np.astype('uint8'))

    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_base, extension = os.path.splitext(os.path.basename(ruta_imagen))
    nuevo_nombre_archivo = f"{ruta_base}_{nombre_kernel}{extension}"
    ruta_guardado = os.path.join(carpeta_salida, nuevo_nombre_archivo)

    try:
        imagen_filtrada_pil.save(ruta_guardado)
        print(f"¡Éxito! Imagen guardada como: {ruta_guardado}")
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return


if __name__ == "__main__":
    
    CARPETA_ORIGINALES = "Originales"

    # --- 1. PREGUNTAR POR EL KERNEL ---
    print("--- Kernels Disponibles ---")
    print(", ".join(KERNELS.keys()))
    
    kernel_a_aplicar = ""
    while kernel_a_aplicar not in KERNELS:
        kernel_a_aplicar = input("Elige el nombre exacto del kernel que quieres aplicar: ")
        if kernel_a_aplicar not in KERNELS:
            print("Error: Kernel no válido. Inténtalo de nuevo.")

    # --- 2. PREGUNTAR POR LA CARPETA DESTINO ---
    carpeta_destino = input("Ingresa el nombre de la carpeta destino (ej: Filtradas_Sobel): ")

    if not carpeta_destino:
        print("Error: Debes ingresar un nombre para la carpeta destino.")
    else:
        print(f"\nProcesando imágenes con el kernel '{kernel_a_aplicar}'. Guardando en '{carpeta_destino}'...")

        # --- 3. ITERAR Y PROCESAR IMÁGENES ---
        for i in range(1, 31):
            nombre_archivo = f"{i}.jpg"
            ruta_imagen = os.path.join(CARPETA_ORIGINALES, nombre_archivo)
            
            if os.path.exists(ruta_imagen):
                print(f"\nProcesando: {ruta_imagen}")
                aplicar_filtro_manual(ruta_imagen, kernel_a_aplicar, carpeta_destino)
            else:
                print(f"\nAdvertencia: No se encontró {ruta_imagen}. Saltando...")
        
        print("\n--- Proceso completado ---")
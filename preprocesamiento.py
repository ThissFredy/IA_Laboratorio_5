
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
    'Sharpen': np.array([   
        [0, -2, 0],
        [-2, 5, -2],
        [0, -2, 0]
    ]),
    'Filtro_norte': np.array([
        [1, 1, 1],
        [0, -2, 0],
        [-1, -1, -1]
    ]),  
    'Filtro_este': np.array([
        [-1, 0, 1],
        [-1, -2, 1],
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

# --- 2. FUNCIÓN PRINCIPAL PARA APLICAR EL FILTRO ---

def aplicar_filtro(ruta_imagen, nombre_kernel, carpeta_salida, ancho_deseado=800, alto_deseado=368):
        
    # Validar que el kernel existe
    if nombre_kernel not in KERNELS:
        print(f"Error: El kernel '{nombre_kernel}' no está definido.")
        print(f"Kernels disponibles: {list(KERNELS.keys())}")
        return

    imagen_original = Image.open(ruta_imagen)
    
    # Verificar si la imagen se cargó correctamente
    if imagen_original is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        print("Asegúrate de que la ruta es correcta y el archivo existe.")
        return

    
    # Usamos INTER_AREA que es eficiente para reducir el tamaño (encoger)
    imagen = imagen_original.resize((ancho_deseado, alto_deseado), Image.Resampling.BICUBIC)
    # -----------------------------------------------

    # Obtener el kernel del diccionario
    kernel = KERNELS[nombre_kernel]

    os.makedirs(carpeta_salida, exist_ok=True)

    # Aplicar el filtro (convolución 2D) a la imagen REDIMENSIONADA
    imagen_filtrada = cv2.filter2D(src=imagen, ddepth=-1, kernel=kernel)

    # --- 3. GENERAR EL NUEVO NOMBRE Y GUARDAR LA IMAGEN ---
    ruta_base, extension = os.path.splitext(ruta_imagen)
    nombre_base = os.path.basename(ruta_base)
    directorio = os.path.dirname(ruta_imagen)

    # Crear el nuevo nombre de archivo
    nuevo_nombre_archivo = f"{nombre_base}_{nombre_kernel}{extension}"
    ruta_guardado = os.path.join(directorio, nuevo_nombre_archivo)

    # Guardar la imagen filtrada
    try:
        cv2.imwrite(ruta_guardado, imagen_filtrada)
        print(f"¡Éxito! Imagen guardada como: {ruta_guardado}")
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return

    # --- 4. MOSTRAR LAS IMÁGENES ---
    
    # Mostramos la imagen ya redimensionada como la "Original"
    cv2.imshow(f'Imagen Original ({ancho_deseado}x{alto_deseado})', imagen)
    cv2.imshow(f'Filtro Aplicado: {nombre_kernel}', imagen_filtrada)
    
    # Esperar a que el usuario presione una tecla para cerrar las ventanas
    print("Presiona cualquier tecla para cerrar las ventanas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 5. BLOQUE DE EJECUCIÓN ---

if __name__ == "__main__":
    
    # --- ¡CONFIGURA ESTO! ---
    # 1. Pon el nombre de tu imagen aquí.
    MI_IMAGEN = "Originales/1.jpg"  
        
    # 2. Elige el nombre del kernel que quieres aplicar
    KERNEL_A_APLICAR = ""

    CARPETA_DESTINO = "Filtradas"

    # Llamar a la función principal
    aplicar_filtro(MI_IMAGEN, KERNEL_A_APLICAR)
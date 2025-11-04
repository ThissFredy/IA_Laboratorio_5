# interfaz.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import io
import pandas as pd
from PIL import Image, ImageTk, ImageFilter
import math 
import re 

from backpropagation import RedNeuronalBackpropagation
from procesador import procesar_datos_excel, procesar_imagen_a_vector
from filtros import KERNELS, aplicar_convolucion_matematica

class BackpropagationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RNA Longitud de Peces - Tratamiento de Imagenes")
        self.geometry("1200x900") 
        self.minsize(1000, 700)
        
        self.base_width = 400
        self.base_height = 184
        self.preview_width = 800
        self.preview_height = 368
        self.scale_factor = self.preview_width / self.base_width
        
        self.entrenamiento_activo = False
        self.red_neuronal = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        
        self.params_norm = {}
        self.img_preproc_original = None
        self.img_preproc_standard_original_rgb = None
        self.img_preproc_actual = None
        self.img_preproc_tk = None
        self.secuencia_filtros = []
        self.line_start = None
        self.current_line = None
        self.pixel_scale = None
        self.img_prediccion_tk = None

        self._crear_encabezado()
        
        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook, padding=10)
        self.tab_preprocesamiento = ttk.Frame(self.notebook, padding=10)
        self.tab_prediccion = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_entrenamiento, text=' 1. Entrenamiento ')
        self.notebook.add(self.tab_preprocesamiento, text=' 2. Preprocesamiento ')
        self.notebook.add(self.tab_prediccion, text=' 3. Predicción ')
        self.notebook.pack(expand=True, fill="both")

        self._crear_ui_entrenamiento()
        self._crear_ui_preprocesamiento()
        self._crear_ui_prediccion()
        
        self.notebook.tab(1, state="disabled")
        self.notebook.tab(2, state="disabled")

    # --- Funciones auxiliares de filtros ---
    def _delete_red(self, arr_img_rgb):
        arr_out = arr_img_rgb.copy()
        arr_out[:, :, 0] = 0
        return arr_out

    def _delete_green(self, arr_img_rgb):
        arr_out = arr_img_rgb.copy()
        arr_out[:, :, 1] = 0
        return arr_out

    def _delete_blue(self, arr_img_rgb):
        arr_out = arr_img_rgb.copy()
        arr_out[:, :, 2] = 0
        return arr_out

    def _crear_encabezado(self):
        header_frame = ttk.Frame(self, padding="5", style="Header.TFrame")
        header_frame.pack(side=tk.TOP, fill=tk.X)
        style = ttk.Style(self)
        style.configure("Header.TFrame", background="#f0f0f0")
        
        def resource_path(relative_path):
            try: base_path = sys._MEIPASS
            except Exception: base_path = os.path.abspath(".")
            return os.path.join(base_path, relative_path)
            
        try:
            ruta_logo = resource_path("logo.png")
            img_original = Image.open(ruta_logo)
            img_redimensionada = img_original.resize((60, 60), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(img_redimensionada)
            logo_label = ttk.Label(header_frame, image=self.logo, background="#f0f0f0")
            logo_label.grid(row=0, column=0, rowspan=5, padx=10, pady=2, sticky="ns")
        except Exception as e: 
            print(f"Error al cargar el logo (asegúrate de tener 'logo.png'): {e}")
            ttk.Label(header_frame, text="Logo\nUDEC", background="#f0f0f0").grid(row=0, column=0, rowspan=5, padx=10, pady=2, sticky="ns")
            
        ttk.Label(header_frame, text="Universidad de Cundinamarca", font=("Helvetica", 15, "bold"), background="#f0f0f0").grid(row=0, column=1, sticky="w")
        ttk.Label(header_frame, text="RNA Predicción de Peces - Tratamiento de Imagenes", font=("Helvetica", 11), background="#f0f0f0").grid(row=1, column=1, sticky="w")
        ttk.Label(header_frame, text="INTELIGENCIA ARTIFICIAL", font=("Helvetica", 11), background="#f0f0f0").grid(row=2, column=1, sticky="w")
        ttk.Label(header_frame, text="Profesor: ING. Jaime Eduardo Andrade", font=("Helvetica", 9), background="#f0f0f0").grid(row=3, column=1, sticky="w")
        ttk.Label(header_frame, text="Integrantes: Fredy Alejandro Zarate - Juan David Rodriguez Real", font=("Helvetica", 9), background="#f0f0f0").grid(row=4, column=1, sticky="w")
        
        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=5)

    def _crear_ui_entrenamiento(self):
        frame = self.tab_entrenamiento
        
        panel_config = ttk.LabelFrame(frame, text="Hiperparámetros", padding=10)
        panel_config.pack(fill=tk.X, pady=5)
        
        ttk.Label(panel_config, text="Tasa de Aprendizaje (α):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.campo_alpha = ttk.Entry(panel_config, width=10); self.campo_alpha.insert(0, "0.01"); self.campo_alpha.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(panel_config, text="Momento (β):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.campo_momento = ttk.Entry(panel_config, width=10); self.campo_momento.insert(0, "0.5"); self.campo_momento.grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(panel_config, text="Neuronas Capa Oculta:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.campo_neuronas_ocultas = ttk.Entry(panel_config, width=10); self.campo_neuronas_ocultas.insert(0, "5"); self.campo_neuronas_ocultas.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(panel_config, text="Precisión (Error Deseado):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.campo_precision = ttk.Entry(panel_config, width=10); self.campo_precision.insert(0, "0.01"); self.campo_precision.grid(row=1, column=3, padx=5, pady=5)
        ttk.Label(panel_config, text="Máx. Épocas:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.campo_epocas = ttk.Entry(panel_config, width=10); self.campo_epocas.insert(0, "300"); self.campo_epocas.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(panel_config, text="Entrenamiento (%):").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.campo_train_percent = ttk.Entry(panel_config, width=10); self.campo_train_percent.insert(0, "70"); self.campo_train_percent.grid(row=2, column=3, padx=5, pady=5)
        
        panel_datos = ttk.LabelFrame(frame, text="Datos de Entrenamiento", padding=10)
        panel_datos.pack(fill=tk.X, pady=5)
        ttk.Button(panel_datos, text="1. Cargar Excel de Datos (.xlsx)", command=self.cargar_excel_entrenamiento).pack(side=tk.LEFT, padx=5)
        self.etiqueta_excel = ttk.Label(panel_datos, text="No cargado.", foreground="red"); self.etiqueta_excel.pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_datos, text="2. Cargar Carpeta de Imágenes", command=self.cargar_carpeta_imagenes).pack(side=tk.LEFT, padx=5)
        self.etiqueta_carpeta_img = ttk.Label(panel_datos, text="No cargada.", foreground="red"); self.etiqueta_carpeta_img.pack(side=tk.LEFT, padx=5)
        self.boton_cargar_datos = ttk.Button(panel_datos, text="3. Procesar Datos", command=self.procesar_datos_entrenamiento, state=tk.DISABLED)
        self.boton_cargar_datos.pack(side=tk.LEFT, padx=20)
        self.etiqueta_datos = ttk.Label(panel_datos, text="", foreground="green"); self.etiqueta_datos.pack(side=tk.LEFT, padx=5)
        self.ruta_excel = ""
        self.ruta_carpeta_img = ""

        panel_control = ttk.Frame(frame, padding=10)
        panel_control.pack(fill=tk.X)
        self.boton_entrenar = ttk.Button(panel_control, text="▶ Iniciar Entrenamiento", command=self.iniciar_entrenamiento, state=tk.DISABLED); self.boton_entrenar.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.boton_parar = ttk.Button(panel_control, text="■ Parar", command=self.parar_entrenamiento, state=tk.DISABLED); self.boton_parar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.boton_exportar = ttk.Button(panel_control, text="Exportar Pesos", command=self.exportar_pesos, state=tk.DISABLED)
        self.boton_exportar.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.boton_probar_test = ttk.Button(panel_control, text="Probar (Test Set)", command=self.ejecutar_prueba_test, state=tk.DISABLED)
        self.boton_probar_test.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        contenedor_inferior = ttk.Frame(frame, padding=10)
        contenedor_inferior.pack(fill=tk.BOTH, expand=True)
        self.figura_lms = Figure(figsize=(6, 4), dpi=100)
        self.grafica_lms = self.figura_lms.add_subplot(111)
        self.lienzo_lms = FigureCanvasTkAgg(self.figura_lms, master=contenedor_inferior)
        self.lienzo_lms.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.texto_resultados = tk.Text(contenedor_inferior, width=35, height=15, state=tk.DISABLED, font=("Consolas", 10))
        self.texto_resultados.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

    # --- Funciones Pestaña 1 ---
    def cargar_excel_entrenamiento(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Excel", filetypes=[("Archivos Excel", "*.xlsx")])
        if not ruta: return
        self.ruta_excel = ruta
        self.etiqueta_excel.config(text=os.path.basename(ruta), foreground="green")
        self._verificar_estado_procesar()

    def cargar_carpeta_imagenes(self):
        ruta = filedialog.askdirectory(title="Seleccionar Carpeta con Imágenes")
        if not ruta: return
        self.ruta_carpeta_img = ruta
        self.etiqueta_carpeta_img.config(text=os.path.basename(ruta), foreground="green")
        self._verificar_estado_procesar()

    def _verificar_estado_procesar(self):
        if self.ruta_excel and self.ruta_carpeta_img:
            self.boton_cargar_datos.config(state=tk.NORMAL)
        else:
            self.boton_cargar_datos.config(state=tk.DISABLED)

    def procesar_datos_entrenamiento(self):
        try:
            train_percent = float(self.campo_train_percent.get())
            if not 10 < train_percent < 100:
                raise ValueError("El porcentaje de entrenamiento debe estar entre 10 y 99")

            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, \
            n_in, n_out, self.params_norm = \
                procesar_datos_excel(self.ruta_excel, self.ruta_carpeta_img, train_percent)
            
            total_patrones = len(self.X_train) + len(self.X_val) + len(self.X_test)
            
            info = (f"Cargados {total_patrones} patrones.\n"
                    f" - Entrenamiento: {len(self.X_train)} ({train_percent:.0f}%)\n"
                    f" - Validación: {len(self.X_val)} (~{(100-train_percent)/2:.0f}%)\n"
                    f" - Prueba: {len(self.X_test)} (~{(100-train_percent)/2:.0f}%)\n"
                    f" - {n_in} Neuronas Entrada\n"
                    f" - {n_out} Neurona Salida")
            
            self.etiqueta_datos.config(text=info, foreground="blue")
            self.boton_entrenar.config(state=tk.NORMAL)
            messagebox.showinfo("Éxito", info)

        except Exception as e:
            messagebox.showerror("Error al Procesar Datos", f"Ocurrió un error:\n{e}")
            self.etiqueta_datos.config(text="Error al procesar", foreground="red")
            self.boton_entrenar.config(state=tk.DISABLED)

    def iniciar_entrenamiento(self):
        try:
            alpha = float(self.campo_alpha.get())
            momento = float(self.campo_momento.get())
            neuronas_ocultas = int(self.campo_neuronas_ocultas.get())
            precision = float(self.campo_precision.get())
            epocas = int(self.campo_epocas.get())
        except ValueError: 
            messagebox.showerror("Error de Parámetros", "Los hiperparámetros deben ser números válidos."); return
        
        self.texto_resultados.config(state=tk.NORMAL); self.texto_resultados.delete("1.0", tk.END); self.texto_resultados.config(state=tk.DISABLED)
        self.grafica_lms.clear(); self.lienzo_lms.draw()
        
        n_entradas = self.X_train.shape[1]
        n_salidas = self.y_train.shape[1]
        
        self.red_neuronal = RedNeuronalBackpropagation(n_entradas, neuronas_ocultas, n_salidas)
        self.red_neuronal.min_escala = self.params_norm["min_escala"]
        self.red_neuronal.max_escala = self.params_norm["max_escala"]
        self.red_neuronal.min_longitud = self.params_norm["min_longitud"]
        self.red_neuronal.max_longitud = self.params_norm["max_longitud"]
        
        self.datos_lms = []
        generador = self.red_neuronal.entrenar(self.X_train, self.y_train, alpha, momento, epocas, precision)
        
        self.entrenamiento_activo = True
        self.boton_entrenar.config(state=tk.DISABLED); self.boton_parar.config(state=tk.NORMAL)
        self.boton_exportar.config(state=tk.DISABLED); self.boton_probar_test.config(state=tk.DISABLED)
        
        self.after(10, self.actualizar_paso_entrenamiento, generador)

    def parar_entrenamiento(self): 
        self.entrenamiento_activo = False
        self.finalizar_entrenamiento("Entrenamiento detenido por el usuario.", ejecutar_validacion=True)

    def actualizar_paso_entrenamiento(self, generador):
        if not self.entrenamiento_activo:
            self.finalizar_entrenamiento("Entrenamiento detenido por el usuario.")
            return
        try:
            epoca, mse = next(generador)
            self.datos_lms.append((epoca + 1, mse))
            if (epoca + 1) % 1 == 0 or epoca == 0:
                self.actualizar_grafica_lms()
                info = f"Época: {epoca + 1}\nMSE: {mse:.8f}\n"; 
                self.texto_resultados.config(state=tk.NORMAL); self.texto_resultados.delete("1.0", tk.END); self.texto_resultados.insert(tk.END, info); self.texto_resultados.config(state=tk.DISABLED)
            self.after(1, self.actualizar_paso_entrenamiento, generador)
        except StopIteration:
            if self.datos_lms:
                epoca_final, mse_final = self.datos_lms[-1]
                info = f"Época Final: {epoca_final}\nMSE Final: {mse_final:.8f}\n"; 
                self.texto_resultados.config(state=tk.NORMAL); self.texto_resultados.delete("1.0", tk.END); self.texto_resultados.insert(tk.END, info); self.texto_resultados.config(state=tk.DISABLED)
            
            mensaje_final = "Entrenamiento finalizado."
            if self.datos_lms and self.datos_lms[-1][1] <= float(self.campo_precision.get()):
                mensaje_final += "\n(Se alcanzó la precisión deseada)"
            
            self.finalizar_entrenamiento(mensaje_final, ejecutar_validacion=True)
            
        except Exception as e:
            messagebox.showerror("Error en Entrenamiento", f"Error: {e}")
            self.finalizar_entrenamiento(f"Entrenamiento fallido: {e}", ejecutar_validacion=False)

    def finalizar_entrenamiento(self, mensaje, ejecutar_validacion=False):
        messagebox.showinfo("Fin del Entrenamiento", mensaje)
        self.actualizar_grafica_lms()
        self.entrenamiento_activo = False
        self.boton_entrenar.config(state=tk.NORMAL)
        self.boton_parar.config(state=tk.DISABLED)
        self.boton_exportar.config(state=tk.NORMAL)
        
        if ejecutar_validacion:
            self.ejecutar_validacion()
            self.boton_probar_test.config(state=tk.NORMAL)
            
        self.notebook.tab(1, state="normal")
        self.notebook.tab(2, state="normal")
        messagebox.showinfo("Siguiente Paso", "La red ha sido entrenada. Ya puedes ir a la pestaña 'Preprocesamiento' para cargar una imagen.")

    def actualizar_grafica_lms(self):
        self.grafica_lms.clear()
        self.grafica_lms.set_title("Error Cuadrático Medio (MSE) vs. Época")
        self.grafica_lms.set_xlabel("Época")
        self.grafica_lms.set_ylabel("MSE")
        self.grafica_lms.set_yscale("log") 
        if self.datos_lms:
            epocas, errores = zip(*self.datos_lms)
            self.grafica_lms.plot(epocas, errores, linestyle='-')
        self.grafica_lms.grid(True); self.lienzo_lms.draw()

    def _calcular_confianza(self, X_set, y_set):
        """Calcula la confianza (100-MAPE) para un set de datos."""
        if X_set is None or len(X_set) == 0:
            return 0, 0

        pred_norm = self.red_neuronal.predecir(X_set)
        
        min_l = self.red_neuronal.min_longitud
        max_l = self.red_neuronal.max_longitud
        
        pred_cm = ((pred_norm * (max_l - min_l)) + min_l).ravel()
        real_cm = ((y_set * (max_l - min_l)) + min_l).ravel()
        
        epsilon = 1e-8
        mape = np.mean(np.abs((real_cm - pred_cm) / (real_cm + epsilon))) * 100
        
        confianza = 100.0 - mape
        return confianza, mape

    def ejecutar_validacion(self):
        """Se ejecuta automáticamente al final del entrenamiento."""
        confianza, mape = self._calcular_confianza(self.X_val, self.y_val)
        
        info = (f"\n--- Validación ({len(self.X_val)} patrones) ---\n"
                f"Confianza: {confianza:.2f}%\n"
                f"Error Promedio: {mape:.2f}%\n")
        
        self.texto_resultados.config(state=tk.NORMAL)
        self.texto_resultados.insert(tk.END, info)
        self.texto_resultados.config(state=tk.DISABLED)

    def ejecutar_prueba_test(self):
        """Se ejecuta al presionar el botón de Test."""
        confianza, mape = self._calcular_confianza(self.X_test, self.y_test)
        
        info = (f"--- Prueba Final ({len(self.X_test)} patrones) ---\n"
                f"Confianza: {confianza:.2f}%\n"
                f"Error Promedio: {mape:.2f}%\n\n"
                f"(Este es el resultado final con datos que la red nunca vio)")
        
        messagebox.showinfo("Resultado de la Prueba", info)

    def exportar_pesos(self):
        if not self.red_neuronal:
            messagebox.showwarning("Sin datos", "No hay una red entrenada para exportar.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de Texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if not ruta_archivo: return
        
        try:
            with open(ruta_archivo, 'w') as f:
                f.write("# --- Parametros Normalizacion ---\n")
                f.write(f"min_escala: {self.params_norm['min_escala']}\n")
                f.write(f"max_escala: {self.params_norm['max_escala']}\n")
                f.write(f"min_longitud: {self.params_norm['min_longitud']}\n")
                f.write(f"max_longitud: {self.params_norm['max_longitud']}\n")
                
                f.write("# --- Pesos Capa Oculta (w_ji) ---\n")
                np.savetxt(f, self.red_neuronal.w_ji)
                f.write("# --- Sesgos Capa Oculta (b_j) ---\n")
                np.savetxt(f, self.red_neuronal.b_j)
                f.write("# --- Pesos Capa Salida (w_kj) ---\n")
                np.savetxt(f, self.red_neuronal.w_kj)
                f.write("# --- Sesgos Capa Salida (b_k) ---\n")
                np.savetxt(f, self.red_neuronal.b_k)

            messagebox.showinfo("Éxito", f"Pesos y parámetros guardados en:\n{ruta_archivo}")
        except Exception as e:
            messagebox.showerror("Error al guardar", f"Ocurrió un error: {e}")
            
    # --- PESTAÑA 2: PREPROCESAMIENTO ---
    def _crear_ui_preprocesamiento(self):
        frame = self.tab_preprocesamiento
        panel_superior = ttk.Frame(frame)
        panel_superior.pack(fill=tk.X, pady=5)
        
        ttk.Button(panel_superior, text="Cargar Imagen para Preprocesar", command=self.preproc_cargar_imagen).pack(side=tk.LEFT, padx=5)
        self.label_nombre_img_preproc = ttk.Label(panel_superior, text="Ninguna imagen cargada.", foreground="red")
        self.label_nombre_img_preproc.pack(side=tk.LEFT, padx=5)

        panel_principal = ttk.Frame(frame)
        panel_principal.pack(fill=tk.BOTH, expand=True, pady=10)

        panel_filtros = ttk.LabelFrame(panel_principal, text="Controles de Filtros", padding=10)
        panel_filtros.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(panel_filtros, text="Aplicar Filtro Individual:").pack(anchor="w", pady=(0, 5))
        
        lista_filtros = [
            "Mediana",
            "Convertir a Grises",
            "Sin Rojo",
            "Sin Verde",
            "Sin Azul",
            "Segmentar (Binarizar)",
        ] + list(KERNELS.keys())
        
        self.combo_filtros = ttk.Combobox(panel_filtros, values=lista_filtros, state="readonly", width=27)
        self.combo_filtros.pack(fill=tk.X)
        self.combo_filtros.current(0)
        self.combo_filtros.bind("<<ComboboxSelected>>", self._on_filter_select)
        
        self.frame_segment_options = ttk.Frame(panel_filtros)
        ttk.Label(self.frame_segment_options, text="Umbral (0-255):").pack(side=tk.LEFT, padx=(0,5))
        self.campo_umbral = ttk.Entry(self.frame_segment_options, width=5); self.campo_umbral.insert(0, "128")
        self.campo_umbral.pack(side=tk.LEFT, padx=5)
        self.combo_segment_tipo = ttk.Combobox(self.frame_segment_options, values=["Pez Claro (> Umbral)", "Pez Oscuro (< Umbral)"], state="readonly", width=20)
        self.combo_segment_tipo.pack(side=tk.LEFT)
        self.combo_segment_tipo.current(0)
        
        ttk.Button(panel_filtros, text="Aplicar Filtro Seleccionado", command=self.preproc_aplicar_filtro).pack(fill=tk.X, pady=5)

        guia_texto = ("Nota: La red esta entrenada con imagenes segmentadas (\"Desenfoque - Grises - Mediana - Detección de Bordes - Binarización (8)\").")
        ttk.Label(panel_filtros, text=guia_texto, wraplength=180, justify="left", relief="sunken", padding=5).pack(pady=5)
        
        ttk.Separator(panel_filtros, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(panel_filtros, text="Secuencia de Filtros (Avanzado):").pack(anchor="w", pady=(0, 5))
        self.lista_secuencia = tk.Listbox(panel_filtros, height=5, width=27)
        self.lista_secuencia.pack(fill=tk.X, pady=5)
        ttk.Button(panel_filtros, text="Añadir Filtro a Secuencia ➔", command=self.preproc_anadir_secuencia).pack(fill=tk.X, pady=5)
        ttk.Button(panel_filtros, text="Ejecutar Secuencia Completa", command=self.preproc_ejecutar_secuencia).pack(fill=tk.X, pady=5)
        ttk.Button(panel_filtros, text="Limpiar Secuencia", command=self.preproc_limpiar_secuencia).pack(fill=tk.X, pady=10)
        ttk.Separator(panel_filtros, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Button(panel_filtros, text="Restaurar Imagen Original", command=self.preproc_restaurar_original).pack(fill=tk.X, pady=5)
        self.boton_ir_a_predecir = ttk.Button(panel_filtros, text="Usar esta imagen para Predecir ➔", command=self.preproc_ir_a_prediccion, state=tk.DISABLED)
        self.boton_ir_a_predecir.pack(fill=tk.X, pady=20)

        panel_preview = ttk.LabelFrame(panel_principal, text=f"Previsualización (Escalada a {self.preview_width}x{self.preview_height})", padding=10)
        panel_preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.label_preview_img = ttk.Label(panel_preview, text="Carga una imagen...", relief="sunken", anchor="center")
        self.label_preview_img.pack(fill=tk.BOTH, expand=True)

    def _on_filter_select(self, event=None):
        if self.combo_filtros.get() == "Segmentar (Binarizar)":
            self.frame_segment_options.pack(fill=tk.X, pady=5)
        else:
            self.frame_segment_options.pack_forget()

    def preproc_cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen", 
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        )
        if not ruta: return
        try:
            self.img_preproc_original = Image.open(ruta)
            w, h = self.img_preproc_original.size
            self.label_nombre_img_preproc.config(
                text=f"{os.path.basename(ruta)} ({w}x{h}px)", 
                foreground="green"
            )
            self.preproc_restaurar_original()
            self.boton_ir_a_predecir.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo cargar la imagen: {e}")
            self.label_nombre_img_preproc.config(text="Error al cargar", foreground="red")
            self.boton_ir_a_predecir.config(state=tk.DISABLED)

    def _standardize_image(self, img_pil_in):
        img_rgb = img_pil_in.convert("RGB") 
        ancho, alto = img_rgb.size
        if alto > ancho:
            img_rgb = img_rgb.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
        return img_rgb.resize((self.base_width, self.base_height), Image.Resampling.BICUBIC)

    def _update_preproc_preview(self, img_pil_base):
        self.img_preproc_actual = img_pil_base
        img_preview = img_pil_base.resize((self.preview_width, self.preview_height), Image.Resampling.NEAREST)
        self.img_preproc_tk = ImageTk.PhotoImage(img_preview)
        self.label_preview_img.config(image=self.img_preproc_tk, text="")

    def preproc_restaurar_original(self):
        if self.img_preproc_original:
            self.secuencia_filtros = []
            self.lista_secuencia.delete(0, tk.END)
            standard_img_rgb = self._standardize_image(self.img_preproc_original)
            self.img_preproc_standard_original_rgb = standard_img_rgb.copy()
            self._update_preproc_preview(standard_img_rgb)
            self._on_filter_select() 
        else:
            messagebox.showwarning("Sin Imagen", "Primero debes cargar una imagen.")

    def preproc_aplicar_filtro(self):
        if not self.img_preproc_standard_original_rgb:
            messagebox.showwarning("Sin Imagen", "Primero debes cargar una imagen."); return
            
        nombre_filtro = self.combo_filtros.get()
        if "---" in nombre_filtro: return 

        try:
            # Siempre empieza desde la original RGB estandarizada
            img_base_filtrar = self.img_preproc_standard_original_rgb.copy()
            arr_base = np.array(img_base_filtrar)
            img_resultado = None

            if nombre_filtro == "Mediana":
                # Filtro Mediana de PIL funciona en RGB
                img_resultado = img_base_filtrar.filter(ImageFilter.MedianFilter(size=3))
            
            elif nombre_filtro == "Convertir a Grises":
                img_resultado = img_base_filtrar.convert("L")
            
            elif nombre_filtro == "Sin Rojo":
                arr_filtrado = self._delete_red(arr_base)
                img_resultado = Image.fromarray(arr_filtrado)
            
            elif nombre_filtro == "Sin Verde":
                arr_filtrado = self._delete_green(arr_base)
                img_resultado = Image.fromarray(arr_filtrado)
            
            elif nombre_filtro == "Sin Azul":
                arr_filtrado = self._delete_blue(arr_base)
                img_resultado = Image.fromarray(arr_filtrado)
            
            elif nombre_filtro == "Segmentar (Binarizar)":
                try:
                    umbral = int(self.campo_umbral.get())
                    if not 0 <= umbral <= 255: raise ValueError
                except ValueError:
                    messagebox.showerror("Error", "El umbral debe ser un número entero entre 0 y 255.")
                    return
                
                tipo = self.combo_segment_tipo.get()
                img_base_gray = img_base_filtrar.convert("L")
                arr_base_gray = np.array(img_base_gray)
                
                # --- AQUÍ ESTABA EL BUG ---
                # Se aplicaba a 'arr_filtrado' (que era None) en lugar de 'arr_base_gray'
                if "Pez Claro" in tipo:
                    arr_filtrado = (arr_base_gray > umbral) * 255
                else: # Pez Oscuro
                    arr_filtrado = (arr_base_gray < umbral) * 255
                # --- FIN DE LA CORRECCIÓN ---
                
                img_resultado = Image.fromarray(arr_filtrado.astype(np.uint8))
            
            else:
                kernel = KERNELS.get(nombre_filtro)
                if kernel is not None:
                    arr_filtrado = aplicar_convolucion_matematica(arr_base, kernel) 
                    img_resultado = Image.fromarray(arr_filtrado)
                else:
                    messagebox.showerror("Error", "Filtro no reconocido."); return
            
            self._update_preproc_preview(img_resultado)
            
        except Exception as e:
            messagebox.showerror("Error de Filtro", f"No se pudo aplicar el filtro: {e}")

    def preproc_anadir_secuencia(self):
        nombre_filtro = self.combo_filtros.get()
        if "---" in nombre_filtro: return
        
        if nombre_filtro == "Segmentar (Binarizar)":
            try:
                umbral = int(self.campo_umbral.get())
                if not 0 <= umbral <= 255: raise ValueError
            except ValueError:
                messagebox.showerror("Error", "El umbral debe ser un número entero entre 0 y 255.")
                return
            tipo = self.combo_segment_tipo.get()
            
            filtro_guardado = f"Segmentar (U={umbral}, T={tipo})"
            self.secuencia_filtros.append(filtro_guardado)
            self.lista_secuencia.insert(tk.END, f"{len(self.secuencia_filtros)}. {filtro_guardado}")
        else:
            self.secuencia_filtros.append(nombre_filtro)
            self.lista_secuencia.insert(tk.END, f"{len(self.secuencia_filtros)}. {nombre_filtro}")

    def preproc_limpiar_secuencia(self):
        self.secuencia_filtros = []
        self.lista_secuencia.delete(0, tk.END)

    # --- CORRECCIÓN DE BUG: Secuencia de binarización ---
    def preproc_ejecutar_secuencia(self):
        if not self.img_preproc_original:
            messagebox.showwarning("Sin Imagen", "Primero debes cargar una imagen."); return
        
        img_temp_pil = self._standardize_image(self.img_preproc_original)

        try:
            for nombre_filtro in self.secuencia_filtros:
                arr_temp = np.array(img_temp_pil)
                
                if nombre_filtro == "Mediana":
                    img_temp_pil = img_temp_pil.filter(ImageFilter.MedianFilter(size=3))
                
                elif nombre_filtro == "Convertir a Grises":
                    img_temp_pil = img_temp_pil.convert("L")

                elif nombre_filtro in ["Sin Rojo", "Sin Verde", "Sin Azul"]:
                    if img_temp_pil.mode != "RGB":
                        raise ValueError(f"Filtro '{nombre_filtro}' solo se aplica a imágenes RGB.")
                    if nombre_filtro == "Sin Rojo": arr_filtrado = self._delete_red(arr_temp)
                    if nombre_filtro == "Sin Verde": arr_filtrado = self._delete_green(arr_temp)
                    if nombre_filtro == "Sin Azul": arr_filtrado = self._delete_blue(arr_temp)
                    img_temp_pil = Image.fromarray(arr_filtrado)

                elif nombre_filtro.startswith("Segmentar"):
                    try:
                        umbral = int(re.search(r'U=(\d+)', nombre_filtro).group(1))
                        tipo = re.search(r'T=(.+)\)', nombre_filtro).group(1)
                    except Exception as e:
                        raise ValueError(f"Error al leer parámetros de secuencia: {nombre_filtro}")

                    if img_temp_pil.mode == "RGB":
                        img_temp_pil = img_temp_pil.convert("L")
                    arr_temp_gray = np.array(img_temp_pil)
                    
                    # --- AQUÍ ESTABA EL BUG ---
                    # Faltaba la lógica 'if/else' para el tipo
                    if "Pez Claro" in tipo:
                        arr_filtrado = (arr_temp_gray > umbral) * 255
                    else:
                        arr_filtrado = (arr_temp_gray < umbral) * 255
                    # --- FIN DE LA CORRECCIÓN ---
                    
                    img_temp_pil = Image.fromarray(arr_filtrado.astype(np.uint8))
                
                else:
                    kernel = KERNELS.get(nombre_filtro)
                    if kernel is not None:
                        # Asegurarse de que el kernel se aplica al modo de imagen correcto
                        if img_temp_pil.mode == "L" and arr_temp.ndim == 3:
                            # Esto puede pasar si el filtro anterior era 'Convertir a Grises'
                            arr_temp = np.array(img_temp_pil)
                        
                        arr_filtrado = aplicar_convolucion_matematica(arr_temp, kernel)
                        img_temp_pil = Image.fromarray(arr_filtrado)
            
            self._update_preproc_preview(img_temp_pil)

        except Exception as e:
            messagebox.showerror("Error de Secuencia", f"Falló al aplicar la secuencia: {e}")

    def preproc_ir_a_prediccion(self):
        if not self.img_preproc_actual:
            messagebox.showwarning("Error", "No hay imagen procesada para enviar a predicción.")
            return
            
        if self.img_preproc_actual.mode != "L":
            messagebox.showerror("Error de Modo",
                                 "La imagen debe estar en ESCALA DE GRISES para la predicción.\n\n"
                                 "Por favor, aplique el filtro 'Convertir a Grises' o "
                                 "un filtro de 'Segmentar' antes de continuar.")
            return
        
        img_preview = self.img_preproc_actual.resize((self.preview_width, self.preview_height), Image.Resampling.NEAREST)
        self.img_prediccion_tk = ImageTk.PhotoImage(img_preview)
        
        center_x = self.preview_width // 2
        center_y = self.preview_height // 2
        
        self.canvas_prediccion.delete("all")
        self.canvas_prediccion.create_image(center_x, center_y, anchor="center", image=self.img_prediccion_tk)
        self.canvas_prediccion.image = self.img_prediccion_tk
        
        self.pixel_scale = None
        self.line_start = None
        self.label_escala_calculada.config(text="Escala: No calculada", foreground="red")
        self.label_resultado_prediccion.config(text="Dibuja una línea de 1cm y presiona 'Predecir'.", font=("Helvetica", 10), foreground="black")
        
        self.notebook.select(self.tab_prediccion)

    # --- PESTAÑA 3: PREDICCIÓN ---
    def _crear_ui_prediccion(self):
        frame = self.tab_prediccion
        
        panel_superior = ttk.LabelFrame(frame, text="Configuración de Predicción", padding=10)
        panel_superior.pack(fill=tk.X, pady=5)
        
        ttk.Button(panel_superior, text="Cargar Pesos Externos (.txt)", command=self.cargar_pesos_prediccion).pack(side=tk.LEFT, padx=5)
        self.label_pesos_prediccion = ttk.Label(panel_superior, text="Usando red entrenada en esta sesión.", foreground="blue")
        self.label_pesos_prediccion.pack(side=tk.LEFT, padx=10)

        panel_principal = ttk.Frame(frame)
        panel_principal.pack(fill=tk.BOTH, expand=True, pady=10)
        
        panel_preview = ttk.LabelFrame(panel_principal, text="Imagen a Predecir (Dibuja la escala de 1cm)", padding=10)
        panel_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas_prediccion = tk.Canvas(panel_preview, width=self.preview_width, height=self.preview_height, bg="#f0f0f0", relief="sunken")
        self.canvas_prediccion.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_prediccion.bind("<Button-1>", self.on_line_start)
        self.canvas_prediccion.bind("<B1-Motion>", self.on_line_draw)
        self.canvas_prediccion.bind("<ButtonRelease-1>", self.on_line_end)
        
        panel_control_pred = ttk.Frame(panel_principal, width=300)
        panel_control_pred.pack(side=tk.RIGHT, fill=tk.Y)
        
        panel_instrucciones = ttk.LabelFrame(panel_control_pred, text="Instrucciones", padding=10)
        panel_instrucciones.pack(fill=tk.X)
        
        ttk.Label(panel_instrucciones, text="1. Usa la cuadrícula de la imagen como guía.").pack(anchor="w")
        ttk.Label(panel_instrucciones, text="2. Haz clic y arrastra para dibujar una línea").pack(anchor="w")
        ttk.Label(panel_instrucciones, text="   que represente exactamente 1 cm.").pack(anchor="w")
        ttk.Label(panel_instrucciones, text="3. Presiona 'Predecir Longitud'.").pack(anchor="w", pady=(5,0))
        
        self.label_escala_calculada = ttk.Label(panel_instrucciones, text="Escala: No calculada", foreground="red", font=("Helvetica", 10, "bold"))
        self.label_escala_calculada.pack(pady=10)
        
        self.boton_predecir = ttk.Button(panel_control_pred, text="Predecir Longitud", command=self.ejecutar_prediccion)
        self.boton_predecir.pack(fill=tk.X, pady=20)
        
        panel_resultado_final = ttk.LabelFrame(panel_control_pred, text="Resultado", padding=10)
        panel_resultado_final.pack(fill=tk.BOTH, expand=True)
        
        self.label_resultado_prediccion = ttk.Label(panel_resultado_final, text="Esperando predicción...", font=("Helvetica", 14), anchor="center", justify="center")
        self.label_resultado_prediccion.pack(fill=tk.BOTH, expand=True)

    # --- Funciones de dibujo ---
    def on_line_start(self, event):
        self.canvas_prediccion.delete("linea_escala")
        self.line_start = (event.x, event.y)
        self.current_line = self.canvas_prediccion.create_line(self.line_start[0], self.line_start[1], event.x, event.y, fill="red", width=2, tags="linea_escala")

    def on_line_draw(self, event):
        if self.line_start:
            self.canvas_prediccion.delete(self.current_line)
            self.current_line = self.canvas_prediccion.create_line(self.line_start[0], self.line_start[1], event.x, event.y, fill="red", width=2, tags="linea_escala")

    def on_line_end(self, event):
        if self.line_start:
            x1_scaled, y1_scaled = self.line_start
            x2_scaled, y2_scaled = event.x, event.y
            distancia_scaled = math.sqrt((x2_scaled - x1_scaled)**2 + (y2_scaled - y1_scaled)**2)
            distancia_original = distancia_scaled / self.scale_factor
            self.pixel_scale = distancia_original
            self.line_start = None
            self.label_escala_calculada.config(text=f"Escala: {self.pixel_scale:.2f} px/cm (base {self.base_width}x{self.base_height})", foreground="green")

    # --- Cargar Pesos desde .TXT ---
    def cargar_pesos_prediccion(self):
        ruta_archivo = filedialog.askopenfilename(
            title="Cargar Pesos", 
            filetypes=[("Archivos de Texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if not ruta_archivo: return
        
        try:
            with open(ruta_archivo, 'r') as f:
                lineas = f.readlines()

            def extraer_params(lineas):
                params = {}
                seccion_params = False
                for linea in lineas:
                    if linea.startswith("# --- Parametros Normalizacion ---"):
                        seccion_params = True
                        continue
                    if linea.startswith("# ---"):
                        seccion_params = False
                    
                    if seccion_params and ":" in linea:
                        partes = linea.split(":")
                        key = partes[0].strip()
                        value = float(partes[1].strip())
                        params[key] = value
                
                if len(params) != 4:
                    raise ValueError("No se encontraron los 4 parámetros de normalización (min/max escala/longitud).")
                return params

            def extraer_array(nombre_bloque):
                try:
                    inicio = lineas.index(f"# --- {nombre_bloque} ---\n") + 1
                    fin = len(lineas) 
                    for i in range(inicio, len(lineas)):
                        if lineas[i].startswith("# ---"):
                            fin = i
                            break
                    bloque_texto = "".join(lineas[inicio:fin])
                    return np.loadtxt(io.StringIO(bloque_texto), ndmin=1)
                except (ValueError, IndexError):
                    raise ValueError(f"No se pudo encontrar o leer el bloque '{nombre_bloque}' en el archivo.")

            params_norm = extraer_params(lineas)
            w_ji = extraer_array("Pesos Capa Oculta (w_ji)")
            b_j = extraer_array("Sesgos Capa Oculta (b_j)")
            w_kj = extraer_array("Pesos Capa Salida (w_kj)")
            b_k = extraer_array("Sesgos Capa Salida (b_k)")
            
            if b_j.ndim == 0: b_j = np.array([b_j])
            if b_k.ndim == 0: b_k = np.array([b_k])
            
            n_in, n_oc = w_ji.shape
            n_out = w_kj.shape[1] if w_kj.ndim > 1 else 1

            self.red_neuronal = RedNeuronalBackpropagation(n_in, n_oc, n_out)
            self.red_neuronal.w_ji = w_ji
            self.red_neuronal.b_j = b_j
            self.red_neuronal.w_kj = w_kj
            self.red_neuronal.b_k = b_k
            
            self.red_neuronal.min_escala = params_norm['min_escala']
            self.red_neuronal.max_escala = params_norm['max_escala']
            self.red_neuronal.min_longitud = params_norm['min_longitud']
            self.red_neuronal.max_longitud = params_norm['max_longitud']
            
            self.label_pesos_prediccion.config(text=f"Pesos cargados: {os.path.basename(ruta_archivo)}", foreground="green")
            self.notebook.tab(1, state="normal"); self.notebook.tab(2, state="normal")
            messagebox.showinfo("Pesos Cargados", "Pesos externos y parámetros cargados. Ya puedes ir a 'Preprocesamiento'.")
            self.notebook.select(self.tab_preprocesamiento)
            
        except Exception as e:
            messagebox.showerror("Error al Cargar Pesos", f"No se pudieron cargar los pesos. El formato del TXT es incorrecto.\nError: {e}")

    # --- Ejecutar Predicción ---
    def ejecutar_prediccion(self):
        if not self.red_neuronal:
            messagebox.showwarning("Red no lista", "Primero debes entrenar una red o cargar pesos.")
            return
        if not self.img_preproc_actual:
            messagebox.showwarning("Sin Imagen", "Primero debes cargar y procesar una imagen en la Pestaña 2.")
            return
        
        if self.img_preproc_actual.mode != "L":
            messagebox.showerror("Error de Modo",
                                 "La imagen debe estar en ESCALA DE GRISES para la predicción.\n\n"
                                 "Aplica el filtro 'Convertir a Grises' o 'Segmentar' en la Pestaña 2.")
            return
        
        if self.pixel_scale is None or self.pixel_scale == 0:
            messagebox.showerror("Error de Entrada", "Debes dibujar una línea de 1cm en la imagen para definir la escala.")
            return
            
        px_cm_valor = self.pixel_scale

        try:
            arr_img = np.array(self.img_preproc_actual)
            vector_img = arr_img.flatten()
            vector_img_norm = vector_img / 255.0
            
            prediccion_final_cm = self.red_neuronal.predecir_cm(vector_img_norm, px_cm_valor)
            peso_gramos = 0.0141 * (prediccion_final_cm ** 2.95)
            
            resultado_texto = f"Resolución (Escala): {px_cm_valor:.2f} px/cm\n\n"
            resultado_texto += f"Longitud Predicha:\n{prediccion_final_cm:.2f} cm\n\n"
            resultado_texto += f"Peso Predicho:\n{peso_gramos:.2f} g\n\n"
            resultado_texto += f"Fórmula: P = 0.0141 * L^(2.95)"
            
            self.label_resultado_prediccion.config(text=resultado_texto, font=("Helvetica", 14, "bold"), foreground="black", justify="left")
            
        except Exception as e:
            messagebox.showerror("Error de Predicción", f"Ocurrió un error al predecir:\n{e}")
            self.label_resultado_prediccion.config(text="Error", foreground="red")
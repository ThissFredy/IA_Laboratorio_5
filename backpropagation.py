# backpropagation.py
import numpy as np

class RedNeuronalBackpropagation:
    def __init__(self, neuronas_entrada, neuronas_ocultas, neuronas_salida):
        # Inicialización de pesos y sesgos
        self.w_ji = np.random.uniform(size=(neuronas_entrada, neuronas_ocultas)) * 0.2 - 0.1
        self.b_j = np.zeros(neuronas_ocultas)
        self.w_kj = np.random.uniform(size=(neuronas_ocultas, neuronas_salida)) * 0.2 - 0.1
        self.b_k = np.zeros(neuronas_salida)

        # Variables para el término de momento
        self.cambio_anterior_w_ji = np.zeros_like(self.w_ji)
        self.cambio_anterior_w_kj = np.zeros_like(self.w_kj)
        self.cambio_anterior_b_j = np.zeros_like(self.b_j)
        self.cambio_anterior_b_k = np.zeros_like(self.b_k)

        # Parámetros de normalización (se establecerán durante el entrenamiento)
        self.min_longitud = 0
        self.max_longitud = 1
        self.min_escala = 0
        self.max_escala = 1

    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _derivada_sigmoide(self, x):
        return x * (1 - x)

    def _lineal(self, x):
        return x

    def _derivada_lineal(self, x):
        return 1

    def entrenar(self, X, y, alpha, momento, epocas, precision):
        epoca_actual = 0
        mse = float('inf')

        while epoca_actual < epocas and mse > precision:
            errores_epoca = []
            for x_patron, y_deseado in zip(X, y):
                
                # --- Propagación hacia Adelante (Forward Pass) ---
                
                # 1. De capa de entrada (i) a capa oculta (j)
                net_j = np.dot(x_patron, self.w_ji) + self.b_j
                y_j = self._sigmoide(net_j)

                # 2. De capa oculta (j) a capa de salida (k)
                net_k = np.dot(y_j, self.w_kj) + self.b_k
                salida_obtenida = self._lineal(net_k)
                
                # 1. Error y Delta en la capa de salida (k)
                error_salida = y_deseado - salida_obtenida
                delta_k = error_salida * self._derivada_lineal(salida_obtenida)
                
                # 2. Error y Delta en la capa oculta (j)
                error_oculta = delta_k.dot(self.w_kj.T)
                delta_j = error_oculta * self._derivada_sigmoide(y_j)

                # --- Actualización de Pesos y Sesgos con Momento ---
                
                # 1. Actualizar pesos y sesgo de la capa de salida (k)
                cambio_actual_w_kj = alpha * np.outer(y_j, delta_k) + momento * self.cambio_anterior_w_kj
                self.w_kj += cambio_actual_w_kj
                self.cambio_anterior_w_kj = cambio_actual_w_kj

                cambio_actual_b_k = alpha * delta_k + momento * self.cambio_anterior_b_k
                self.b_k += cambio_actual_b_k
                self.cambio_anterior_b_k = cambio_actual_b_k

                # 2. Actualizar pesos y sesgo de la capa oculta (j)
                cambio_actual_w_ji = alpha * np.outer(x_patron, delta_j) + momento * self.cambio_anterior_w_ji
                self.w_ji += cambio_actual_w_ji
                self.cambio_anterior_w_ji = cambio_actual_w_ji

                cambio_actual_b_j = alpha * delta_j + momento * self.cambio_anterior_b_j
                self.b_j += cambio_actual_b_j
                self.cambio_anterior_b_j = cambio_actual_b_j

                errores_epoca.append(np.sum(error_salida**2))
            
            mse = np.mean(errores_epoca)
            yield epoca_actual, mse
            epoca_actual += 1
            
    def predecir(self, x):
        """Predice un valor normalizado (0-1)"""
        entrada_neta_oculta = np.dot(x, self.w_ji) + self.b_j
        salida_oculta = self._sigmoide(entrada_neta_oculta)
        
        entrada_neta_salida = np.dot(salida_oculta, self.w_kj) + self.b_k
        salida_obtenida = self._lineal(entrada_neta_salida)
        return salida_obtenida

    def predecir_cm(self, x_img, x_escala):
        """
        Toma una imagen aplanada (0-1) y un valor de escala (real),
        los normaliza, predice y desnormaliza el resultado.
        """
        # Normalizar la escala
        x_escala_norm = (x_escala - self.min_escala) / (self.max_escala - self.min_escala)
        
        # Combinar en el vector de entrada final
        x_completo = np.append(x_img, [x_escala_norm])
        
        # Predecir
        prediccion_norm = self.predecir(x_completo)
        
        # Desnormalizar
        prediccion_cm = (prediccion_norm * (self.max_longitud - self.min_longitud)) + self.min_longitud
        return prediccion_cm[0]
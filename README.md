# TFG---Redes-neuronales-aplicadas-a-la-mecanica-de-fluidos

En este repositorio se encuentra el código escrito para el Trabajo de Fin de Grado "Redes neuronales aplicadas a la mecánica de fluidos.

El repositorio está estructurado en 5 carpetas:

**`pinn_burgers_fixed_lambda/`** contiene la PINN desarrollada para resolver la ecuación de Burgers autosimilar para un alpha fijo. En concreto contiene las 7 versiones desarrolladas sucesivamente y el código para graficar resultados. 

**`pinn_burgers_first_sol/`** contiene el modelo para la búsqueda de la primera solución suave de la ecuación de Burgers autosimilar.

**`pinn_burgers_second_sol/`** contiene el modelo para la búsqueda de la segunda solución suave de la ecuación de Burgers autosimilar.

**`pinn_de_gregorio_first_sol/`** contiene el modelo para la búsqueda de la primera solución de la ecuación de De Gregorio autosimilar.

**`pinn_de_gregorio_second_sol/`** contiene el modelo para la búsqueda de la segunda solución de la ecuación de De Gregorio autosimilar.

Dentro de las 4 últimas carpetas, la simulación se divide en distintos documentos de la siguiente manera: 
* **`models.py`**: Contiene la arquitectura de la red neuronal (*fully-connected*) y la capa de post-procesamiento encargada de forzar las condiciones de simetría analítica (imparidad) de los perfiles buscados.
* **`loss.py`**: Define la función de pérdida del sistema, calculando mediante diferenciación automática (`torch.autograd`) el residuo de la ecuación diferencial correspondiente (Burgers o De Gregorio) en los puntos de colocalización, junto con la penalización de las condiciones de contorno.
* **`utils.py`**: Reúne las funciones auxiliares de soporte matemático, como el cálculo de las soluciones exactas asintóticas para la validación del error y la rutina de generación de gráficos.
* **`main.py`**: Script principal que actúa como director de orquesta. Inicializa los hiperparámetros del modelo (malla de puntos, pesos de las pérdidas, número de épocas), gestiona los bucles de optimización (Adam, esquemas de *learning rate decay* y refinamiento mediante L-BFGS) y exporta los resultados finales.

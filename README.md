# TFG - Redes neuronales aplicadas a la mecánica de fluidos

En este repositorio se encuentra el código escrito para el Trabajo de Fin de Grado "Redes neuronales aplicadas a la mecánica de fluidos.

El repositorio está estructurado en 5 carpetas:

* **`pinn_burgers_fixed_lambda/`**: Contiene la PINN desarrollada para resolver la ecuación de Burgers autosimilar para un alpha fijo. En concreto contiene las 7 versiones desarrolladas sucesivamente y el código para graficar resultados. 
* **`pinn_burgers_first_sol/`**: Contiene el modelo para la búsqueda de la primera solución suave de la ecuación de Burgers autosimilar.
* **`pinn_burgers_second_sol/`**: Contiene el modelo para la búsqueda de la segunda solución suave de la ecuación de Burgers autosimilar.
* **`pinn_de_gregorio_first_sol/`**: Contiene el modelo para la búsqueda de la primera solución de la ecuación de De Gregorio autosimilar.
* **`pinn_de_gregorio_second_sol/`**: Contiene el modelo para la búsqueda de la segunda solución de la ecuación de De Gregorio autosimilar.

Dentro de las 4 últimas carpetas, la simulación se divide en distintos documentos de la siguiente manera: 
* **`config.py`**: Define los hiperparámetros de la simulación.
* **`losses.py`**: Evaluación de las distintas componentes de la función loss
* **`model.py`**: Crea la arquitectura de la red neuronal y impone la imparidad.
* **`plotting.py`**: Genera las gráficas para visualizar los resultados
* **`sweep.py`**: Script utilizado para realizar pruebas con distintos valores de parámetros.
* **`train.py`**: Entrenamiento de la red.

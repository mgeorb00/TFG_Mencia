# TFG Mencía George Bayón: Modelos de visión en la agricultura

## 1. Descripción
Este proyecto analiza la eficiencia de diferentes modelos de machine learning para predecir X, comparando resultados en un dataset pequeño (fase exploratoria) y aplicando los modelos seleccionados al dataset completo (fase principal).

## 2. Estructura del repositorio
El repositorio está dividido en dos partes: 
- `Experiments/Exploratory/`  
  Contiene la fase exploratoria del proyecto. Se realizan pruebas con distintos modelos y configuraciones sobre un dataset reducido, COCO128.

- `Experiments/Main/`  
  Incluye la fase principal, donde se entrenan los modelos seleccionados con los datasets elaborados.
  El modelo LeYOLO permite ser entrenado utilizando diferentes variantes de arquitectura (Small, Medium, etc.) modificando los archivos de configuración basados en formato `.yaml`. 

### Uso de variantes estándar y personalizadas:
Para lanzar un entrenamiento con una variante específica o utilizar la configuración personalizada desarrollada en este TFG, se debe apuntar al archivo `.yaml` correspondiente ubicado dentro del directorio de configuración de Ultralytics:

* **Variante Personalizada (Custom):** `Experiments/Main/LeYOLO/LeYOLO/ultralytics/cfg/cfg/leyolomedium_custom.yaml`
* **Otras Variantes:** Modificando el parámetro del modelo hacia el archivo `.yaml` de la arquitectura deseada.

-  `utils`
  Incluye los scripts utilizados para la generación de diferentes gráficas utilizadas en la memoria del TFG, también los scripts utilizados en el preprocesado de los conjuntos de datos. 


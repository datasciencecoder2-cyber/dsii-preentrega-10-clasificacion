# Data Science II — Pre-entrega 10

## Modelado y evaluación de clasificación: Churn

**Objetivo:** identificar clientes de una compañía de telecomunicaciones que podrían abandonar el servicio (clasificación binaria). Se implementa una **regresión logística**, división estratificada **Train/Test 80/20**, preprocesamiento seguro con `Pipeline` y `ColumnTransformer`, y evaluación con **Accuracy, Precision, Recall, F1-score y matriz de confusión**.

> **Transparencia sobre el origen:** la ejecución incluida usa **2.400 registros SIMULADOS**, generados con semilla fija. **No son datos reales** ni corresponden a IBM. Emulan las columnas de la muestra educativa **IBM Telco Customer Churn**, cuya referencia oficial figura abajo. Los puntajes no deben extrapolarse a un negocio real.

### Requisitos

- Python 3.11 o superior (probado con Python 3 en el entorno de generación; las librerías están disponibles para Python 3.14).
- Paquetes de `requirements.txt`: pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook, nbformat e ipykernel.
- No hace falta conectarse a Internet después de instalar dependencias.

### Pasos para ejecutar (Windows PowerShell)

1. Descargá o cloná el repositorio y abrí PowerShell en su carpeta raíz.
2. Instalá las dependencias:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Abrí el notebook (también podés abrirlo desde VS Code):

   ```powershell
   py -m jupyter notebook notebooks/modelado_metricas.ipynb
   ```

4. En Jupyter seleccioná **Run All / Ejecutar todas las celdas**. La versión incluida ya tiene las salidas calculadas para facilitar su revisión desde GitHub.
5. Como alternativa reproducible sin abrir el notebook, ejecutá el script CLI:

   ```powershell
   py ejecutar.py
   ```

6. Ejecutá las pruebas automatizadas:

   ```powershell
   py -m unittest discover -s tests -v
   ```

7. Para recrear el dataset simulado original: `py generar_datos.py` (sobrescribe el CSV con **la misma semilla**).

### Archivos principales

| Archivo / carpeta | Función |
|---|---|
| `notebooks/modelado_metricas.ipynb` | **Entregable principal:** notebook ejecutado, procedimiento, métricas, gráfico y conclusión en Markdown. |
| `datos/clientes_churn_simulados.csv` | Datos ficticios reproducibles (2.400 clientes). |
| `src/modelado.py` | Lectura, validación, construcción del pipeline y cálculo de métricas. |
| `generar_datos.py` | Reconstrucción determinista del dataset de prueba. |
| `ejecutar.py` | Ejecución alternativa, reportes y matriz de confusión exportada. |
| `evidencia/` | `metricas.json`, `reporte_clasificacion.txt`, `matriz_confusion.png`. |
| `tests/test_modelado.py` | **20 pruebas** sobre datos, split, prevención de fuga, notebook, métricas e inferencia. |
| `.gitignore` | Excluye cachés y archivos temporales de Python/Jupyter. |
| `descargar_ibm.py` | Descarga opcional del archivo IBM; no necesaria para reproducir esta entrega. |

### Metodología

1. **Adquisición y EDA:** verificar cantidad de registros, proporción de `Churn`, tipos y nulos. La variable objetivo es `Churn` (0 = permanece, 1 = abandona). El identificador `customerID` no se utiliza como predictor.
2. **Train/Test 80/20:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
3. **Preprocesamiento dentro de `Pipeline`:** mediana + `StandardScaler` para numéricas, moda + `OneHotEncoder(handle_unknown='ignore')` para categóricas. Ningún transformador se ajusta sobre el conjunto de prueba.
4. **Clasificador:** `LogisticRegression(max_iter=2000, class_weight='balanced')`.
5. **Evaluación:** comparación con un modelo baseline que siempre predice la clase mayoritaria; métricas sobre el **test**, matriz 2×2 y reflexión de negocio.

### Resultados verificados (muestra sintética)

| Indicador | Resultado |
|---|---:|
| Clientes totales | 2.400 |
| Train / Test | 1.920 / 480 |
| Accuracy | 67,92 % |
| Precision — Churn | 50,23 % |
| Recall — Churn | 71,61 % |
| F1-score — Churn | 59,04 % |
| Verdaderos negativos (TN) | 215 |
| Falsos positivos (FP) | 110 |
| Falsos negativos (FN) | 44 |
| Verdaderos positivos (TP) | 111 |

**Interpretación de negocio:** Recall es la métrica prioritaria en esta práctica porque los **falsos negativos** representan clientes que se van sin recibir una alerta de retención. Sin embargo, alertar de más también cuesta dinero: Precision y F1 ayudan a vigilar ese intercambio. La decisión óptima requiere costos reales de pérdida de cliente y de contacto, que este ejemplo no incluye.

La matriz de confusión se visualiza en el notebook y se exporta como `evidencia/matriz_confusion.png`. **No se presenta el rendimiento sintético como una estimación realista de producción.**

### Fuente externa y alternativa real (optativa)

- [IBM / Telco Customer Churn — archivo CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- [IBM — proyecto original de ejemplo](https://github.com/IBM/telco-customer-churn-on-icp4d)

El repositorio original fue archivado en 2024, pero es accesible en la fecha de preparación. El CSV de IBM posee 7.043 registros y 21 columnas; los nombres de campos son compatibles con este proyecto. **Para trabajar opcionalmente con datos de IBM**, ejecutá `py descargar_ibm.py` (requiere conexión); el script verifica un SHA-256 esperado. Luego, en la celda 2 del notebook, reemplazá la ruta `clientes_churn_simulados.csv` por `Telco-Customer-Churn-IBM.csv` y ejecutá todas las celdas. Los resultados serán distintos y tendrán que interpretarse nuevamente; no se puede utilizar el README sintético para reportar métricas del conjunto IBM.

### Límites del estudio

- Es un ejemplo educativo con datos sintéticos, **no validado para decisiones comerciales**.
- Dataset de un único corte, no longitudinal: el split aleatorio no prueba generalización temporal.
- No hay optimización de umbral sobre validación independiente ni estudio económico del costo FP/FN.
- No se realiza ninguna acción automática sobre clientes.

**Estado técnico:** notebook ejecutado de principio a fin, CLI y pruebas automatizadas disponibles.

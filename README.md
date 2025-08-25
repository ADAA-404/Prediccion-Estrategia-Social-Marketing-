# Prediccion-Estrategia-Social-Marketing 📈
Este proyecto de Data Science ofrece un análisis integral y una simulación predictiva para optimizar las estrategias de marketing digital. Utiliza un conjunto de datos de rendimiento de campañas para realizar un análisis exploratorio, construir un modelo predictivo y ejecutar una simulación de Monte Carlo para la asignación óptima de presupuesto.

El objetivo principal es responder a preguntas clave: ¿Cómo se han desempeñado nuestras campañas? ¿Qué factores impulsan las conversiones? y ¿Cómo podemos distribuir nuestro presupuesto para maximizar el retorno de la inversión (ROI)?

## Tecnologias usadas 🐍
- Pandas: Para la manipulación y análisis de datos en DataFrames.
- NumPy: Para operaciones numéricas eficientes, especialmente en la simulación.
- Scikit-learn: Para la construcción del modelo predictivo (pipelines y el algoritmo GradientBoostingRegressor).
- Matplotlib y Seaborn: Para la creación de visualizaciones y el análisis exploratorio de datos (EDA).
- Tabulate: Para la impresión de tablas con formato en la terminal.
- OS y Sys: Para el manejo de rutas de archivos y control de salida del script

## Consideraciones en Instalación ⚙️
- Clonar el repositorio (Bash):
git clone https://github.com/ADAA-404/Prediccion-Estrategia-Social-Marketing-.git
cd Prediccion-Estrategia-Social-Marketing-

- Crear un entorno virtual (opcional pero muy recomendado por la compatibilidad de librerias):
python -m venv venv
source venv/bin/activate  # En macOS/Linux
venv\Scripts\activate      # En Windows

- Instalar las dependencias:
pip install pandas numpy scikit-learn matplotlib seaborn tabulate

- Descargar los datos (utiliza un conjunto de datos disponible en Kaggle. Descarga los archivos CSV  y colócalos en una carpeta con el nombre [data] dentro de la raíz del proyecto.
https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance

- Configurar la ruta de los datos:
DATA_FOLDER_PATH = r"ruta/a/la/carpeta/data"

En esta ocasion el codigo se escribio en Jupyter Notebook para Python .

## Ejemplo de uso 📎
El script se ejecutará en secuencia, mostrando los resultados en la terminal y generando varias visualizaciones que te guiarán a través de todo el proceso de análisis.
Verás impresiones detalladas sobre la preparación de datos, un resumen del EDA y los resultados de la simulación de Monte Carlo, incluyendo el presupuesto óptimo y la distribución de las conversiones estimadas.
- Matriz de correlación: Para entender la relación entre las métricas numéricas.
- Rendimiento por plataforma y demografía: Gráficos de barras y boxplots que comparan CTR y conversiones.
- Análisis temporal: Visualizaciones de conversiones por día y hora.
- Distribución de Conversiones: Un histograma que muestra la distribución de las conversiones estimadas por el modelo en la simulación.
- Principio de Pareto (Regla 80/20): Un gráfico que identifica las campañas de alto rendimiento que generan la mayor parte de las conversiones.

## Contribuciones 🖨️
Si te interesa contribuir a este proyecto o usarlo independiente, considera:
- Hacer un "fork" del repositorio.
- Crear una nueva rama (git checkout -b feature/nueva-caracteristica).
- Realizar tus cambios y "commitearlos" (git commit -am 'Agregar nueva característica').
- Subir tus cambios a la rama (git push origin feature/nueva-caracteristica).
- Abrir un "Pull Request".

## Licencia 📜
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.


[English Version](README.en.md)

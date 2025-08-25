[Versión en Español](README.md)

# Prediction-strategy-Social-marketing 📈
This Data Science project provides a comprehensive analysis and predictive simulation to optimize digital marketing strategies. It uses a campaign performance dataset to perform exploratory data analysis, build a predictive model, and run a Monte Carlo simulation for optimal budget allocation.  

The main goal is to answer key questions: How have our campaigns performed? What factors drive conversions? And how can we distribute our budget to maximize return on investment (ROI)?  

## Technologies Used 🐍
- Pandas: For data manipulation and analysis in DataFrames.
- NumPy: For efficient numerical operations, especially in the simulation.
- Scikit-learn: For building the predictive model (pipelines and the GradientBoostingRegressor algorithm).
- Matplotlib and Seaborn: For creating visualizations and exploratory data analysis (EDA).
- Tabulate: For formatted table printing in the terminal.
- OS and Sys: For handling file paths and controlling script output. 

## Installation Notes ⚙️
- Clone the repository (Bash):  
git clone https://github.com/ADAA-404/Prediccion-Estrategia-Social-Marketing-.git   
cd Prediccion-Estrategia-Social-Marketing- 

- Create a virtual environment (optional but highly recommended for library compatibility):    
python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate      # On Windows  

- Instalar las dependencias:  
pip install pandas numpy scikit-learn matplotlib seaborn tabulate

- Download the data (the script uses a dataset available on Kaggle. Download the CSV files and place them in a folder named [data] inside the project's root directory):   
https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance

- Configure the data path:  
DATA_FOLDER_PATH = r"ruta/a/la/carpeta/data"

This code was written in a Python Jupyter Notebook.

## Example of Use 📎
The script will run in sequence, displaying results in the terminal and generating several visualizations to guide you through the entire analysis process.  
You will see detailed prints about data preparation, an EDA summary, and the results of the Monte Carlo simulation, including the optimal budget and the distribution of estimated conversions.
- Correlation Matrix: To understand the relationship between numerical metrics.  
- Platform and Demographic Performance: Bar plots and box plots comparing CTR and conversions.  
- Temporal Analysis: Visualizations of conversions by day and hour.  
- Conversions Distribution: A histogram showing the distribution of conversions estimated by the model in the simulation.  
- Pareto Principle (80/20 Rule): A chart that identifies the high-performing campaigns generating most of the conversions.  

## Contributions 🖨️
If you're interested in contributing to this project or using it independently, consider:   
- Forking the repository.
- Creating a new branch (git checkout -b feature/new-feature).
- Making your changes and committing them (git commit -am 'Add new feature').
- Pushing your changes to the branch (git push origin feature/new-feature).
- Opening a Pull Request.

## Licencia 📜
This project is under the MIT License. See the LICENSE file (if applicable) for more details.

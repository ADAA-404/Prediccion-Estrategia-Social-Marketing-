#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""
social_marketing_analytics.py

Este script realiza un análisis completo de datos de marketing digital,
incluyendo la limpieza de datos, el análisis exploratorio (EDA),
el desarrollo de un modelo predictivo para conversiones, un algoritmo
de optimización de presupuesto y la visualización de los resultados.


Este script unificado realiza un análisis completo de datos de marketing digital.
Incluye:
1. Carga, unión y limpieza de datos.
2. Análisis Exploratorio de Datos (EDA) con visualizaciones clave.
3. Construcción de un modelo predictivo con pipelines de Scikit-learn.
4. Simulación de Monte Carlo para optimizar la asignación de presupuesto.
5. Análisis y visualización de los resultados de la simulación.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer  # Importación corregida
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import os
import sys

# --- Constantes y configuración ---
# Ruta de la carpeta donde se encuentran los archivos CSV.
# MODIFICAR ESTA RUTA SEGÚN TU ENTORNO.
DATA_FOLDER_PATH = r"file path"

# Configuración de estilos para los gráficos.
plt.style.use('seaborn-v0_8-whitegrid')


def load_and_prepare_data(folder_path: str, avg_cpa: float = 5.0):
    """
    Carga, une y limpia los datasets de marketing.

    Args:
        folder_path (str): Ruta a la carpeta que contiene los archivos CSV.
        avg_cpa (float): Costo promedio por adquisición (CPA) para el cálculo de costos.

    Returns:
        tuple: DataFrame de rendimiento de campañas (pd.DataFrame) y el DataFrame combinado (pd.DataFrame).
    """
    try:
        ad_events_df = pd.read_csv(os.path.join(folder_path, 'ad_events.csv'))
        ads_df = pd.read_csv(os.path.join(folder_path, 'ads.csv'))
        campaigns_df = pd.read_csv(os.path.join(folder_path, 'campaigns.csv'))
        users_df = pd.read_csv(os.path.join(folder_path, 'users.csv'))
        print("✅ Datasets cargados correctamente.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Asegúrate de que los archivos CSV estén en la carpeta correcta.")
        sys.exit(1)

    # Unión de DataFrames
    df_combined = pd.merge(ads_df, campaigns_df, on='campaign_id', how='left')
    df_combined = pd.merge(ad_events_df, df_combined, on='ad_id', how='left')
    df_combined = pd.merge(df_combined, users_df, on='user_id', how='left', suffixes=('', '_user'))

    # Limpieza y preparación de datos
    df_combined.dropna(inplace=True)
    df_combined.drop_duplicates(inplace=True)
    df_combined['event_type'] = df_combined['event_type'].str.strip().str.lower()
    df_combined['cost'] = (df_combined['event_type'] == 'purchase') * avg_cpa
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])

    # Agregación a nivel de campaña
    campaign_performance = df_combined.groupby('campaign_id').agg(
        total_impressions=('event_type', lambda x: (x == 'impression').sum()),
        total_clicks=('event_type', lambda x: (x == 'click').sum()),
        total_conversions=('event_type', lambda x: (x == 'purchase').sum()),
        total_cost=('cost', 'sum'),
        ad_platform=('ad_platform', 'first'),
        ad_type=('ad_type', 'first'),
        target_gender=('target_gender', 'first'),
        target_age_group=('target_age_group', 'first'),
        total_budget=('total_budget', 'first'),
        duration_days=('duration_days', 'first')
    ).reset_index()

    # Cálculo de métricas derivadas
    campaign_performance['CTR'] = np.where(campaign_performance['total_impressions'] > 0,
                                         (campaign_performance['total_clicks'] / campaign_performance['total_impressions']) * 100, 0)
    campaign_performance['ROI_Proxy'] = np.where(campaign_performance['total_cost'] > 0,
                                               (campaign_performance['total_conversions'] * 10 - campaign_performance['total_cost']) / campaign_performance['total_cost'], 0)
    
    print("\n--- Vista preliminar del DataFrame de rendimiento de campañas ---")
    print(tabulate(campaign_performance.head(), headers='keys', tablefmt='psql'))
    print("\n--- Información general del DataFrame combinado ---")
    df_combined.info()

    return campaign_performance, df_combined


def run_exploratory_data_analysis(campaign_performance: pd.DataFrame, df_combined: pd.DataFrame):
    """
    Realiza el Análisis Exploratorio de Datos (EDA) y genera visualizaciones.
    """
    print("\n--- 🕵️ Análisis Exploratorio de Datos (EDA) ---")

    # EDA 1: Análisis por Plataforma Publicitaria
    plt.figure(figsize=(15, 5))
    plt.suptitle('Distribución de Métricas por Plataforma Publicitaria', fontsize=16)
    for i, metric in enumerate(['CTR', 'total_conversions', 'ROI_Proxy']):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x='ad_platform', y=metric, data=campaign_performance)
        plt.title(f'{metric} por Plataforma')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # EDA 2: Matriz de Correlación de características numéricas
    numeric_cols = campaign_performance.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(12, 10))
    sns.heatmap(campaign_performance[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlación entre Características Numéricas')
    plt.show()

    # EDA 3: Análisis de segmentación demográfica
    print("\n--- Análisis de Rendimiento por Segmento Demográfico ---")
    demographic_performance = df_combined.groupby(['target_gender', 'target_age_group']).agg(
        total_impressions=('event_type', lambda x: (x == 'impression').sum()),
        total_clicks=('event_type', lambda x: (x == 'click').sum()),
        total_conversions=('event_type', lambda x: (x == 'purchase').sum())
    ).reset_index()
    demographic_performance['CTR'] = np.where(demographic_performance['total_impressions'] > 0,
                                              (demographic_performance['total_clicks'] / demographic_performance['total_impressions']) * 100, 0)
    print("\nResumen del rendimiento por segmento demográfico:")
    print(tabulate(demographic_performance, headers='keys', tablefmt='psql'))
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='target_age_group', y='CTR', hue='target_gender', data=demographic_performance, palette='viridis')
    plt.title('CTR por Grupo de Edad y Género', fontsize=16)
    plt.xlabel('Grupo de Edad', fontsize=12)
    plt.ylabel('CTR (%)', fontsize=12)
    plt.legend(title='Género')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='target_age_group', y='total_conversions', hue='target_gender', data=demographic_performance, palette='magma')
    plt.title('Conversiones Totales por Grupo de Edad y Género', fontsize=16)
    plt.xlabel('Grupo de Edad', fontsize=12)
    plt.ylabel('Total de Conversiones', fontsize=12)
    plt.legend(title='Género')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # EDA 4: Análisis de tipo de anuncio
    print("\n--- Análisis de Rendimiento por Tipo de Anuncio ---")
    ad_type_performance = df_combined.groupby('ad_type').agg(
        total_impressions=('event_type', lambda x: (x == 'impression').sum()),
        total_clicks=('event_type', lambda x: (x == 'click').sum()),
        total_conversions=('event_type', lambda x: (x == 'purchase').sum())
    ).reset_index()
    ad_type_performance['CTR'] = np.where(ad_type_performance['total_impressions'] > 0,
                                          (ad_type_performance['total_clicks'] / ad_type_performance['total_impressions']) * 100, 0)
    print("\nResumen del rendimiento por tipo de anuncio:")
    print(tabulate(ad_type_performance, headers='keys', tablefmt='psql'))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ad_type', y='CTR', data=ad_type_performance, palette='Blues_d')
    plt.title('CTR por Tipo de Anuncio', fontsize=16)
    plt.xlabel('Tipo de Anuncio', fontsize=12)
    plt.ylabel('CTR (%)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ad_type', y='total_conversions', data=ad_type_performance, palette='Oranges_d')
    plt.title('Conversiones Totales por Tipo de Anuncio', fontsize=16)
    plt.xlabel('Tipo de Anuncio', fontsize=12)
    plt.ylabel('Total de Conversiones', fontsize=12)
    plt.tight_layout()
    plt.show()

    # EDA 5: Análisis temporal
    print("\n--- Análisis de la Distribución Temporal de Conversiones ---")
    df_combined['day_of_week'] = df_combined['timestamp'].dt.day_name()
    df_combined['hour'] = df_combined['timestamp'].dt.hour
    conversions_by_day = df_combined[df_combined['event_type'] == 'purchase'].groupby('day_of_week').size().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]).reset_index(name='total_conversions')
    conversions_by_hour = df_combined[df_combined['event_type'] == 'purchase'].groupby('hour').size().reset_index(name='total_conversions')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='day_of_week', y='total_conversions', data=conversions_by_day, palette='cividis')
    plt.title('Conversiones Totales por Día de la Semana', fontsize=16)
    plt.xlabel('Día de la Semana', fontsize=12)
    plt.ylabel('Total de Conversiones', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='total_conversions', data=conversions_by_hour, palette='viridis')
    plt.title('Conversiones Totales por Hora del Día (Todos los días)', fontsize=16)
    plt.xlabel('Hora del Día (24h)', fontsize=12)
    plt.ylabel('Total de Conversiones', fontsize=12)
    plt.tight_layout()
    plt.show()

    top_2_days = conversions_by_day.sort_values(by='total_conversions', ascending=False).head(2)['day_of_week'].tolist()
    df_top_days = df_combined[df_combined['day_of_week'].isin(top_2_days)]
    conversions_by_hour_top_days = df_top_days[df_top_days['event_type'] == 'purchase'].groupby(['day_of_week', 'hour']).size().reset_index(name='total_conversions')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='total_conversions', hue='day_of_week', data=conversions_by_hour_top_days, palette='plasma')
    plt.title(f'Conversiones por Hora en {top_2_days[0]} y {top_2_days[1]}', fontsize=16)
    plt.xlabel('Hora del Día (24h)', fontsize=12)
    plt.ylabel('Total de Conversiones', fontsize=12)
    plt.legend(title='Día de la Semana')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # EDA 6: Principio de Pareto (Regla 80/20)
    print("\n--- Analizando el Principio de Pareto (Regla 80/20) ---")
    campaign_performance_sorted = campaign_performance.sort_values(by='total_conversions', ascending=False)
    total_conversions = campaign_performance_sorted['total_conversions'].sum()
    campaign_performance_sorted['conversion_contribution'] = (campaign_performance_sorted['total_conversions'] / total_conversions) * 100
    campaign_performance_sorted['cumulative_contribution'] = campaign_performance_sorted['conversion_contribution'].cumsum()
    
    fig, ax1 = plt.subplots(figsize=(18, 10))
    sns.barplot(x=np.arange(len(campaign_performance_sorted)), y='conversion_contribution', data=campaign_performance_sorted, color='skyblue', ax=ax1, label='Contribución Individual')
    ax2 = ax1.twinx()
    sns.lineplot(x=np.arange(len(campaign_performance_sorted)), y='cumulative_contribution', data=campaign_performance_sorted, ax=ax2, color='red', marker='o', label='Contribución Acumulada')
    ax2.axhline(80, color='darkgreen', linestyle='--', linewidth=2, label='80% de Conversiones')
    
    pareto_index = campaign_performance_sorted['cumulative_contribution'].searchsorted(80, side='left')
    pareto_campaign_id = campaign_performance_sorted.iloc[pareto_index]['campaign_id']
    pareto_percentage = round((pareto_index + 1) / len(campaign_performance_sorted) * 100)
    
    ax1.axvline(x=pareto_index, color='orange', linestyle='--', linewidth=2, label=f'Punto 80/20 ({pareto_percentage}%)')
    ax2.text(x=pareto_index, y=85, s=f"Punto 80/20\nID: {pareto_campaign_id}\n{pareto_percentage}% de campañas", 
             color='black', ha='center', va='bottom', fontsize=11, 
             bbox=dict(facecolor='yellow', alpha=0.5))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax1.set_xlabel('Campaña (ordenada por rendimiento)', fontsize=12)
    ax1.set_ylabel('Contribución de Conversiones (%)', fontsize=12)
    ax2.set_ylabel('Contribución Acumulada (%)', color='red', fontsize=12)
    plt.title('Principio de Pareto: Contribución de Conversiones por Campaña', fontsize=16)
    plt.tight_layout()
    plt.show()

# --- Bloques Nuevos para el Modelo y la Simulación de Monte Carlo ---

def build_predictive_model(df_final: pd.DataFrame):
    """
    Construye y entrena el pipeline para el modelo predictivo.
    """
    print("\n--- 2. Preparación del Modelo Predictivo ---")

    features = ['ad_platform', 'ad_type', 'target_gender', 'target_age_group', 'total_budget', 'duration_days', 'CTR', 'total_impressions']
    target = 'total_conversions'

    X = df_final[features]
    y = df_final[target]

    categorical_features = ['ad_platform', 'ad_type', 'target_gender', 'target_age_group']
    numerical_features = ['total_budget', 'duration_days', 'CTR', 'total_impressions']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))])

    model_pipeline.fit(X, y)
    print("Modelo GradientBoostingRegressor entrenado correctamente.")
    return model_pipeline, features


def run_monte_carlo_simulation(model_pipeline, features: list, campaign_performance: pd.DataFrame):
    """
    Ejecuta la simulación de Monte Carlo.
    """
    print("\n--- 3. Simulación de Monte Carlo ---")
    
    num_simulations = 10000
    total_budget = 1000000

    unique_combinations = campaign_performance[['ad_platform', 'ad_type', 'target_gender', 'target_age_group', 'duration_days', 'CTR', 'total_impressions']].drop_duplicates().reset_index(drop=True)
    
    simulation_results = []

    for i in range(num_simulations):
        budgets = np.random.dirichlet(np.ones(len(unique_combinations)), size=1) * total_budget
        
        scenario_df = unique_combinations.copy()
        scenario_df['total_budget'] = budgets.flatten()
        
        predicted_conversions = model_pipeline.predict(scenario_df[features])
        total_predicted_conversions = np.sum(predicted_conversions)
        
        simulation_results.append({
            'scenario': i,
            'total_conversions': total_predicted_conversions,
            'budget_allocation': scenario_df
        })

    print(f"Se completaron {num_simulations} simulaciones.")
    return simulation_results


def analyze_and_visualize_results(simulation_results: list):
    """
    Analiza y visualiza los resultados de la simulación.
    """
    print("\n--- 4. Análisis de los Resultados de Monte Carlo ---")
    
    results_df = pd.DataFrame([{'total_conversions': res['total_conversions']} for res in simulation_results])
    best_scenario_index = results_df['total_conversions'].idxmax()
    best_scenario = simulation_results[best_scenario_index]

    print(f"\nResultados del escenario óptimo (Simulación #{best_scenario['scenario']}):")
    print(f"Total de conversiones estimadas: {round(best_scenario['total_conversions'])}")
    print("\nAsignación de presupuesto optimizada:")
    print(tabulate(best_scenario['budget_allocation'][['ad_platform', 'ad_type', 'total_budget']], headers='keys', tablefmt='psql'))

    print("\n--- Interpretación de los Resultados ---")
    top_3_budgets = best_scenario['budget_allocation'].sort_values(by='total_budget', ascending=False).head(3)
    print(f"**Novedad 1: Asignación de Presupuesto Óptima**")
    for index, row in top_3_budgets.iterrows():
        print(f"- **{row['ad_platform']}_{row['ad_type']}** ({row['target_gender']}_{row['target_age_group']}) tiene el mayor presupuesto: ${row['total_budget']:.2f}")

    avg_conversions = results_df['total_conversions'].mean()
    max_conversions = results_df['total_conversions'].max()
    print(f"\n**Novedad 2: Distribución de Conversiones**")
    print(f"- El promedio de conversiones estimadas es de: {round(avg_conversions)}")
    print(f"- El potencial máximo de conversiones (punto óptimo) es de: {round(max_conversions)}")

    print("\n--- 5. Visualización de los Resultados de Monte Carlo ---")
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['total_conversions'], bins=50, kde=True, color='skyblue')
    plt.title('Distribución de Conversiones Estimadas (Simulación Monte Carlo)', fontsize=16)
    plt.xlabel('Conversiones Totales Estimadas', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.axvline(best_scenario['total_conversions'], color='red', linestyle='--', label='Máximas Conversiones')
    plt.legend()
    plt.tight_layout()
    plt.show()

    random_combination = best_scenario['budget_allocation'].sort_values(by='total_budget', ascending=False).iloc[0]
    combo_name = f"{random_combination['ad_platform']}_{random_combination['ad_type']}"
    
    scatter_data = []
    for res in simulation_results:
        budget_for_combo = res['budget_allocation'][
            (res['budget_allocation']['ad_platform'] == random_combination['ad_platform']) &
            (res['budget_allocation']['ad_type'] == random_combination['ad_type'])
        ]['total_budget'].iloc[0]
        
        conversions = res['total_conversions']
        scatter_data.append({'budget': budget_for_combo, 'conversions': conversions})

    scatter_df = pd.DataFrame(scatter_data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='budget', y='conversions', data=scatter_df, alpha=0.3)
    plt.title(f'Relación entre Presupuesto y Conversiones (Simulación de {combo_name})', fontsize=16)
    plt.xlabel(f'Presupuesto para {combo_name}', fontsize=12)
    plt.ylabel('Conversiones Totales Estimadas', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\n**Novedad 3: Concentración del Presupuesto en la Simulación de {combo_name}**")
    bins = np.linspace(0, scatter_df['budget'].max(), 11)
    budget_ranges = pd.cut(scatter_df['budget'], bins=bins).value_counts().sort_index()
    best_range = budget_ranges.idxmax()
    print(f"- El rango de presupuesto con la mayor concentración de puntos (escenarios) es: {best_range}")

    print("\nEl análisis de Monte Carlo se ha completado, se han interpretado las novedades clave de la simulación.")


if __name__ == "__main__":
    print("--- 1. Carga y Preparación de Datos ---")
    campaign_performance_df, combined_df = load_and_prepare_data(DATA_FOLDER_PATH)

    # 2. Realizar el Análisis Exploratorio de Datos
    run_exploratory_data_analysis(campaign_performance_df, combined_df)

    # 3. Desarrollar el modelo predictivo
    model_pipeline, features = build_predictive_model(campaign_performance_df)

    # 4. Ejecutar la simulación de Monte Carlo
    simulation_results = run_monte_carlo_simulation(model_pipeline, features, campaign_performance_df)

    # 5. Analizar y visualizar los resultados
    analyze_and_visualize_results(simulation_results)


# In[ ]:





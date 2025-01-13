import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import make_scorer, f1_score
import joblib

def load_config(ruta_config):
    """
    Lee un archivo JSON con la configuración.
    """
    with open(ruta_config, 'r') as config_file:
        return json.load(config_file)

def load_csv(ruta, columnas):
    """
    Carga un archivo CSV y selecciona columnas específicas.
    """
    df = pd.read_csv(ruta)
    return df[columnas]

def cruzar_df(df1, df2, llave):
    """
    Cruza dos DataFrames utilizando una llave común.
    """
    return pd.merge(df1, df2, on=llave, how='left')

def aplicar_missing_threshold(df, umbral):
    """
    Identifica columnas para eliminar según el porcentaje de valores nulos.
    """
    porcentaje_nulos = df.isnull().mean()
    print("Porcentaje de valores nulos por columna:")
    print(porcentaje_nulos)

    columnas_a_conservar = porcentaje_nulos[porcentaje_nulos <= umbral].index
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral].index

    print("\nColumnas a eliminar por superar el umbral:")
    print(columnas_a_eliminar)
    
    return df[columnas_a_conservar]

def imputacion(df):
    """
    Realiza imputación de valores nulos en el DataFrame.
    """

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Imputar valores nulos
    for col in numeric_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df.loc[:, col] = df[col].fillna('Desconocido')

    return df

def analisis_descriptivo(df):
    """
    Genera estadísticas descriptivas y gráficos para las variables del DataFrame.
    """
    variables_categoricas = ['banca', 'segmento', 'producto', 'genero_cli']
    variables_numericas = ['dias_mora_fin', 'vlr_vencido', 'endeudamiento', 'fecha_var_rpta_alt']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Variables", fontsize=16)

    # Gráficos para variables numéricas
    for i, var in enumerate(variables_numericas):
        sns.histplot(df[var], kde=True, ax=axes[0, i], color='skyblue', bins=20)
        axes[0, i].set_title(f"Distribución de {var}")
        axes[0, i].set_xlabel(var)
        axes[0, i].set_ylabel("Frecuencia")

    # Gráficos para variables categóricas
    for i, var in enumerate(variables_categoricas):
        sns.countplot(y=df[var], ax=axes[1, i], order=df[var].value_counts().index, palette="viridis")
        axes[1, i].set_title(f"Distribución de {var}")
        axes[1, i].set_xlabel("Frecuencia")
        axes[1, i].set_ylabel(var)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def duplicados(df):
    """
    Identifica y elimina duplicados en el DataFrame.
    """
    duplicados = df[df.duplicated(subset=['ID', 'fecha_var_rpta_alt'], keep=False)]

    print(f"Total de registros duplicados: {len(duplicados)}")

    # Ordenar por ID y fecha, y mantener el registro más reciente
    df = df.sort_values(by=['ID', 'fecha_var_rpta_alt'], ascending=[True, False])
    df = df.drop_duplicates(subset='ID', keep='first').reset_index(drop = True)
    
    return df

def escalar_codificar(df):
    """
    Escala y codifica variables numéricas y categóricas en el DataFrame.
    """
    # Definir las columnas relevantes
    id_col = 'ID'  
    response_var = 'var_rpta_alt'  

    # Separar las columnas numéricas y categóricas, excluyendo ID y la variable de respuesta
    numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in [id_col, response_var]]
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in [id_col, response_var]]

    # Escalar las columnas numéricas
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Codificar las columnas categóricas con LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Verificación de los datos transformados
    print("\nDatos después del escalamiento y codificación:")
    print(df.head())

    # Verificar estadísticas de las columnas escaladas
    print("\nEstadísticas de las columnas numéricas escaladas:")
    print(df[numeric_cols].describe())

    return df

def entrenar_modelos(df, config):
    """
    Entrena modelos de aprendizaje automático en el DataFrame.
    """

    # Separar características y variable de respuesta
    X = df.drop(columns=['var_rpta_alt'])
    y = df['var_rpta_alt']

    # Conservar el ID para el resultado
    ids = X['ID']
    X = X.drop(columns=['ID'])  # Eliminar ID para el entrenamiento

    # Dividir datos en entrenamiento y validación
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, ids, test_size=0.2, random_state=42)

    # Modelos y configuraciones
    modelos = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "SVC": SVC(probability=True),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB()
    }

    # Métrica de evaluación
    scorer = make_scorer(f1_score)

    # Resultados
    resultados = []
    predicciones_finales = pd.DataFrame() 

    # Validación cruzada y predicciones
    for modelo_nombre, modelo in modelos.items():
        print(f"\nEntrenando {modelo_nombre}...")
        hyperparams = config["modelos"].get(modelo_nombre, {})
        
        grid = GridSearchCV(
            estimator=modelo,
            param_grid=hyperparams,
            scoring=scorer,
            cv=config["cv_folds"],
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        mejor_modelo = grid.best_estimator_
        mejor_puntaje = grid.best_score_
        print(f"Mejor hiperparámetros: {grid.best_params_}")
        print(f"Mejor F1-Score: {mejor_puntaje:.4f}")

        filename = os.getcwd() + config["ruta_modelos"] + modelo_nombre + '.pkl'
        try:
            joblib.dump(mejor_modelo, filename)
            print(f"Modelo guardado en {filename}.")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
        
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1] 
        

        predicciones = pd.DataFrame({
            'ID': ids_test,              
            'var_rpta_alt': y_pred,      
            'Prob_uno': y_prob   ,
            'Modelo' : modelo_nombre        
        })
        
        predicciones_finales = pd.concat([predicciones_finales, predicciones], ignore_index=True)
        
        resultados.append({
            "Modelo": modelo_nombre,
            "Mejor Hiperparámetros": grid.best_params_,
            "Mejor F1-Score": mejor_puntaje
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_modelos.csv", index=False)
    print("\nResultados guardados en: resultados_modelos.csv")

    predicciones_finales = predicciones_finales.sort_values(['ID', 'Modelo'])
    predicciones_finales.to_csv("resultado_prueba.csv", index=False)
    print("\nPredicciones finales guardadas en: resultado_prueba.csv")    
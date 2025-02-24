import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta, index_col=False, compression='zip')

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], errors='ignore', inplace=True)
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return df

def construir_pipeline(x: pd.DataFrame) -> Pipeline:
    caracteristicas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    caracteristicas_numericas = list(set(x.columns) - set(caracteristicas_categoricas))
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), caracteristicas_numericas),
            ('cat', OneHotEncoder(), caracteristicas_categoricas)
        ],
        remainder='passthrough'
    )
    return Pipeline([
        ('preprocesamiento', preprocesador),
        ('pca', PCA()),
        ('selector_kbest', SelectKBest(f_classif)),
        ('clasificador', SVC(kernel="rbf", max_iter=-1, random_state=42))
    ])

def configurar_gridsearch(pipeline: Pipeline) -> GridSearchCV:
    parametros = {
        "pca__n_components": [0.8, 0.9, 0.95, 0.99],
        "selector_kbest__k": [10, 20, 30],
        "clasificador__C": [0.1, 1, 10],
        "clasificador__gamma": [0.1, 1, 10]
    }
    return GridSearchCV(
        pipeline,
        param_grid=parametros,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

def guardar_modelo(ruta: str, modelo: GridSearchCV):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, 'wb') as archivo:
        pickle.dump(modelo, archivo)

def calcular_metricas(tipo: str, y_real, y_predicho) -> dict:
    return {
        'type': 'metrics',
        'dataset': tipo,
        'precision': precision_score(y_real, y_predicho, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_real, y_predicho),
        'recall': recall_score(y_real, y_predicho, zero_division=0),
        'f1_score': f1_score(y_real, y_predicho, zero_division=0)
    }

def calcular_matriz_confusion(tipo: str, y_real, y_predicho) -> dict:
    matriz = confusion_matrix(y_real, y_predicho)
    return {
        'type': 'cm_matrix',
        'dataset': tipo,
        'true_0': {"predicted_0": int(matriz[0, 0]), "predicted_1": int(matriz[0, 1])},
        'true_1': {"predicted_0": int(matriz[1, 0]), "predicted_1": int(matriz[1, 1])}
    }

def ejecutar():
    ruta_entrada = "./files/input/"
    ruta_modelo = "./files/models/model.pkl.gz"
    
    test_df = cargar_datos(os.path.join(ruta_entrada, 'test_data.csv.zip'))
    train_df = cargar_datos(os.path.join(ruta_entrada, 'train_data.csv.zip'))
    
    test_df = limpiar_datos(test_df)
    train_df = limpiar_datos(train_df)
    
    x_test, y_test = test_df.drop(columns=['default']), test_df['default']
    x_train, y_train = train_df.drop(columns=['default']), train_df['default']
    
    pipeline = construir_pipeline(x_train)
    modelo_entrenado = configurar_gridsearch(pipeline)
    modelo_entrenado.fit(x_train, y_train)
    
    guardar_modelo(ruta_modelo, modelo_entrenado)
    
    y_train_pred = modelo_entrenado.predict(x_train)
    y_test_pred = modelo_entrenado.predict(x_test)
    
    metricas = [
        calcular_metricas('train', y_train, y_train_pred),
        calcular_metricas('test', y_test, y_test_pred),
        calcular_matriz_confusion('train', y_train, y_train_pred),
        calcular_matriz_confusion('test', y_test, y_test_pred)
    ]
    
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as archivo:
        for resultado in metricas:
            archivo.write(json.dumps(resultado) + "\n")

if __name__ == "__main__":
    ejecutar()

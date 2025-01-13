import os
from src.utils import load_config, load_csv, cruzar_df, aplicar_missing_threshold, imputacion, analisis_descriptivo, duplicados, escalar_codificar, entrenar_modelos

def main():

    config = load_config('config.json')
    df_1 = load_csv(ruta = os.getcwd() + config["ruta_1"], columnas = config["columnas_1"])
    df_2 = load_csv(ruta = os.getcwd() + config["ruta_2"], columnas = config["columnas_2"])

    df = cruzar_df(df1 = df_1, df2 = df_2, llave = 'nit_enmascarado')

    df['ID'] =  df['nit_enmascarado'].astype(str) + '#' + \
                df['num_oblig_orig_enmascarado'].astype(str) + '#' + \
                df['num_oblig_enmascarado'].astype(str)

    df.drop(columns=['nit_enmascarado', 'num_oblig_orig_enmascarado', 'num_oblig_enmascarado'], inplace=True)

    print("Información general del conjunto de datos:")
    print(df.info())

    print("Valores nulos por columna:")
    print(df.isnull().sum())

    df = aplicar_missing_threshold(df, 0.5)
    df = imputacion(df)
    df = duplicados(df)
    analisis_descriptivo(df)
    df = escalar_codificar(df)

    entrenar_modelos(df=df, config= config)

if __name__ == "__main__":
    main()

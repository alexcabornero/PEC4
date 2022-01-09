import os
import pandas as pd

"""
Funciones para el pre-procesamiento de los datos originales y 
preparación de los archivo de trabajo.
"""


def upper_words(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    r"""
    Cambia la primera letra de cada palabra a mayúscula de una columna.

    Parameters:
        df: pd.Dataframe
        columna: str

    Returns:
        DataFrame con la columna modificada.
    """
    df[columna] = df[columna].apply(lambda x: x.title())
    return df


def save_clean_csv(df: pd.DataFrame, file: str) -> None:
    r"""
    Crea un csv con los datos procesados.

    Parameters:
        df: pd.Dataframe
        file: csv

    Returns:
        None
    """
    path = os.path.join(os.getcwd(), 'data/processed/', file)
    df.to_csv(path, index=False, sep=';', encoding='utf-8')


def count_null_values(df: pd.DataFrame, columna: str) -> int:
    r"""
    Cuenta los null de una columna en un dataframe.

    Parameters:
        df: pd.Dataframe
        columna: str

    Returns:
        Número de nulos en una columna.
    """
    return df[columna].isnull().sum()


def null_values_as_mean(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    r"""
    Sustituye los null de una columna con el valor de la media.

    Parameters:
        df: pd.Dataframe
        columna: str

    Returns:
        Dataframe con los datos modificados
    """
    null_values = df[columna].isnull()
    new_value = df[columna].mean()
    df[null_values] = new_value
    return df


def merge_df(df1: pd.DataFrame,
             df2: pd.DataFrame,
             cols: list,
             suffix: list = []) -> pd.DataFrame:
    r"""
    Crea un Dataframe a partir de 2 dataframes tomando como referencia una
    serie de columnas.

    Parameters:
        df1: pd.Dataframe
        df2: pd.Dataframe
        cols: list str
        suffix: list str

    Returns:
        Dataframe
    """
    new_df = pd.merge(df1, df2, on=cols, suffixes=suffix)
    return new_df


def transform_data() -> None:
    r"""
    Crea un dataset des-normalizado a partir del zip con los datos iniciales.

    Parameters:

    Returns:
        None. Crea un csv data_norm.csv en /data/processed/
    """
    # Directorio de los datos en csv
    path = os.path.join(os.getcwd(), 'data/interim/')

    # Path de cada uno de los archivos
    albums_data = os.path.join(path, 'albums_norm.csv')
    artists_data = os.path.join(path, 'artists_norm.csv')
    tracks_data = os.path.join(path, 'tracks_norm.csv')

    # Carga de los datos en un dataframe
    albums = pd.read_csv(albums_data, sep=';')
    artists = pd.read_csv(artists_data, sep=';')
    tracks = pd.read_csv(tracks_data, sep=';')

    # Capitalizamos el nombre de los artistas y guardamos
    # en un csv límpio
    artists = upper_words(artists, 'name')

    # Recuento de registros de tracks con popularity sin valor
    # Asignación de la media a valores null
    popularity_null = count_null_values(tracks, 'popularity')
    print('El número de registros de tracks con Popularity sin valor es {}.'.
          format(popularity_null))
    print('Media con valores null {}'.format(tracks.popularity.mean()))
    tracks = null_values_as_mean(tracks, 'popularity')
    print('Después de sustituir con media los null son {} registros.'.
          format(count_null_values(tracks, 'popularity')))

    # Creación de los csv límpios con los valores modificados
    artist_file = 'artists_clean.csv'
    tracks_file = 'tracks_clean.csv'
    albums_file = 'albums_clean.csv'
    # Creación csv con los datos pre-procesados
    save_clean_csv(artists, artist_file)
    save_clean_csv(tracks, tracks_file)
    save_clean_csv(albums, albums_file)

    # Creación del dataframe normalizado
    first_merge = merge_df(tracks, artists, 'artist_id', ['_track', '_artist'])
    last_merge = merge_df(first_merge, albums,
                          ['artist_id', 'album_id'], ['', '_album'])

    # Almacenamos el dataframe de trabajo en un csv
    save_clean_csv(last_merge, 'data_norm.csv')

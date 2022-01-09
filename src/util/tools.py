import datetime
# import math
import json
import os.path
import pandas as pd
import time
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import numpy as np
# from math import dist

# from scipy import spatial
from scipy.spatial import distance


def load_data(file: str) -> pd.DataFrame:
    r"""
    Carga los datos de un csv en un dataframe.

    Parameters:
        file: str

    Returns:
        pd.Dataframe: datos procesados
    """
    path = os.path.join(os.getcwd(), 'data/processed/', file)
    data = pd.read_csv(path, sep=';')

    return data


def show_info(data: pd.DataFrame) -> None:
    r"""
    Muestra información sobre el total de tracks y el número de columnas en el
    dataframe.

    Parameters:
        data: pd.Dataframe

    Returns:
        None
    """
    # Función para mostrar info del ejercicio 1
    total_tracks = len(data.track_id.unique())
    total_cols = len(data.columns.values)
    print('-----------------------------------------------------')
    print('El número total de tracks son {}'.format(total_tracks))
    print('-----------------------------------------------------')
    print('El número total de columnas son {}'.format(total_cols))
    print(data.columns.values)


def get_column_pandas(path: str, column_name: str) -> (list, float):
    r"""
    Lee la columna de un dataframe con pandas y la devuelve como lista y
    el tiempo total utilizado en su lectura.

    Parameters:
        path: str
        column_name: str

    Returns:
        result: list de la variable leída
        t: tiempo total empleado en la lectura
    """
    start_time = time.time()
    df = pd.read_csv(path, sep=';')
    result = df[column_name].tolist()
    stop_time = time.time()
    t = stop_time - start_time

    return result, t


def get_column_manual_read(path: str, column_to_read: str) -> (list, float):
    r"""
    Lee la columna de un dataframe usando lectura secuencial línea a línea
    y la devuelve como lista y el tiempo total utilizado en su lectura.

    Parameters:
        path: str
        column_to_read: str

    Returns:
        list_result: list de la variable leída
        t: tiempo total empleado en la lectura
    """
    start_time = time.time()
    list_result = []
    with open(path) as f:
        headers = f.readline()
        data = f.readlines()

        for line in data:
            # toma la columna
            c = line.split(';')[column_to_read]
            if c not in list_result:
                list_result.append(c)
    stop_time = time.time()
    t = stop_time - start_time

    return list_result, t


def plot_time(time1: list, time2: list,
              x_labels: str = None, title_x: str = None, title_y: str = None,
              title_graph: str = None) -> None:
    r"""
    Crea un gráfico de barras a partir de listas de tiempo para comparar
    tiempos de los métodos de lectura.

    Parameters:
        time1: list
        time2: list
        x_labels: str
        title_x: str
        title_y: str
        title_graph: str

    Returns:
        None
    """
    barwidth = 0.25
    plt.figure(figsize=(10, 5))

    r1 = np.arange(len(time1))
    r2 = [x + barwidth for x in r1]

    plt.bar(r1, time1, color='green', width=barwidth, label='pandas')
    plt.bar(r2, time2, color='blue', width=barwidth, label='manual')

    plt.xlabel(title_x)
    plt.xticks([r + barwidth for r in range(len(time1))],
               x_labels)
    plt.ylabel(title_y)
    plt.title(title_graph)
    plt.legend()
    plt.show(block=False)
    plt.pause(interval=5)
    plt.close("all")


def get_tracks(df: pd.DataFrame, field: str, value) -> pd.DataFrame:
    r"""
    Obtiene las canciones en función del valor de una variable dada.

    Parameters:
        df: pd.Dataframe
        field: str
        value: Any (puede tomar cuaquier valor str, int, float)

    Returns:
        pd.DataFrame: tracks en función del valor de una audio feature.
    """
    return df[df[field] == value]


def tracks_with_word(serie: pd.Series, word: str) -> int:
    r"""
    Obtiene el número de veces que aparece una palabra en una serie. Busca
    tanto palabras que comienzan en minúscula o mayúscula.

    Parameters:
        serie: pd.Series
        word: str

    Returns:
        int: número de tracks con la palabra en el título.
    """
    result = serie.str.contains(word).sum()
    result += serie.str.contains(str.capitalize(word)).sum()

    return result


def count_tracks_decade(df: pd.DataFrame, decade: list) -> int:
    r"""
    Obtiene el número de canciones en una década.

    Parameters:
        df: pd.DataFrame
        decade: list

    Returns:
        int: número de tracks en una década.
    """
    return df[df['release_year'].isin(decade)]['track_id'].count()


def get_tracks_highest_feature(df: pd.DataFrame,
                               feature: str, years: list) -> pd.DataFrame:
    r"""
    Obtiene las canciones con puntuación más alta de una audio feature.

    Parameters:
        df: pd.DataFrame
        feature: str
        years: list

    Returns:
        pd.DataFrame: tracks con el valor máximo de la audio feature.
    """
    # Filtrado de canciones en los años indicacos
    tracks = df[df['release_year'].isin(years)]
    # Selección de las canciones con puntuación máxima
    tracks_most_popular = \
        tracks[tracks[feature] == tracks[feature].max()]
    return tracks_most_popular


def get_feature_stats(feature: pd.Series) -> (float, float, float):
    r"""
    Obtiene los estadísticos mínimo, media y máximo de una serie de datos.

    Parameters:
        feature: pd.Series

    Returns:
        minimo: float mínimo de la serie
        media: float media de la serie
        maximo: float máximo de la serie
    """
    minimo = np.min(feature)
    media = np.mean(feature)
    maximo = np.max(feature)

    return minimo, media, maximo


def get_feature_artist(
        df: pd.DataFrame, feature: str, artist: str) -> pd.Series:
    r"""
    Devuelve una audio feature de un artista.

    Parameters:
        df: pd.DataFrame
        feature: str
        artist: str

    Returns:
        pd.Series: audio feature de un artista
    """
    return get_tracks(df, 'name_artist', artist)[feature]


def get_dist_euclidean(v1: list, v2: list) -> float:
    r"""
    Calcula la similitud euclidea entre dos vectores.

    Parameters:
        v1: list
        v2: list

    Returns:
        float: similitud euclidea
    """
    return distance.euclidean(v1, v2)


def get_dist_cosine(v1: list, v2: list) -> float:
    r"""
    Calcula la similitud cosinus entre dos vectores.

    Parameters:
        v1: list
        v2: list

    Returns:
        float: similitud euclidea
    """
    return 1 - distance.cosine(v1, v2)


def get_audio_vector(df_artist: pd.DataFrame, audio_features: list) -> list:
    r"""
    Obtiene el vector de las medias de las audio features de un artista.

    Parameters:
        df_artist: pd.DataFrame
        audio_features: list

    Returns:
        list: media de cada audio feature del artista.
    """
    vector = [df_artist[x].mean() for x in audio_features]

    return vector


def get_decades(first_year: int, last_year: int) -> list:
    r"""
    Crea una lista de listas con cada década desde el primer año hasta el
    último año.

    Parameters:
        first_year: int. Primer año a contabilizar.
        last_year: int. Último año a contabilizar

    Returns:
        list: lista con las décadas en forma de lista entre los años indicados.
    """
    next_decade = first_year
    decades = []
    finished = False

    while not finished:
        decades.append(np.arange(next_decade, next_decade + 10))
        if next_decade + 10 > last_year:
            finished = True
        else:
            next_decade += 10

    return decades


def count_artist_tracks_decade(df: pd.DataFrame, artist: str, decade: list) -> int:
    r"""
    Obtiene el número de canciones en una década de un artista.

    Parameters:
        df: pd.DataFrame
        artist: str.
        decade: list

    Returns:
        int: número de tracks en una década de un artista.
    """
    artist_tracks = get_tracks(df, 'name_artist', artist)

    return artist_tracks[artist_tracks['release_year'].
        isin(decade)]['track_id'].count()


def get_artists_all_decades(df: pd.DataFrame, decades: list) -> list:
    r"""
    Obtiene los artistas con canciones en todas las décadas.

    Parameters:
        df: pd.DataFrame
        decades: list de las décadas

    Returns:
        list: artistas con canciones en todas las décadas.
    """
    # Lista de todos los artistas
    artists = df['name_artist'].unique()
    # Lista vacía para añadir los artistas que tengan canciones en todas las
    # décadas
    artists_list = []

    # Comprobamos cada artista en cada década
    for artist in artists:
        # Contador de décadas con canciones
        counter_decades = 0
        for decade in decades:
            # Si tiene canciones en la década
            if count_artist_tracks_decade(
                    df, artist, decade) > 0:
                # Incremento del contador de décadas
                counter_decades += 1
        # Si el contador es igual al número de décadas se añade el artista
        if counter_decades == len(decades):
            artists_list.append(artist)

    return artists_list


def plot_feature_mean(df: pd.DataFrame, feature: str) -> None:
    r"""
    Crea un gráfico de barras con la media de una audio feature de un artista.

    Parameters:
        df: pd.DataFrame
        feature: str

    Returns:
        None
    """
    df_plot = df[['name', feature]]
    sns.histplot(data=df_plot, kde=True, stat='probability', bins=50)
    plt.show(block=False)
    plt.pause(interval=3)
    plt.close("all")


def plot_hist_density_feature(df: pd.DataFrame, feature: str,
                              density: bool = False) -> None:
    r"""
    Crea un histograma de una audio feature de un artista.

    Parameters:
        df: pd.DataFrame
        feature: str
        density: boolean

    Returns:
        None
    """
    data = df[feature]
    sns.histplot(data=data, kde=True, stat='probability', bins=50)
    plt.title('{}'.format(feature))
    plt.show(block=False)
    plt.pause(interval=3)
    plt.close("all")


def plot_dual_hist(
        data1: pd.Series, data2: pd.Series, density: bool = False) -> None:
    r"""
    Crea un histograma múltiple comparando dos artistas.

    Parameters:
        data1: pd.Series
        data2: pd.Series
        density: boolean

    Returns:
        None
    """
    '''
    data1.hist(
        label='Artist1', density=density, alpha=0.5, color='green', bins=50)
    data2.hist(
        label='Artist2', density=density, alpha=0.5, color='blue', bins=50)
    '''
    sns.histplot(
        data=data1, kde=True, stat='probability', bins=50, color='green')
    sns.histplot(
        data=data2, kde=True, stat='probability', bins=50, color='red')
    plt.show(block=False)
    plt.pause(interval=3)
    plt.close("all")


def plot_matrix_similarity(
        df: pd.DataFrame, distance_type: str = 'euclidean') -> None:
    r"""
    Crea un heatmap en función de la similitud entre dos vectores

    Parameters:
        df: pd.DataFrame con los vectores de medias de las audio features
        distance_type: str tipo de similitud

    Returns:
        None
    """
    plt.style.use("seaborn")
    # Calculo de la dístancia entre los artistas
    vector_result = []

    for art1 in df.columns.values:
        for art2 in df.columns.values:
            if distance_type == 'euclidean':
                vector_result.append(get_dist_euclidean(df[art1], df[art2]))
            if distance_type == 'cosine':
                vector_result.append(get_dist_cosine(df[art1], df[art2]))

    # Calculo del tamaño de las líneas para crear la matriz
    n = int(np.sqrt(len(vector_result)).round(0))
    matriz = np.array(vector_result).reshape(n, n)

    # Mostrar el heatmap
    plt.figure(figsize=(10, 10))
    heat_map = sns.heatmap(matriz, linewidth=1, annot=True)
    plt.title("Heatmap según la similitud {}".format(distance_type))
    plt.show(block=False)
    plt.pause(interval=3)
    plt.close("all")


def get_API_artist(url: str, artist: str) -> requests.Response:
    r"""
    Realiza una llamada a la url especificada con el nombre del artista.

    Parameters:
        url: str con la dirección de consulta
        artist: str con el nombre del artista

    Returns:
        Datos del artista
    """
    wait = 4
    url_artist = url + '?s=' + artist
    response = requests.get(url_artist, timeout=wait)
    return response


def create_API_dataframe(artists: list, url) -> pd.DataFrame:
    r"""
    Crea un dataframe con los datos de los artistas de una lista conseguidos
    a través de la url.

    Parameters:
        artists: list con el nombre del artista
        url: str con la dirección de consulta

    Returns:
        DataFrame con los datos de los artistas.
    """
    df = pd.DataFrame(columns=['artist_name', 'formed_year', 'country'])

    for artist in artists:
        # Request para los datos del artista
        artist_data = get_API_artist(url, artist)
        # Convertir el json a dict para extraer los datos necesarios
        data_dict = artist_data.json()["artists"][0]
        # Se añaden los datos del artista al dataframe
        df = df.append(
            {'artist_name': data_dict['strArtist'],
             'formed_year': data_dict['intFormedYear'],
             'country': data_dict['strCountry']},
            ignore_index=True)

    return df


def get_exerc_2() -> (list, list):
    r"""
    Realiza el ejercicio 2 de la PEC calculando los tiempos de lectura usando
    pandas o un método manual

    Parameters:

    Returns:
        (list, list): vectores con los tiempos empleados usando lectura con
        pandas y lectura manual
    """
    # Carga de los datos
    albums_file = os.path.join(
        os.getcwd(), 'data/interim/', 'albums_norm.csv')
    artists_file = os.path.join(
        os.getcwd(), 'data/interim/', 'artists_norm.csv')
    tracks_file = os.path.join(
        os.getcwd(), 'data/interim/', 'tracks_norm.csv')

    # Leemos la columna con pandas o de forma manual y obtenemos el tiempo
    albums_pandas, t_alb_pd = get_column_pandas(albums_file, 'album_id')
    albums_manual, t_alb_man = get_column_manual_read(albums_file, 1)
    artists_pandas, t_art_pd = get_column_pandas(artists_file, 'artist_id')
    artists_manual, t_art_man = get_column_manual_read(artists_file, 0)
    tracks_pandas, t_trck_pd = get_column_pandas(tracks_file, 'track_id')
    tracks_manual, t_trck_man = get_column_manual_read(tracks_file, 2)

    # Guardamos los datos en una lista por cada método usado
    pandas_times = [t_alb_pd, t_art_pd, t_trck_pd]
    manual_times = [t_alb_man, t_art_man, t_trck_man]

    return pandas_times, manual_times


def get_exerc_3() -> None:
    r"""
    Realiza el ejercicio 3 de la PEC realizando las llamadas a las funciones
    creadas para cada uno de los apartados.

    Parameters:

    Returns:
        None
    """
    # Cargamos los datos
    tracks_file = os.path.join(
        os.getcwd(), 'data/processed/', 'data_norm.csv')
    df = pd.read_csv(tracks_file, sep=';')

    # Apartado A
    # Búsqueda de las tracks de Radiohead
    artist = 'Radiohead'
    result = get_tracks(df, 'name_artist', artist)['track_id'].count()
    # Muestra en pantalla el resultado
    print('A. El número de tracks del artista {} es {}.'.
          format(artist, result))

    # Apartado B
    # Búsqueda de los tracks con police en el título
    word = 'police'
    serie = df['name_track']
    # Muestra en pantalla el resultado
    print('B. El número de tracks que cotiene la palabra {} es {}.'.
          format(word, tracks_with_word(serie, word)))

    # Apartado C
    # Búsqueda de los tracks publicados en la década de 1990
    years = np.arange(1990, 2000)
    # Muestra en pantalla el resultado
    print('C. El número de tracks de albumes publicados en los 90s es {}.'.
          format(count_tracks_decade(df, years)))

    # Apartado D
    # Búsqueda del track con más popularidad de los últimos 10 años
    years = np.arange(datetime.date.today().year - 9,
                      datetime.date.today().year + 1)
    feature = 'popularity_track'
    result = get_tracks_highest_feature(df, feature, years)
    # Muestra en pantalla el resultado
    print('D. Las canciones con la popularity más alta son: {}'.
          format(result[['name_track', 'release_year', 'popularity_track']]))

    # Apartado E
    # Búsqueda de los artistas con tracks en cada una de las décadas desde 1960
    print('Apartado E')
    print('Los artistas con canciones en todas las décadas desde 1960 son: {}'.
          format(get_artists_all_decades(df, get_decades(1960, 2020))))


def get_exerc_4() -> None:
    r"""
    Realiza el ejercicio 4 de la PEC realizando las llamadas a las funciones
    creadas para cada uno de los apartados.

    Parameters:

    Returns:
        None
    """
    # Cargamos los datos
    tracks_file = os.path.join(
        os.getcwd(), 'data/processed/', 'data_norm.csv')
    df = pd.read_csv(tracks_file, sep=';')
    # Apartado A
    # Búsqueda de las canciones de Metallica
    artist = 'Metallica'
    metallica = get_tracks(df, 'name_artist', artist)
    feature = 'energy'
    mi, me, ma = get_feature_stats(metallica[feature])
    print('A. Los estadísticos de {} de {} son:'.format(artist, feature))
    print('Mínimo: {}; Media: {}; Máximo: {}'.format(mi, me, ma))

    # Apartado B
    # Búsqueda de las canciones de Coldplay
    artist = 'Coldplay'
    coldplay = get_tracks(df, 'name_artist', artist)
    feature = 'danceability'

    plot_feature_mean(coldplay, feature)


def get_exerc_5() -> None:
    r"""
    Realiza el ejercicio 5 de la PEC realizando las llamadas a las funciones
    creadas para cada uno de los apartados.

    Parameters:

    Returns:
        None
    """
    # Cargamos los datos
    tracks_file = os.path.join(
        os.getcwd(), 'data/processed/', 'data_norm.csv')
    df = pd.read_csv(tracks_file, sep=';')

    # Búsqueda de las canciones de Ed Sheeran
    artist = 'Ed Sheeran'
    ed_sheeran = get_tracks(df, 'name_artist', artist)
    feature = 'acousticness'

    plot_hist_density_feature(ed_sheeran, feature, density=True)


def get_exerc_6() -> None:
    r"""
    Realiza el ejercicio 6 de la PEC realizando las llamadas a las funciones
    creadas para cada uno de los apartados.

    Parameters:

    Returns:
        None
    """
    # Cargamos los datos
    tracks_file = os.path.join(
        os.getcwd(), 'data/processed/', 'data_norm.csv')
    df = pd.read_csv(tracks_file, sep=';')
    feature = 'energy'
    artist1 = 'Adele'
    artist2 = 'Extremoduro'

    features1 = get_feature_artist(df, feature, artist1)
    features2 = get_feature_artist(df, feature, artist2)

    plot_dual_hist(features1, features2, density=True)


def get_exerc_7() -> None:
    r"""
    Realiza el ejercicio 7 de la PEC realizando las llamadas a las funciones
    creadas para cada uno de los apartados.

    Parameters:

    Returns:
        None
    """
    # Cargamos los datos
    tracks_file = os.path.join(
        os.getcwd(), 'data/processed/', 'data_norm.csv')
    df = pd.read_csv(tracks_file, sep=';')

    audio_features = ['danceability', 'energy',
                      'key', 'loudness', 'mode', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness',
                      'valence', 'tempo', 'time_signature']

    artist = ['Metallica', 'Extremoduro', 'Ac/Dc', 'Hans Zimmer']
    data = pd.DataFrame()

    for art in artist:
        data[art] = get_audio_vector(
            get_tracks(df, 'name_artist', art), audio_features)

    plot_matrix_similarity(data, 'euclidean')
    plot_matrix_similarity(data, 'cosine')


def get_exerc_8():
    # url de consulta para las requests
    url = 'https://www.theaudiodb.com/api/v1/json/2/search.php'

    # lista de artistas de tests
    artists = ['coldplay', 'metallica', 'beyonce', 'u2', 'pink floyd']

    # Dataframe con los datos de los artistas
    data = create_API_dataframe(artists, url)

    # Creación del csv
    data.to_csv('artists_audiodb.csv')
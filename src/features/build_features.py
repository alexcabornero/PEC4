import os
import zipfile as zf


"""
Funciones para la extracción de los datos originales.
"""


def clean_csv_data(path: str) -> None:
    r"""
    Elimina los archivos y directorios del path específicado.

    Parameters:
        path: str

    Returns:
        None
    """
    # Directorio de los datos descomprimidos
    directory = os.path.join(os.getcwd(), 'data/', path)

    # Recorremos el directorio y eliminamos el contenido
    with os.scandir(directory) as dir_list:
        for f in dir_list:
            if os.path.isfile(f):
                os.remove(f)
            if os.path.isdir(f):
                os.rmdir(f)


def read_raw_data(zfile: str) -> None:
    r"""
    Extrae los archivos comprimidos en un zip.

    Parameters:
        zfile: str

    Returns:
        None
    """
    # Eliminamos los archivos pre-tratados y límpios
    clean_csv_data('interim/')
    clean_csv_data('processed/')

    # Ruta de los datos de trabajo
    raw_zip_data = os.path.join(os.getcwd(), 'data/raw/', zfile)

    # Descomprimimos el fichero zip en el directorio data/interim
    if zf.is_zipfile(raw_zip_data):
        with zf.ZipFile(raw_zip_data, 'r') as zip_f:
            zip_f.extractall(path=os.path.join(os.getcwd(), 'data/interim/'))
import os
import src.features.build_features as bf
import src.data.make_dataset as md
import src.util.tools as tools


def main():
    # Ejercicio 1
    # Lectura de los datos comprimidos en archivo zip
    print('Ejecutando Ejercicio 1.....')
    file = 'data.zip'
    bf.read_raw_data(file)

    # Preparación y transformación de los datos
    md.transform_data()

    # Carga de los datos transformados y límpios
    file_norm = 'data_norm.csv'
    df = tools.load_data(file_norm)

    # Información solicitada en ejercicio 1
    tools.show_info(df)

    # Ejercicio 2
    print('Ejecutando Ejercicio 2.....')
    t1, t2 = tools.get_exerc_2()

    print('Tiempos con método pandas: {}'.format(t1))
    print('Tiempos con método manual: {}'.format(t2))

    tools.plot_time(t1, t2, x_labels=['Albums', 'Artists', 'Tracks'],
                    title_graph='Tiempos por método usado', title_x='Datos',
                    title_y='Tiempo')

    # Ejercicio 3
    print('Ejecutando Ejercicio 3.....')
    tools.get_exerc_3()

    # Ejercicio 4
    print('Ejecutando Ejercicio 4.....')
    tools.get_exerc_4()

    # Ejercicio 5
    print('Ejecutando Ejercicio 5.....')
    tools.get_exerc_5()

    # Ejercicio 6
    print('Ejecutando Ejercicio 6.....')
    tools.get_exerc_6()

    # Ejercicio 7
    print('Ejecutando Ejercicio 7.....')
    tools.get_exerc_7()

    # Ejercicio 8
    print('Ejecutando Ejercicio 8.....')
    tools.get_exerc_8()


if __name__ == '__main__':
    main()

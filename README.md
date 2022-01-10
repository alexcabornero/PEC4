# PEC4 - ALEJANDRO CABORNERO LÓPEZ
UOC Programación para la ciencia de datos

El proyecto pretende responder a las distintas preguntas y ejercicios 
planteados para la PEC4 de la asignatura Programación para la ciencia de 
datos.  

La estructura de archivos del mismo es la siguiente:

PEC4
* data
    * interim
    * processed
    * raw
* plots
* src
    * data
    * features
    * util
* tests


data: contiene los datos de trabajo dividido en los siguientes dir:  
* raw: datos brutos de inicio.  
* interim: datos pre-procesados de los archivos originales.  
* processed: datos limpios y preparados para el estudio.  

plots: contiene los archivos de imagenes de los gráficos solicitados.  
src: directorio de código  
* data: funciones para la manipulación y limpieza de los archivos   
         iniciales y preparación del dataset de trabajo.  
* features: funciones para carga y limpieza de directorios.  
* util: funciones para la realización de los distintos ejercicios. 

<h3>Requerimientos</h3>
<p>Para la realización de la PEC se ha utilizado un entorno virtual en el
que se han cargado los paquetes utilizados. Para replicar dicho entorno 
de trabajo, todos los paquetes necesarios están disponibles en el 
archivo requeriments.txt. Se recomienda la instalación de los mismos 
paquetes para evitar incompatibilidades. Para ello, desde el directorio
PEC4 ejecute el siguiente comando desde la terminal:</p>
> pip install -r requeriments.txt

<h3>Ejecucción código</h3>
<p>Atención: la estructura de directorios proporcionada no contiene ningún
fichero en los directorios data/interim ni data/processed puesto que son
creados mediante el código. El script utiliza la función clean_csv_data() 
de src/build_features.py que en cada ejecucción del código elimina todos
los archivos previamente creados. Por tanto, si se desea ejecutar varias
veces el script, se deben tener los permisos del sistema necesarios para
permitir que la función clean_csv_data() limpie dichos directorios. De otro
modo se producirá un error durante la ejecucción.</p>

<p>Para ejecutar el código correctamente, se debe situar en el directorio
PEC4 y desde la terminal ejecutar el siguiente comando:</p>
> python3 main.py
<p>En la terminal se irá mostrando la información solicitada en la PEC
y los gráficos se mostrarán durante 3 segundos antes de cerrarse y 
continuar con el resto del ejercicio. Si se necesita consultar alguno
de los gráficos, se encuentran disponibles en el directorio plots.</p>

<h3>Realización tests</h3>
<p>Para la realización de los test de prueba creados, desde el directorio
PEC4 ejecutar en la terminal el siguiente comando:</p>
> coverage run --source=. -m unittest discover -s tests/
<p>Es posible obtener un informe detallado con el comando:</p>
> coverage report -m
<p>En el momento de la realización de este ejercicio, los resultados
obtenidos se encuentran en el archivo coverage_test.txt</p>

<h3>Repositorio Github</h3>
<p>El proyecto se encuentra disponible como repositorio Github en el siguiente link</p>
https://github.com/alexcabornero/PEC4.git




# ZonaCEM AI: Estimación de las Zonas de Exposición a Campos Electromagnéticos en Estaciones Base Celulares Usando Inteligencia Artificial

## Conjuntos de datos y modelos entrenados disponibles en: 
https://gitlab.com/tulcanjose0/zonacem-ai/-/tree/e95ebba44fefaf4bce6609ad0a3c70d4c37cac2f/

## Artículo disponible en:
https://doi.org/10.5281/zenodo.17110951

## Aplicación disponible en:
[ZonaCEM AI](https://zonacem-ai-app.streamlit.app/)  

Esta app permite estimar la potencia recibida en estaciones base celulares en escenarios urbanos, mediante tres modelos basados en la arquitectura U-Net.

Los modelos fueron entrenados con tres datasets, cada un de ellos con 10.000 escenarios diferentes.
Los datasets se componen de cuatro capas:

- Estructuras: Representación de edificaciones y paredes.
- Posición de la estación base: Ubicación de las antenas trasmisoras.
- Mediciones dispersas: Mediciones de potencia recibida.
- Potencia recibida: Mapa de calor de potencia recibida

Siendo las tres primeras capas, las entradas de los modelos entrenados y la última la salida.
Los tres modelos se diferencian por la frecuencia de funcionamiento de las estaciones base con las que 
se generó cada uno de los datasets. Los datasets fueron generados para las frecuencias de 1.95GHz, 2.13GHz y 2.65GHz.

Las imágenes tienen un tamaño de 256 x 256 con una resolución de 20 cm por pixel.
Los modelos predicen el mapa de potencia recibida a partir de 30 mediciones, lo que corresponde aproximadamente a 0.045% de los pixeles de la imagen.

Esta aplicación permite evaluar los modelos usando imágenes disponibles de los datasets, además de evaluar nuevos escenarios.
Es posible cargar las imágenes de los nuevos escenarios, pero en caso de no tenerlas, la aplicación cuenta con una opción para crear estas imágenes para su posterior evaluación.

## Contribuciones

En esta investigación se propone una metodología de estimación de mapas de entorno radioeléctrico que usa tres modelos en escenarios urbanos, los cuales tienen base en la arquitectura U-Net.
- Tres datasets de imágenes que representan la potencia recibida en estaciones base celulares, obtenidos mediante simulación en el software Matlab empleando el modelo de espacio libre con pérdidas por paredes. Un dataset para cada una de las frecuencias de 1.95 GHz, 2.13 GHz y 2.65 GHz.
- Tres modelos entrenados empleando una arquitectura basada en U-Net que permiten estimar los mapas de entorno radioeléctrico de una estación base celular. Dichos modelos requieren como entradas la posición de la estación base, la disposición de las paredes a su alredor, y el 0.045 % de las medidas, lo que representa una diferencia significativa en relación con los demás métodos del estado del arte.
- Medidas dispersas correspondientes al 0.045 % de las mediciones, las cuales son tomadas en un escenario real y constituyen las métricas que se usan para evaluar la efectividad de los modelos creados. 
- ZonaCEM AI, una aplicación que permite la creación y evaluación de nuevos escenarios con cada uno de los modelos

## Autores
**José Luis Mera Tulcán**
 
**Giovanni Javier Pantoja Mora**




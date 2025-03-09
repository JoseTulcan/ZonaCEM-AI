# ZonaCEM AI: Estimación de las Zonas de Exposición a Campos Electromagnéticos en Estaciones Base Celulares Usando Inteligencia Artificial

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

Esta app permite evaluar los modelos usando imágenes disponibles de los datasets, además de evaluar nuevos escenarios.
Es posible cargar las imágenes de los nuevos escenarios, pero en caso de no tenerlas, la app cuenta con una opción para crear estas imágenes para su posterior
evaluación.

Modelos entrenados disponibles en: https://gitlab.com/tulcanjose1/zonacem-ai/-/tree/main

## App disponible en:

[ZonaCEM AI](https://zonacem-ai.streamlit.app/) 

## Contribuciones


## Como citar este material


## Autores
**José Luis Mera Tulcán**

**Giovanni Javier Pantoja Mora**




#Creador: Ivan Alejandro Dominguez Perez
#Grupo: 6E1
#Registro: 21310234
#Este archivo se encarga de inportar los archvivos generados por el entrenamiento para poder ejecutar
#y hacer funcionar el chatbot
import random  # Importa el módulo random para operaciones aleatorias
import json  # Importa el módulo json para manejar archivos JSON
import pickle  # Importa el módulo pickle para serialización de objetos
import numpy as np  # Importa NumPy para operaciones con arrays

import nltk  # Importa NLTK para procesamiento de lenguaje natural
from nltk.stem import WordNetLemmatizer  # Importa el lematizador de WordNet

from keras.models import load_model  # Importa la función para cargar un modelo de Keras

# Inicializa el lematizador de WordNet
lemmatizer = WordNetLemmatizer()

# Carga los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())  # Carga el archivo JSON con los intents (intenciones)
words = pickle.load(open('words.pkl', 'rb'))  # Carga el archivo pickle con las palabras
classes = pickle.load(open('classes.pkl', 'rb'))  # Carga el archivo pickle con las clases
model = load_model('chatbot_model.h5')  # Carga el modelo entrenado

# Función para limpiar una oración y lematizar sus palabras
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokeniza la oración
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lematiza las palabras
    return sentence_words

# Función para convertir una oración en una bolsa de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Limpia y tokeniza la oración
    bag = [0] * len(words)  # Crea una bolsa de palabras inicializada en 0
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:  # Si la palabra está en el vocabulario
                bag[i] = 1  # Marca la posición correspondiente con 1
    return np.array(bag)  # Retorna la bolsa de palabras como un array de NumPy

# Función para predecir la clase de una oración
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Convierte la oración en una bolsa de palabras
    res = model.predict(np.array([bow]))[0]  # Realiza la predicción con el modelo
    max_index = np.where(res == np.max(res))[0][0]  # Encuentra el índice de la clase con mayor probabilidad
    category = classes[max_index]  # Obtiene la clase correspondiente
    return category

# Función para obtener una respuesta aleatoria basada en la clase predicha
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']  # Obtiene la lista de intenciones
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:  # Si el tag coincide
            result = random.choice(i['responses'])  # Escoge una respuesta aleatoria
            break
    return result  # Retorna la respuesta

# Función principal para generar una respuesta a un mensaje
def respuesta(message):
    ints = predict_class(message)  # Predice la clase del mensaje
    res = get_response(ints, intents)  # Obtiene una respuesta basada en la clase
    return res  # Retorna la respuesta

# Bucle principal para interactuar con el chatbot
while True:
    message = input()  # Recibe un mensaje del usuario
    print(respuesta(message))  # Imprime la respuesta del chatbot

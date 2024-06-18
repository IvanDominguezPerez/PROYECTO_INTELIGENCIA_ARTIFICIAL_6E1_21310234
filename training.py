#Creador: Ivan Alejandro Dominguez Perez
#Grupo: 6E1
#Registro: 21310234
#Este archivo esta encargado de crear la red neuronal que se va a encargar de que la IA pueda analizar
#los datos ingresados por el usuario para poder responder al usuario
import random  # Importa el módulo random para operaciones aleatorias
import json  # Importa el módulo json para manejar archivos JSON
import pickle  # Importar para trabajar con archivos que podamos guaradar
import numpy as np  # Importa NumPy para operaciones con arrays

import nltk  # Importa NLTK para procesamiento de lenguaje natural
from nltk.stem import WordNetLemmatizer  # Importa el lematizador de WordNet

# Importa clases y funciones de Keras para construir la red neuronal
from keras.models import Sequential  # Para crear un modelo secuencial
from keras.layers import Dense, Activation, Dropout  # Capas necesarias para el modelo
from keras.optimizers import SGD  # Optimizador Stochastic Gradient Descent

# Inicializa el lematizador de WordNet
lemmatizer = WordNetLemmatizer() #pasar a unos a ceros

# Carga el archivo JSON con los intents (intenciones)
intents = json.loads(open('intents.json').read())

# Descarga los paquetes necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializa listas para palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']  # Caracteres a ignorar

# Procesa cada intent (intención) en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']: #entrar a los patrones que el usuario ingresa
        word_list = nltk.word_tokenize(pattern)  # Tokeniza las frases
        words.extend(word_list)  # Añade las palabras a la lista de palabras
        documents.append((word_list, intent["tag"]))  # Añade a documentos la lista de palabras y su categoría
        if intent["tag"] not in classes:
            classes.append(intent["tag"])  # Añade nuevas clases a la lista de clases

# Lematiza las palabras y elimina caracteres a ignorar, luego ordena y elimina duplicados
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guarda las palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicializa la lista de entrenamiento y una plantilla para las salidas
training = []
output_empty = [0] * len(classes)
for document in documents: 
    bag = []
    word_patterns = document[0] #todas las palabras de la word_list
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] #Poner en minusculas
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # Crea la bolsa de palabras con pallabras en la lista
    output_row = list(output_empty) # 1 si estan en el patron y 0 si no estan
    output_row[classes.index(document[1])] = 1  # Marca la clase correspondiente con un 1
    training.append([bag, output_row])  # Añade la bolsa de palabras y la salida a la lista de entrenamiento

# Mezcla aleatoriamente los datos de entrenamiento
random.shuffle(training)

# Separa los datos de entrenamiento en características (train_x) y etiquetas (train_y)
train_x = []
train_y = []
for i in training:
    train_x.append(i[0])
    train_y.append(i[1])

# Convierte las listas a arrays de NumPy
train_x = np.array(train_x)
train_y = np.array(train_y)

# Crea la estructura de la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))  # Capa de entrada con 128 neuronas
model.add(Dropout(0.5, name="hidden_layer1"))  # Capa de dropout para prevenir sobreajuste
model.add(Dense(64, name="hidden_layer2", activation='relu'))  # Capa oculta con 64 neuronas
model.add(Dropout(0.5, name="hidden_layer3"))  # Otra capa de dropout
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))  # Capa de salida con activación softmax

# Configura el optimizador y compila el modelo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True) #parametros paa optimizar
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 

# Entrena el modelo con los datos de entrenamiento
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guarda el modelo entrenado en un archivo
model.save("chatbot_model.h5")

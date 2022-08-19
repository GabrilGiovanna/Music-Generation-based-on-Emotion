import os
import glob
import pickle
import json
import tensorflow.keras as keras
from keras.utils import np_utils
import numpy
from music21 import *
MIDIPATH= "teste/*.mid"
SEQUENCE_LENGTH = 64
OUTPUT_UNITS = 43 #for the number of neurons in the network
NUM_UNITS = [256]  #number of neurons in the internal layers of the network
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 70
BATCH_SIZE = 64 #Amount of samples that the network is going to see before running back propagation
MODEL_PATH = "model.h5"
STEPS= 133901 // BATCH_SIZE 



def create_data_generator():
    notes=[]
    with open ("notes", "rb") as file:
        notes = pickle.load(file)

    int_songs = []

        # load mappings
    with open("mapping2.json", "r") as fp:
        mappings = json.load(fp)

        # transform songs string to list

        # map songs to int
    for symbol in notes:
        int_songs.append(mappings[symbol])

    seq = SEQUENCE_LENGTH
        
    #int_songs = convert_songs_to_int()

        #the inputs will have a fixed length
        #the targets will be the value of the item that comes after each sequence in the time series
        # [11,12,13,14,15...] -> i: [11,12] t:13
        #Essentially, we're passing some historical information about the musical events and the network predicts the next one in the sequence
        #each training sample will have 64 steps, which equals 4 bars of music(which is usually an entire phrase)
    
    
    
    
    num_sequences = len(int_songs) - seq
    #print(num_sequences)
    def _my_generator():
        i = 0
        
        while True:
            inputs = []
            targets = []
            inputs.append(int_songs[i:i+seq]) #At each step, takes a slice of the list time series, and when i is incremented we move to the right of list time series by one step
            targets.append(int_songs[i+seq])
            vocabulary_size = len(set(int_songs))
            
            
            
        #inputs: (number of sequences, sequence length)
        # one-hot enconding(easiest way to deal with categorical data in neural networks [[0,1,2][1,1,2]] - > [ [ [1,0,0] [0,1,0] [0,0,1] ] , []
    
            inputs = keras.utils.to_categorical(inputs,num_classes= vocabulary_size)
            targets = numpy.array(targets)
            
            
            i = (i+1) % num_sequences
            
            yield inputs,targets
    
    
    
    return _my_generator()

def build_model(output_units, num_units, loss, learning_rate):
    #create the model architecture, using the Functional API(allows to create very complex architectures that have no linear topologies)
    input = keras.layers.Input(shape=(None,output_units))    #input layer, None value in shape enables us to have as many time steps as we want in the network, so we can generate music with whatever length as we want // output_units value basically tells us how many elements we have for each time step
    x = keras.layers.LSTM(num_units[0])(input)    #add another node to our model, to do that we pass another layer to our input, passing the input to this LSTM layer, this is how you add nodes in the Functional API
    x = keras.layers.Dropout(0.2)(x)  #adding another node(Dropout layer) to the model. Dropout is a technique to avoid overfitting. (check tutorial)

    output = keras.layers.Dense(output_units, activation = "softmax")(x)  #check why softmax

    model = keras.Model(input,output)


    #compile model

    model.compile(loss=loss,optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics = ["accuracy"])

    model.summary() #print some info about all the layers of the model, like the relative shape for the inputs and outputs aswell as for all the parameters we have in the model itself.

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    #build the LSTM
    generator = create_data_generator()
    model = build_model(output_units, num_units, loss, learning_rate)  #function to build the LSTM Network

    #train it
    model.fit(x=generator,steps_per_epoch=STEPS,epochs = EPOCHS)


    #save the model

    model.save(MODEL_PATH)
    pass

if __name__ == "__main__":
    train()
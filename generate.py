from training import MODEL_PATH, SEQUENCE_LENGTH
import tensorflow.keras as keras
import json
import numpy as np
import music21

SEQUENCE_LENGTH=64

class MelodyGenerator:
    def __init__(self,model_path="modelHappy.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open("mapping2.json", "r") as fp:
            self.mappings2 = json.load(fp)

        self.start_symbols = ["/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / "] #delimiter of a song



    def generate_music(self, seed,num_steps,max_sequence_length,temperature ): #a seed is a piece of music that we want to put into the network so the network can continue
    
        #create seed with start symbols
        seed = seed.split()
        music = seed
        seed = self.start_symbols + seed
        
        #map seed to int

        seed = [self.mappings2[symbol] for symbol in seed]

        for _ in range(num_steps):
            #limit seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            #one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes= len(self.mappings2)) # (max_sequence_length, number of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...] #adds an extra dimension to the array, in order to make it compatible with the network (1,max_sequence_length, number of symbols in the vocabulary)

            #make a prediction
            prob = self.model.predict(onehot_seed)[0] #because we will only pass one sample
            # [0.1,0.2,0.1,0.6] -> 1 because of softmax
            output_int = self._sample_with_temperature(prob,temperature)

            #update seed

            seed.append(output_int)

            #map int to encoding
            output_symbol = [k for k, v in self.mappings2.items() if v == output_int][0]
            #output_symbol = [self.mappings[symbol] for symbol in output_symbol]

            #check whether we're at the end of a music
            if "/" in output_symbol:
                break

            #update the music

            music.append(output_symbol)
        return music



    def _sample_with_temperature(self,prob,temperature):
        #temperature -> infinity the probability distribution becomes an homogenous distribution
        #temperature -> 0 the probability distribution is remodeled where the original value that had the highest probability, now has 1.0
        #temperature -> 1 the prob distrib doesn't change
        predictions = np.log(prob) / temperature #we take the log of prob and apply the temperature, 
        prob = np.exp(predictions) / np.sum(np.exp(predictions)) #if predictions are small values, by applying the softmax function, the differences between the values of all the items is going to shrink. If predictions have high values, the distribution is gonna be more conservative(values with higher score are much more likely to be picked)
        
        choices = range(len(prob)) # [0,1,2,3]
        index = np.random.choice(choices, p = prob) #to each item of choices, we have a correspondent probability

        return index
    
    def save_music(self,music,step_duration=0.25, format = "midi",file_name="happy.mid"):

        #create a music21 stream

        

        #parse all the symbols in the melody and create note/rest objects
        start_symbol = None 
        step_counter = 1
        stream = music21.stream.Stream()

        final_notes = []

        for i,symbol in enumerate(music):
            #handle case in which we have a note/rest
        
            i = 0
            if (symbol != "_" and ('+' not in symbol)) or (i+1==len(music)): #the Or is if we are at the end of the music
                if start_symbol is not None: #ensure we're dealing with note/rest beyond the first one
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1, therefore the note is a quarter note
                    #handle rest
                    if start_symbol == "r":
                        event = music21.note.Rest(quarterLength=quarter_length_duration)
                        stream.append(event)

                    #handle note
                    else:
                        if ('+' not in start_symbol) and (start_symbol.isdigit()==False) and(start_symbol is not None):
                            #print(start_symbol)
                            #print(start_symbol)
                            event = music21.note.Note(start_symbol,quarterLength=quarter_length_duration)
                            
                            stream.append(event)
                    #final_notes.append(event)

                    step_counter = 1

                start_symbol = symbol
            

            #handle case in which we have a chord
            
            if ('+' in symbol) or symbol.isdigit() or (i+1==len(music)):
                if start_symbol is not None:
                    if ( '+' in start_symbol):
                        quarter_length_duration = step_duration * step_counter
                        notes_in_chord = start_symbol.split('+')
                        temp_notes=[]
                        for note in notes_in_chord:
                            new_note = music21.note.Note(int(note))
                            temp_notes.append(new_note)
                        new_chord = music21.chord.Chord(temp_notes,quarterLength=quarter_length_duration)
                        #final_notes.append(new_chord) #saves chord
                        stream.append(new_chord)
                    
                    step_counter = 1

                start_symbol = symbol




            #handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        
        #stream = music21.stream.Stream(final_notes)
        stream.write(format,fp=file_name)
        print("Music saved in current directory")
        print(stream.analyze("key"))
        
            
            



            
                
            
            





        #write the music21 stream to a midi file

class MelodyGeneratorSad:
    def __init__(self,model_path="modelSad.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open("mapping3.json", "r") as fp:
            self.mappings2 = json.load(fp)

        self.start_symbols = ["/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / "] #delimiter of a song



    def generate_music(self, seed,num_steps,max_sequence_length,temperature ): #a seed is a piece of music that we want to put into the network so the network can continue
    
        #create seed with start symbols
        seed = seed.split()
        music = seed
        seed = self.start_symbols + seed
        
        #map seed to int

        seed = [self.mappings2[symbol] for symbol in seed]

        for _ in range(num_steps):
            #limit seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            #one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes= len(self.mappings2)) # (max_sequence_length, number of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...] #adds an extra dimension to the array, in order to make it compatible with the network (1,max_sequence_length, number of symbols in the vocabulary)

            #make a prediction
            prob = self.model.predict(onehot_seed)[0] #because we will only pass one sample
            # [0.1,0.2,0.1,0.6] -> 1 because of softmax
            output_int = self._sample_with_temperature(prob,temperature)

            #update seed

            seed.append(output_int)

            #map int to encoding
            output_symbol = [k for k, v in self.mappings2.items() if v == output_int][0]
            #output_symbol = [self.mappings[symbol] for symbol in output_symbol]

            #check whether we're at the end of a music
            if "/" in output_symbol:
                break

            #update the music

            music.append(output_symbol)
        return music



    def _sample_with_temperature(self,prob,temperature):
        #temperature -> infinity the probability distribution becomes an homogenous distribution
        #temperature -> 0 the probability distribution is remodeled where the original value that had the highest probability, now has 1.0
        #temperature -> 1 the prob distrib doesn't change
        predictions = np.log(prob) / temperature #we take the log of prob and apply the temperature, 
        prob = np.exp(predictions) / np.sum(np.exp(predictions)) #if predictions are small values, by applying the softmax function, the differences between the values of all the items is going to shrink. If predictions have high values, the distribution is gonna be more conservative(values with higher score are much more likely to be picked)
        
        choices = range(len(prob)) # [0,1,2,3]
        index = np.random.choice(choices, p = prob) #to each item of choices, we have a correspondent probability

        return index
    
    def save_music(self,music,step_duration=0.25, format = "midi",file_name="sad.mid"):

        #create a music21 stream

        

        #parse all the symbols in the melody and create note/rest objects
        start_symbol = None 
        step_counter = 1
        stream = music21.stream.Stream()

        final_notes = []

        for i,symbol in enumerate(music):
            #handle case in which we have a note/rest
        
            i = 0
            if (symbol != "_" and ('+' not in symbol)) or (i+1==len(music)): #the Or is if we are at the end of the music
                if start_symbol is not None: #ensure we're dealing with note/rest beyond the first one
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1, therefore the note is a quarter note
                    #handle rest
                    if start_symbol == "r":
                        event = music21.note.Rest(quarterLength=quarter_length_duration)
                        stream.append(event)

                    #handle note
                    else:
                        if ('+' not in start_symbol) and (start_symbol.isdigit()==False) and(start_symbol is not None):
                            #print(start_symbol)
                            #print(start_symbol)
                            event = music21.note.Note(start_symbol,quarterLength=quarter_length_duration)
                            
                            stream.append(event)
                    #final_notes.append(event)

                    step_counter = 1

                start_symbol = symbol
            

            #handle case in which we have a chord
            
            if ('+' in symbol) or symbol.isdigit() or (i+1==len(music)):
                if start_symbol is not None:
                    if ( '+' in start_symbol):
                        quarter_length_duration = step_duration * step_counter
                        notes_in_chord = start_symbol.split('+')
                        temp_notes=[]
                        for note in notes_in_chord:
                            new_note = music21.note.Note(int(note))
                            temp_notes.append(new_note)
                        new_chord = music21.chord.Chord(temp_notes,quarterLength=quarter_length_duration)
                        #final_notes.append(new_chord) #saves chord
                        stream.append(new_chord)
                    
                    step_counter = 1

                start_symbol = symbol




            #handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        
        #stream = music21.stream.Stream(final_notes)
        stream.write(format,fp=file_name)
        print("Music saved in current directory")
        print(stream.analyze("key"))
        
            
            



            
                
            
            





        #write the music21 stream to a midi file

if __name__ == "__main__":
    testefolk = "E4 _ _ _ F4 _ _ _ _ _ _ _ F4"
    testefolk2 = "F4 _ E4 _ _ D4"
    testefolk3 = "D4 _ _ _ E4 _ E4 _ C4"
    testeSad = "C5 _ _ A3 C4 E4 10+2+5"
    testeSad2 = "10+2+5 _ _ C5 _ _ C4"
    x = int(input("1- Happy\n 2- Sad"))
    if(x == 2):
        gen = MelodyGeneratorSad()
        music = gen.generate_music(testeSad2,500,SEQUENCE_LENGTH,0.7)
    else:
        gen = MelodyGenerator()
        music = gen.generate_music(testefolk2,2000,SEQUENCE_LENGTH,0.7)
        

    
    print(music)
    gen.save_music(music)
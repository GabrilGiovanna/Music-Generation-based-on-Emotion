import os
import glob
import pickle
import json
import tensorflow.keras as keras
import numpy
from music21 import *
import music21 as m21
MIDIPATH= "teste/*"
SEQUENCE_LENGTH = 64

major=0
minor=0

major = 0
minor = 0
# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):#Transposes songs to C Major/C Minor
    
    global major
    global minor
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    keySignature = measures_part0[0][3]

    # estimate key using music21
    if not isinstance(keySignature, m21.key.Key):
        keySignature = song.analyze("key")

    
    # get interval for transposition. E.g., Bmaj -> Cmaj
    if keySignature.mode == "major":
        major = major + 1
        if "C" not in keySignature.tonic.name:
            interval = m21.interval.Interval(keySignature.tonic, m21.pitch.Pitch("C"))
            transposed_song = song.transpose(interval)
        else:
            
            transposed_song = song
    elif keySignature.mode == "minor":
        minor = minor + 1
        if "C" not in keySignature.tonic.name:
            interval = m21.interval.Interval(keySignature.tonic, m21.pitch.Pitch("C"))
            transposed_song = song.transpose(interval)
        else:
            transposed_song = song

    # transpose song by calculated interval
    
    
   
    return transposed_song








# Print the element and its offset (time from starting when it is played)




notes = []
new_song_delimiter = "/ " * SEQUENCE_LENGTH
i = 1
count = 0
for file in glob.glob(MIDIPATH):
    print("Parsing %s..." %file)
    print("---- %d" %i)
    i=i+1
    parsed_file = converter.parse(file)
    if not has_acceptable_durations(parsed_file, ACCEPTABLE_DURATIONS):
        continue

    parsed_file = transpose(parsed_file)
    count = count + 1
    flattened_file = parsed_file.flat.notesAndRests

    for element in flattened_file:
        
        # If the element is a note

        if isinstance(element, note.Note):
            
            symbol = str(element.pitch)

        # If the element is a chord, split it and join the node IDs together into a single sring separated by a '+'
        elif isinstance(element, chord.Chord):
            
            symbol = "+".join(str(n) for n in element.normalOrder)
        #If it's a rest
        elif isinstance(element, note.Rest):
            
            symbol = "r"
        steps = int(element.duration.quarterLength / 0.25)
        
        for step in range(steps): #testar
            if step == 0:
                notes.append(symbol)
            else:
                notes.append("_")
    notesString = " ".join(map(str,notes))
    notesString = notesString + " " + new_song_delimiter
    
    notes.append(" ")
    notes.append(new_song_delimiter)
notesString = notesString[:-1]
    
with open("fileDataset","w") as fp:
    fp.write(notesString)
symbols = sorted(set(notes))
int_to_element = dict ((num, element) for num, element in enumerate (symbols)) #mapping
element_to_int = dict ((element, num) for num, element in int_to_element.items())
with open("mapping.json", "w") as fp: #save mapping in a json file
    json.dump(int_to_element, fp, indent=4)

with open("mapping2.json", "w") as fp: #save mapping in a json file
    json.dump(element_to_int, fp, indent=4)
# Save these notes
with open ("notes", "wb") as filepath:
    pickle.dump (notes, filepath)





print("---------------------------------")
#print(int_songs)
# -*- coding: utf-8 -*-
"""
Created on [Date]

Author: [Your Name]
Description: Augments melodies by transposing pitches while preserving timing 
             and formatting. Outputs each augmented melody on a new line, with
             tokens joined by '|'.
"""

import re
import os

# Define the sequence of notes for transposition (12 semitones in an octave)
NOTES = [
    'C', 'C#', 'D', 'D#', 'E',
    'F', 'F#', 'G', 'G#', 'A',
    'A#','B'
]

# Define duration symbols
DURATION_SYMBOLS = ['w', 'h', 'q', 'e', 's', 't']

# Define the range of octaves (e.g., 0 to 9)
MIN_OCTAVE = 0
MAX_OCTAVE = 9

def transpose_note(note, shift):
    """
    Transpose a single note by a given number of semitones.

    :param note: String, e.g. "C4", "A#3"
    :param shift: Integer, semitone shifts (positive = up, negative = down)
    :return: Transposed note as a string, e.g. "D4", "B3"
    """
    # Regex to parse the note: group(1) = base note (A-G#), group(2) = octave (digit(s))
    match = re.match(r'^([A-G]#?)(\d)$', note)
    if not match:
        return note
    
    base_note, octave_str = match.groups()
    octave = int(octave_str)
    
    try:
        idx = NOTES.index(base_note)
    except ValueError:
        return note
    
    # Calculate new index & octave
    new_index = idx + shift
    new_octave = octave

    while new_index < 0:
        new_index += len(NOTES)
        new_octave -= 1
    while new_index >= len(NOTES):
        new_index -= len(NOTES)
        new_octave += 1
    
    # If new octave out of [MIN_OCTAVE..MAX_OCTAVE], revert to old
    if new_octave < MIN_OCTAVE or new_octave > MAX_OCTAVE:
        return note
    
    return NOTES[new_index] + str(new_octave)

def transpose_token(token, shift):
    """
    Transpose a single token (note or rest + duration).
    e.g. "A#3q" -> "C4q" if shift=3 semitones, or "R..." -> unchanged.
    """
    # R for rest? If starts with 'R' or 'r', treat as rest
    if token.startswith('R') or token.startswith('r'):
        return token
    else:
        # It should be something like "A#3q" or "C4h"
        match = re.match(r'^([A-G]#?\d)([whqest])$', token)
        if match:
            note_part, duration_part = match.groups()
            new_note = transpose_note(note_part, shift)
            return f"{new_note}{duration_part}"
        else:
            return token

def transpose_melody(melody, shift):
    tokens = melody.strip().split()
    # Transpose each token
    transposed_tokens = [transpose_token(tok, shift) for tok in tokens]
    # Join with '|'
    return '|'.join(transposed_tokens)

def augment_melody(original_melody, shifts):
    out = []
    for s in shifts:
        transposed = transpose_melody(original_melody, s)
        out.append(transposed)
    return out

def load_melodies(input_filepath):
    if not os.path.exists(input_filepath):
        print(f"Input file '{input_filepath}' does not exist.")
        return []
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def save_augmented_melodies(augmented_melodies, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for mel in augmented_melodies:
            f.write(mel + '\n')  # each on new line
    print(f"Augmented melodies have been saved to '{output_filepath}'.")



# Input/Output paths
input_file = 'inputMelodies.txt'
output_file = 'inputMelodiesAugmented.txt'
    
# Define semitone shifts
shifts = [1, 2, 3, 4, 5]  # Up by 1..5 semitones
    
# 1) Load the original melodies (one per line)
original_melodies = load_melodies(input_file)
    
# 2) For each original melody, generate all transposed versions
#    and store in augmented_melodies
augmented_melodies = []
for melody in original_melodies:
    transposed_list = augment_melody(melody, shifts)
    # Each transposed version is appended on its own line
    augmented_melodies.extend(transposed_list)
    
# 3) Optionally, you could also re-append the original if needed:
#    augmented_melodies.extend(original_melodies)
    
# 4) Save all augmented versions to a new text file
save_augmented_melodies(augmented_melodies, output_file)
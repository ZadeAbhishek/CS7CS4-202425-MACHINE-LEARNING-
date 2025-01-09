# -*- coding: utf-8 -*-
"""
Author: Giovanni Di Liberto (Modified)
Description: Converts MIDI files to text sequences with octave information and durations,
             and converts text sequences back to MIDI files using those durations.
"""

import os
import re
from mido import MidiFile, MidiTrack, Message, MetaMessage

def generate_midi_note_to_name():
    # 12 semitones in an octave, using sharps
    note_names = [
        'C', 'C#', 'D', 'D#', 'E', 
        'F', 'F#', 'G', 'G#', 'A', 
        'A#','B'
    ]
    midi_note_to_name = {}
    for midi_num in range(128):
        # MIDI note 0 is C-1
        octave = (midi_num // 12) - 1
        name = note_names[midi_num % 12]
        midi_note_to_name[midi_num] = f"{name}{octave}"
    return midi_note_to_name

# Dictionary: midi_num -> "NoteNameOctave"
MIDI_NOTE_TO_NAME = generate_midi_note_to_name()

def map_beats_to_duration(beats):
    
    if beats >= 4.0:
        return 'w'  # Whole note
    elif beats >= 2.0:
        return 'h'  # Half note
    elif beats >= 1.0:
        return 'q'  # Quarter note
    elif beats >= 0.5:
        return 'e'  # Eighth note
    elif beats >= 0.25:
        return 's'  # Sixteenth note
    else:
        return 't'  # Thirty-second note (or smaller)

def map_duration_token_to_ticks(token, ticks_per_beat):
    
    duration_map = {
        'w': 4 * ticks_per_beat,
        'h': 2 * ticks_per_beat,
        'q': 1 * ticks_per_beat,
        'e': ticks_per_beat // 2,
        's': ticks_per_beat // 4,
        't': ticks_per_beat // 8,
    }
    return duration_map.get(token, ticks_per_beat)  # default: 1 beat


def midi_to_text_sequence(midi_path):

    midi = MidiFile(midi_path)
    tpb = midi.ticks_per_beat
    
    # We’ll store for each note_on: 
    #  start_time_in_ticks
    # Then when we see note_off, we measure difference -> note duration.
    # Also track the "gap" from last note_off to next note_on as rest.
    notes_in_progress = {}
    
    # We'll gather events in a single list: (absolute_tick, message)
    events = []
    abs_time = 0
    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            events.append((abs_time, msg))
    
    events.sort(key=lambda e: e[0])
    tokens = []
    last_event_tick = 0
    
    for abs_tick, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            gap_ticks = abs_tick - last_event_tick
            if gap_ticks > 0:
                beats = gap_ticks / tpb
                rest_token = map_beats_to_duration(beats)
                tokens.append(f"R{rest_token}")
            notes_in_progress[msg.note] = abs_tick
            last_event_tick = abs_tick
        
        elif (msg.type == 'note_off' or 
              (msg.type == 'note_on' and msg.velocity == 0)):
            if msg.note in notes_in_progress:
                start_tick = notes_in_progress[msg.note]
                duration_ticks = abs_tick - start_tick
                if duration_ticks < 0:
                    duration_ticks = 0
                beats = duration_ticks / tpb
                dur_token = map_beats_to_duration(beats)
                note_name = MIDI_NOTE_TO_NAME.get(msg.note, None)
                if note_name is not None:
                    tokens.append(f"{note_name}{dur_token}")
            
                del notes_in_progress[msg.note]
                last_event_tick = abs_tick
                
    return ' '.join(tokens)

def text_sequence_to_midi(sequence, output_path, ticks_per_beat=480, tempo=500000):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    tokens = sequence.strip().split()
    current_time = 0
    
    for token in tokens:
        if token.startswith('R'):
            duration_token = token[1:]
            dur_ticks = map_duration_token_to_ticks(duration_token, ticks_per_beat)
            current_time += dur_ticks
        
        else:
            match = re.match(r'^([A-G](?:#)?\d+)([whqest])$', token)
            if match:
                note_str, dur_char = match.groups()
                midi_note = None
                rev_map = {v: k for k, v in MIDI_NOTE_TO_NAME.items()}
                if note_str in rev_map:
                    midi_note = rev_map[note_str]
                else:
                    print(f"Unknown note: {note_str}")
                    continue
                
                dur_ticks = map_duration_token_to_ticks(dur_char, ticks_per_beat)
                track.append(Message('note_on', note=midi_note, velocity=64, time=current_time))
                track.append(Message('note_off', note=midi_note, velocity=64, time=dur_ticks))
                current_time = 0
            else:
                print(f"Invalid token: {token}")
    
    midi.save(output_path)


midi_dir = 'musicDatasetOriginal'
output_dir = 'musicDatasetSimplified'
    
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    text_sequences = []
    
    for file_name in os.listdir(midi_dir):
        if file_name.lower().endswith('.mid') or file_name.lower().endswith('.midi'):
            midi_path = os.path.join(midi_dir, file_name)
            seq = midi_to_text_sequence(midi_path)
            if seq.strip():
                text_sequences.append(seq)
                print(f"\n== Processed: {file_name} ==\n{seq}\n")

    with open("inputMelodies.txt", "w") as f:
        for s in text_sequences:
            f.write(s + "\n")
    print("Wrote all sequences to inputMelodies.txt.")
    
    for i, seq in enumerate(text_sequences, start=1):
        out_path = os.path.join(output_dir, f"reconstructed_{i}.mid")
        text_sequence_to_midi(seq, out_path)
        print(f"Saved: {out_path}")


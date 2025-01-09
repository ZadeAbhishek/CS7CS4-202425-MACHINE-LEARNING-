# -*- coding: utf-8 -*-
"""
Author: Giovanni Di Liberto (Modified)
Description: Converts augmented melodies into .wav audio files, applying short crossfades
             so that one note blends continuously into the next.
"""

import os
import re
import math
import numpy as np
from pydub import AudioSegment

BPM = 120
SAMPLE_RATE = 44100
AMPLITUDE = 0.5

FADE_IN_MS = 10
CROSSFADE_MS = 50

MS_PER_BEAT = 60000.0 / BPM
DURATION_BEATS = {
    'w': 16.0,   # whole note (originally 4.0)
    'h': 8.0,    # half note (originally 2.0)
    'q': 4.0,    # quarter note (originally 1.0)
    'e': 2.0,    # eighth note (originally 0.5)
    's': 1.0,    # sixteenth (originally 0.25)
    't': 0.5,    # thirty-second (originally 0.125)
}

BASE_NOTE_FREQUENCIES = {
    'C': 261.63,
    'C#': 277.18,
    'Db': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'Eb': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'Gb': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'Ab': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'Bb': 466.16,
    'B': 493.88,
}


def beats_to_ms(dur_char: str) -> int:
    """Convert a duration token (e.g. 'q') into ms, based on BPM."""
    beats = DURATION_BEATS.get(dur_char, 1.0)
    return int(round(MS_PER_BEAT * beats))


def calculate_frequency(note_name: str, octave: int) -> float:
    if note_name.upper() == 'R':
        return 0.0
    base_freq = BASE_NOTE_FREQUENCIES.get(note_name, 0.0)
    # e.g. if note_name='C#', base_freq ~277, octave=4 => freq=277
    # if octave=5 => freq=277*(2^(5-4))=554
    return base_freq * (2.0 ** (octave - 4))


def generate_audio_segment(freq: float, duration_ms: int) -> AudioSegment:
    if freq <= 0.0:
        return AudioSegment.silent(duration=duration_ms)

    num_frames = int(SAMPLE_RATE * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_frames, endpoint=False)
    wave = AMPLITUDE * np.sin(2.0 * np.pi * freq * t)

    # convert to 16-bit int
    audio_data = np.int16(wave * 32767)

    seg = AudioSegment(
        audio_data.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    return seg


def parse_token(token: str):

    token = token.strip()
    rest_match = re.match(r'^(R)([whqest])$', token, re.IGNORECASE)
    if rest_match:
        # e.g. "R", "q" => note_name='R', octave=0, dur_char='q'
        _, dur_char = rest_match.groups()
        return ('R', 0, dur_char)

    # else parse normal note pattern: e.g. "C#4q", "Gb3e"
    match = re.match(r'^([A-Ga-g](?:#|b)?)(\d+)([whqest])$', token)
    if match:
        note_str, octave_str, dur_char = match.groups()
        note_str = note_str.capitalize()  # e.g. "c#" => "C#"
        octave_int = int(octave_str)
        return (note_str, octave_int, dur_char)

    print(f"Warning: invalid token '{token}' - skipping.")
    return None


def create_sequence(tokens):
    """Given note tokens, build a single AudioSegment with crossfades for smooth transitions."""
    track = AudioSegment.silent(duration=0)

    for token in tokens:
        parsed = parse_token(token)
        if parsed is None:
            continue
        note_name, octave, dur_char = parsed
        dur_ms = beats_to_ms(dur_char)
        freq = calculate_frequency(note_name, octave)

        seg = generate_audio_segment(freq, dur_ms)
        if dur_ms > FADE_IN_MS > 0:
            seg = seg.fade_in(FADE_IN_MS)

        crossfade_amount = min(CROSSFADE_MS, len(track))
        track = track.append(seg, crossfade=crossfade_amount)

    return track


def process_augmented_melodies(input_filepath, output_directory):
    if not os.path.exists(input_filepath):
        print(f"Cannot find input file: {input_filepath}")
        return

    os.makedirs(output_directory, exist_ok=True)

    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        tokens = line.split('|')

        final_song = create_sequence(tokens)
        out_filename = f"melody_shift_{idx}.midi"
        out_path = os.path.join(output_directory, out_filename)
        final_song.export(out_path, format="midi")
        print(f"Exported: {out_path}")



model_input_file = "model_generated.txt"
baseline_input_file = "trigram_model_generated.txt"
model_input_file_output_dir = "model_generated_melodies"
baseline_input_file_output_dir = "baseline_generated_melodies"
process_augmented_melodies(model_input_file, model_input_file_output_dir)
process_augmented_melodies(baseline_input_file, baseline_input_file_output_dir)

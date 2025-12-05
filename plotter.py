import os
import music21
import librosa
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from synctoolbox.feature.utils import  estimate_tuning
from analyzer import compute_kde, cents_to_note_name, extract_pitch_from_audio_filtered

def plot_intonation_histogram_in_cents_from_musicxml(file_path, out_path = 'plots'):
    # Parse the MusicXML file
    score = music21.converter.parse(file_path)
    # Get song name from the file path
    song_name = file_path.split('\\')[-2]  # Extract the file name without extension
    print(f"Processing MusicXML file: {file_path} for song: {song_name}")
    # Extract pitches as MIDI numbers
    pitches = []
    for note in score.recurse().notes:
        if note.isNote:  # Single note
            pitches.append(note.pitch.midi)
    # Convert MIDI numbers to note names
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    range_semitones = max_pitch - min_pitch
    # Convert MIDI pitches to cents relative to A4 (440 Hz)
    key = score.analyze('key')
    tonic_frequency = key.tonic.frequency  # Reference frequency for key
    cents = [1200 * np.log2(librosa.midi_to_hz(pitch) / tonic_frequency) for pitch in pitches]
    # normalize the cents values
    cents = np.array(cents)
    cents = np.mod(cents, 1200)  # Wrap around to [0, 1200)
    # Compute KDE
    grid, kde_values = compute_kde(cents)

    # Calculate the area under the curve (AUC)
    auc = simpson(kde_values, grid)

    # Normalize the KDE values
    kde_values /= auc
    # Convert grid values to note names
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)
    # Enhance plot aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    plt.plot(grid, kde_values, color="royalblue", lw=2, label="Normalized KDE")
    plt.fill_between(grid, kde_values, color="royalblue", alpha=0.3)
    plt.xlabel("Cents (relative to tonic)")
    plt.title("Pitch KDE from MusicXML Score for " + song_name)
    plt.axvline(0, color='gray', linestyle='--', label=key)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Set x-axis ticks to show all values in cents

    plt.xticks(np.arange(0, 1200, 100), note_labels) # Label x-axis with note names
    plt.legend()
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = os.path.join(out_path, song_name)
    plt.savefig(out_path + ".png")
    return tonic_frequency


def plot_intonation_histogram_in_cents_from_performance(audio_file_path, tonic_frequency):
    audio, fs = librosa.load(audio_file_path, sr=44100)
    # Step 1: Convert tuning offset to Hz
    # Convert anchor candidates.frequency to .. hz to cents
    tuning_cents = estimate_tuning(audio, fs)
    f_ref = tonic_frequency * 2 ** (tuning_cents / 1200)
    # use crepe or basic pitch to extract intonation in cents from the audio file
    #student_file_name = os.path.basename(audio_file_path).split('.')[0]  # Extract the file name without extension
    # Python

    cents = extract_pitch_from_audio_filtered(audio_file_path, f_ref=f_ref)
    # KDE values
    grid, kde_values = compute_kde(cents)

    # Calculate the area under the curve (AUC)
    auc = simpson(kde_values, grid)

    # Normalize the KDE values
    kde_values /= auc
    # Convert grid values to note names
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    plt.plot(grid, kde_values, color="royalblue", lw=2, label="Normalized KDE")
    plt.fill_between(grid, kde_values, color="royalblue", alpha=0.3)
    plt.xlabel("Cents (relative to tonic)")
    plt.title(f"Pitch KDE (Tuning offset: {tuning_cents:+.1f} cents) for student {audio_file_path}")
    plt.axvline(0, color='gray', linestyle='--', label='Tuning Center')
    #plt.grid(True, linestyle='--', alpha=0.7)
    # Set x-axis ticks to show all values in cents

    plt.xticks(np.arange(0, 1200, 100), note_labels) # Label x-axis with note names
    #plt.xlim(0, 1200)  # Limit x-axis to [0, 1200] cents
    plt.legend()
    plt.savefig(f'{audio_file_path}.png', dpi=400)

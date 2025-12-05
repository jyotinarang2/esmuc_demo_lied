# The idea is to compute intonation histograms from scores and from performances using standard fundamental frequency
# detection converted to midi frequency and then to pitch class.
import os
import librosa
import music21
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
from scipy.integrate import simpson
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, skew, spearmanr, kurtosis, pearsonr
from synctoolbox.feature.utils import estimate_tuning


def extract_pitch_from_audio_filtered(
    audio_path: str,
    f_ref: float = 440.0,
    tuning_cents: float = 0.0,
    fmin_note: str = "C2",
    fmax_note: str = "C7",
    prob_thresh: float = 0.8,
    smooth_thresh: float = 0.5,
    medfilt_kernel: int = 15,
    hop_length: int = 256,          # adjust as you like
    sr: int | None = None           # None -> native sr
):
    """
    Mirror CREPE-style filtering on pYIN output:
    - Smooth voiced_prob with median filter
    - Keep frames with prob > prob_thresh and smoothed > smooth_thresh
    - Drop non-positive f0
    - Keep pitches within a MIDI range
    - Convert to cents (relative to f_ref) and wrap to [0, 1200)
    Returns: cents (np.ndarray)
    """
    y, sr = librosa.load(audio_path, sr=sr)
    print(sr)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        hop_length=hop_length
    )


    # Replace NaNs with 0 for filtering steps; keep a mask for final selection
    f0 = np.asarray(f0)
    voiced_prob = np.asarray(voiced_prob)
    valid = ~np.isnan(f0)
    f0_filled = np.where(valid, f0, 0.0)

    # Smooth the probability like you did with CREPE confidence
    # (Ensure kernel is odd and >= 1)
    k = medfilt_kernel if medfilt_kernel % 2 == 1 else medfilt_kernel + 1
    smoothed = medfilt(voiced_prob, kernel_size=k)

    # Candidate frames (mirror your thresholds)
    keep = (
        (voiced_prob > prob_thresh) &
        (smoothed > smooth_thresh) &
        (f0_filled > 0.0)
    )

    # MIDI range filter (same idea as with CREPE)
    midi = np.zeros_like(f0_filled)
    print(midi)
    nonzero = f0_filled > 0
    midi[nonzero] = np.round(librosa.hz_to_midi(f0_filled[nonzero])).astype(int)

    min_pitch = int(np.round(librosa.note_to_midi(fmin_note)))
    max_pitch = int(np.round(librosa.note_to_midi(fmax_note)))
    keep &= (midi >= min_pitch) & (midi <= max_pitch)

    # Final selection
    f0_kept = f0_filled[keep]
    if f0_kept.size == 0:
        return np.array([])
    cents_from_ref = (1200.0 * np.log2(f0_kept / f_ref)) % 1200.0
    print(cents_from_ref)
    # Convert to cents relative to f_ref, apply tuning offset, wrap
    #cents = 1200.0 * np.log2(f0_kept / f_ref) - float(tuning_cents)
    cents = np.mod(cents_from_ref, 1200.0)
    return cents

def cents_to_note_name(cents, tonic_frequency):
    # Convert cents to frequency
    frequencies = tonic_frequency * (2 ** (cents / 1200))
    # Convert frequencies to MIDI numbers
    midi_numbers = np.round(librosa.hz_to_midi(frequencies)).astype(int)
    # Convert MIDI numbers to note names
    return [music21.pitch.Pitch(midi).name for midi in midi_numbers]

def compute_kde(cents):
    """Return normalized KDE over cents in [0,1200)."""
    # temp figure just to let seaborn compute the KDE
    fig_kde, ax_kde = plt.subplots()
    sns.kdeplot(cents, bw_adjust=0.5, ax=ax_kde)
    line = ax_kde.get_lines()[-1]   # use LAST line (current data)
    grid, kde_values = line.get_data()
    plt.close(fig_kde)

    # normalize area
    auc = simpson(kde_values, grid)
    if auc != 0:
        kde_values = kde_values / auc
    return grid, kde_values


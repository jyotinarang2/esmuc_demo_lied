"""
Intonation Analysis Visualizer
Web-based implementation with score and audio selection
"""

from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import pandas as pd
import threading
import queue
import time
import json
import os
import glob
import hashlib
from datetime import datetime
import librosa
from analyzer import extract_pitch_from_audio_filtered, compute_kde
from plotter import plot_intonation_histogram_in_cents_from_musicxml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from synctoolbox.feature.utils import estimate_tuning

app = Flask(__name__)

# Global state
result_queue = queue.Queue()
CACHE_DIR = 'cache'
PLOTS_DIR = 'plots'

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_file_hash(filepath):
    """Generate a hash for cache identification."""
    return hashlib.md5(filepath.encode()).hexdigest()


def get_cached_audio_analysis(audio_path, tonic_frequency):
    """Load cached audio analysis if available."""
    cache_key = f"{get_file_hash(audio_path)}_{tonic_frequency:.2f}"
    cache_file = os.path.join(CACHE_DIR, f"audio_{cache_key}.csv")

    if os.path.exists(cache_file):
        print(f"✓ Cache hit: Loading cached audio analysis for {os.path.basename(audio_path)}")
        df = pd.read_csv(cache_file)
        return df['timestamp'].values, df['cents'].values

    print(f"✗ Cache miss: No cached audio analysis for {os.path.basename(audio_path)}")
    return None, None


def save_audio_analysis_cache(audio_path, tonic_frequency, timestamps, cents):
    """Save audio analysis to cache."""
    cache_key = f"{get_file_hash(audio_path)}_{tonic_frequency:.2f}"
    cache_file = os.path.join(CACHE_DIR, f"audio_{cache_key}.csv")

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cents': cents
    })
    df.to_csv(cache_file, index=False)
    print(f"Saved analysis cache to {cache_file}")


def get_cached_score_analysis(xml_path):
    """Load cached score analysis if available."""
    cache_key = get_file_hash(xml_path)
    cache_file = os.path.join(CACHE_DIR, f"score_{cache_key}.json")

    if os.path.exists(cache_file):
        print(f"✓ Cache hit: Loading cached score analysis for {os.path.basename(xml_path)}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    print(f"✗ Cache miss: No cached score analysis for {os.path.basename(xml_path)}")
    return None


def save_score_analysis_cache(xml_path, tonic_frequency, cents):
    """Save score analysis to cache."""
    cache_key = get_file_hash(xml_path)
    cache_file = os.path.join(CACHE_DIR, f"score_{cache_key}.json")

    data = {
        'tonic_frequency': float(tonic_frequency),
        'cents': [float(c) for c in cents]
    }

    with open(cache_file, 'w') as f:
        json.dump(data, f)
    print(f"Saved score cache to {cache_file}")


def analyze_audio_with_timestamps(audio_path, tonic_frequency):
    """Extract pitch from audio and return timestamps and cents."""
    # Check cache first
    cached_times, cached_cents = get_cached_audio_analysis(audio_path, tonic_frequency)
    if cached_times is not None:
        return cached_times, cached_cents

    print(f"Analyzing audio: {audio_path}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100)

    # Estimate tuning
    tuning_cents = estimate_tuning(audio, sr)
    f_ref = tonic_frequency * 2 ** (tuning_cents / 1200)

    # Extract pitch
    hop_length = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=hop_length
    )

    # Filter and process
    prob_thresh = 0.8
    from scipy.signal import medfilt
    smoothed = medfilt(voiced_prob, kernel_size=15)

    valid = (
        ~np.isnan(f0) &
        (voiced_prob > prob_thresh) &
        (smoothed > 0.5) &
        (f0 > 0)
    )

    # Calculate timestamps
    timestamps = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # Filter valid data
    valid_times = timestamps[valid]
    valid_f0 = f0[valid]

    if len(valid_f0) == 0:
        return np.array([]), np.array([])

    # Convert to cents
    cents = (1200.0 * np.log2(valid_f0 / f_ref)) % 1200.0

    # Save to cache
    save_audio_analysis_cache(audio_path, tonic_frequency, valid_times, cents)

    return valid_times, cents


def background_processor(xml_path, audio_path, thread_id):
    """Background thread function that processes the score and audio."""
    try:
        print(f"Starting analysis for thread {thread_id}")

        # Analyze score
        score_cache = get_cached_score_analysis(xml_path)
        if score_cache:
            tonic_frequency = score_cache['tonic_frequency']
            score_cents = np.array(score_cache['cents'])
        else:
            tonic_frequency = plot_intonation_histogram_in_cents_from_musicxml(
                xml_path, out_path=PLOTS_DIR
            )
            import music21
            score = music21.converter.parse(xml_path)
            pitches = []
            for note in score.recurse().notes:
                if note.isNote:
                    pitches.append(note.pitch.midi)

            cents = [1200 * np.log2(librosa.midi_to_hz(pitch) / tonic_frequency) for pitch in pitches]
            score_cents = np.array(cents) % 1200
            save_score_analysis_cache(xml_path, tonic_frequency, score_cents)

        # Analyze audio
        timestamps, audio_cents = analyze_audio_with_timestamps(audio_path, tonic_frequency)

        # Compute KDEs
        score_grid, score_kde = compute_kde(score_cents)
        audio_grid, audio_kde = compute_kde(audio_cents)

        # Normalize KDEs
        from scipy.integrate import simpson
        score_kde /= simpson(score_kde, score_grid)
        audio_kde /= simpson(audio_kde, audio_grid)

        # Create separate plots
        score_plot_path = generate_score_plot(
            score_grid, score_kde,
            tonic_frequency,
            os.path.basename(xml_path)
        )

        audio_plot_path = generate_audio_plot(
            audio_grid, audio_kde,
            tonic_frequency,
            os.path.basename(audio_path)
        )

        # Send result
        result = {
            'thread_id': thread_id,
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'xml_path': xml_path,
            'audio_path': audio_path,
            'tonic_frequency': float(tonic_frequency),
            'score_plot_path': score_plot_path,
            'audio_plot_path': audio_plot_path,
            'audio_timestamps': timestamps.tolist(),
            'audio_cents': audio_cents.tolist(),
            'score_cents': score_cents.tolist(),
            'score_grid': score_grid.tolist(),
            'score_kde': score_kde.tolist(),
            'audio_grid': audio_grid.tolist(),
            'audio_kde': audio_kde.tolist(),
            'audio_duration': float(timestamps[-1]) if len(timestamps) > 0 else 0
        }

        result_queue.put(result)
        print(f"Analysis complete for thread {thread_id}")

    except Exception as e:
        print(f"Error in background processor: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'thread_id': thread_id,
            'error': str(e)
        })


def generate_score_plot(score_grid, score_kde, tonic_frequency, score_name):
    """Generate plot for score KDE."""
    from analyzer import cents_to_note_name

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax.plot(score_grid, score_kde, color="royalblue", lw=2, label="Score KDE")
    ax.fill_between(score_grid, score_kde, color="royalblue", alpha=0.3)
    ax.set_xlabel("Cents (relative to tonic)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Score Intonation: {score_name}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1200)
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)
    ax.set_xticks(np.arange(0, 1200, 100))
    ax.set_xticklabels(note_labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = f"score_{int(time.time())}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_filename


def generate_audio_plot(audio_grid, audio_kde, tonic_frequency, audio_name):
    """Generate plot for audio KDE."""
    from analyzer import cents_to_note_name

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax.plot(audio_grid, audio_kde, color="coral", lw=2, label="Audio KDE")
    ax.fill_between(audio_grid, audio_kde, color="coral", alpha=0.3)
    ax.set_xlabel("Cents (relative to tonic)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Audio Intonation: {audio_name}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1200)
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)
    ax.set_xticks(np.arange(0, 1200, 100))
    ax.set_xticklabels(note_labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = f"audio_{int(time.time())}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_filename


def generate_combined_plot(score_grid, score_kde, audio_grid, audio_kde,
                          tonic_frequency, score_name, audio_name):
    """Generate combined plot of score and audio KDEs."""
    from analyzer import cents_to_note_name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    # Score plot
    ax1.plot(score_grid, score_kde, color="royalblue", lw=2, label="Score KDE")
    ax1.fill_between(score_grid, score_kde, color="royalblue", alpha=0.3)
    ax1.set_xlabel("Cents (relative to tonic)")
    ax1.set_title(f"Score: {score_name}")
    ax1.set_xlim(0, 1200)
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)
    ax1.set_xticks(np.arange(0, 1200, 100))
    ax1.set_xticklabels(note_labels)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Audio plot
    ax2.plot(audio_grid, audio_kde, color="coral", lw=2, label="Audio KDE")
    ax2.fill_between(audio_grid, audio_kde, color="coral", alpha=0.3)
    ax2.set_xlabel("Cents (relative to tonic)")
    ax2.set_title(f"Audio: {audio_name}")
    ax2.set_xlim(0, 1200)
    ax2.set_xticks(np.arange(0, 1200, 100))
    ax2.set_xticklabels(note_labels)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = f"combined_{int(time.time())}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_filename


def generate_region_plot(timestamps, cents, tonic_frequency, start_time, end_time, name,
                        show_score_overlay=False, score_grid=None, score_kde=None):
    """Generate plot for a specific time region with optional score overlay."""
    # Filter data for time range
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    region_cents = cents[mask]

    if len(region_cents) == 0:
        return None

    # Compute KDE
    grid, kde_values = compute_kde(region_cents)
    from scipy.integrate import simpson
    kde_values /= simpson(kde_values, grid)

    # Plot
    from analyzer import cents_to_note_name
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Plot audio KDE
    ax.plot(grid, kde_values, color="coral", lw=2.5, label="Audio Region KDE", zorder=3)
    ax.fill_between(grid, kde_values, color="coral", alpha=0.3, zorder=2)

    # Optionally overlay score KDE
    if show_score_overlay and score_grid is not None and score_kde is not None:
        score_grid = np.array(score_grid)
        score_kde = np.array(score_kde)
        ax.plot(score_grid, score_kde, color="royalblue", lw=2,
               label="Score KDE (reference)", linestyle='--', alpha=0.7, zorder=2)
        ax.fill_between(score_grid, score_kde, color="royalblue", alpha=0.15, zorder=1)

    ax.set_xlabel("Cents (relative to tonic)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    title = f"{name} - Region: {start_time:.2f}s to {end_time:.2f}s"
    if show_score_overlay:
        title += " (with Score Overlay)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1200)
    note_labels = cents_to_note_name(np.arange(0, 1200, 100), tonic_frequency)
    ax.set_xticks(np.arange(0, 1200, 100))
    ax.set_xticklabels(note_labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save plot
    plot_filename = f"region_{int(time.time())}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_filename


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/list_files')
def list_files():
    """List available XML scores and audio files."""
    xml_files_base_dir = 'data/scores'
    audio_files_base_dir = 'data/audio'

    xml_files = glob.glob(os.path.join(xml_files_base_dir, "**/*.xml"), recursive=True)
    audio_files = []

    # Find audio files (wav, mp3, flac)
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(glob.glob(os.path.join(audio_files_base_dir, "**", ext), recursive=True))

    return jsonify({
        'xml_files': xml_files,
        'audio_files': audio_files
    })


@app.route('/api/audio_waveform')
def get_audio_waveform():
    """Get audio waveform data for visualization."""
    audio_path = request.args.get('path')

    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'Invalid audio path'}), 400

    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100)

    # Downsample for visualization
    max_points = 5000
    if len(audio) > max_points:
        indices = np.linspace(0, len(audio) - 1, max_points, dtype=int)
        downsampled = audio[indices].tolist()
    else:
        downsampled = audio.tolist()

    return jsonify({
        'audio_data': downsampled,
        'total_samples': int(len(audio)),
        'sample_rate': int(sr),
        'duration': float(len(audio) / sr)
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start analysis in background."""
    data = request.json
    xml_path = data.get('xml_path')
    audio_path = data.get('audio_path')
    thread_id = data.get('thread_id', str(time.time()))

    if not xml_path or not audio_path:
        return jsonify({'error': 'Missing paths'}), 400

    if not os.path.exists(xml_path) or not os.path.exists(audio_path):
        return jsonify({'error': 'Files not found'}), 404

    # Check if both are already cached
    score_cache = get_cached_score_analysis(xml_path)
    if score_cache:
        tonic_frequency = score_cache['tonic_frequency']
        audio_cache_times, audio_cache_cents = get_cached_audio_analysis(audio_path, tonic_frequency)

        if audio_cache_times is not None and audio_cache_cents is not None:
            # Both cached - return immediately without spawning thread
            print(f"Both score and audio are cached - loading from cache")

            # Compute KDEs from cached data
            score_cents = np.array(score_cache['cents'])
            score_grid, score_kde = compute_kde(score_cents)
            audio_grid, audio_kde = compute_kde(audio_cache_cents)

            # Normalize KDEs
            from scipy.integrate import simpson
            score_kde /= simpson(score_kde, score_grid)
            audio_kde /= simpson(audio_kde, audio_grid)

            # Generate plots
            score_plot_path = generate_score_plot(
                score_grid, score_kde,
                tonic_frequency,
                os.path.basename(xml_path)
            )

            audio_plot_path = generate_audio_plot(
                audio_grid, audio_kde,
                tonic_frequency,
                os.path.basename(audio_path)
            )

            # Create result
            result = {
                'thread_id': thread_id,
                'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                'xml_path': xml_path,
                'audio_path': audio_path,
                'tonic_frequency': float(tonic_frequency),
                'score_plot_path': score_plot_path,
                'audio_plot_path': audio_plot_path,
                'audio_timestamps': audio_cache_times.tolist(),
                'audio_cents': audio_cache_cents.tolist(),
                'score_cents': score_cents.tolist(),
                'score_grid': score_grid.tolist(),
                'score_kde': score_kde.tolist(),
                'audio_grid': audio_grid.tolist(),
                'audio_kde': audio_kde.tolist(),
                'audio_duration': float(audio_cache_times[-1]) if len(audio_cache_times) > 0 else 0,
                'cached': True
            }

            # Put result directly in queue
            result_queue.put(result)

            return jsonify({
                'status': 'complete',
                'thread_id': thread_id,
                'message': 'Loaded from cache',
                'cached': True
            })

    # Not fully cached - spawn background thread
    thread = threading.Thread(
        target=background_processor,
        args=(xml_path, audio_path, thread_id),
        daemon=True
    )
    thread.start()

    return jsonify({
        'status': 'processing',
        'thread_id': thread_id,
        'message': 'Analysis started in background',
        'cached': False
    })


@app.route('/api/get_results')
def get_results():
    """Poll for results from background threads."""
    results = []

    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            results.append(result)
        except queue.Empty:
            break

    return jsonify({'results': results})


@app.route('/api/update_region', methods=['POST'])
def update_region():
    """Generate plot for selected time region."""
    data = request.json
    timestamps = np.array(data.get('timestamps', []))
    cents = np.array(data.get('cents', []))
    tonic_frequency = data.get('tonic_frequency')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    name = data.get('name', 'Audio')
    show_score_overlay = data.get('show_score_overlay', False)
    score_grid = data.get('score_grid')
    score_kde = data.get('score_kde')

    if len(timestamps) == 0 or len(cents) == 0:
        return jsonify({'error': 'No data'}), 400

    plot_filename = generate_region_plot(
        timestamps, cents, tonic_frequency,
        start_time, end_time, name,
        show_score_overlay, score_grid, score_kde
    )

    if plot_filename is None:
        return jsonify({'error': 'No data in region'}), 400

    return jsonify({'plot_path': plot_filename})


@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images."""
    return send_file(os.path.join(PLOTS_DIR, filename))


def find_available_port(start_port=5001, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            # Port is in use, try next one
            continue

    # If no port found, raise error
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")


if __name__ == '__main__':
    print("\nStarting Intonation Analysis Visualizer...")

    # Find available port
    try:
        port = find_available_port(start_port=5001, max_attempts=10)
        print(f"✓ Found available port: {port}")
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("Please free up some ports and try again.")
        exit(1)

    print(f"Open your browser to: http://localhost:{port}")
    print("\nFeatures:")
    print("- Select score (XML) and audio files from dropdowns")
    print("- Automatic caching for faster repeated analysis")
    print("- View side-by-side intonation plots")
    print("- Select time regions to analyze specific sections\n")

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
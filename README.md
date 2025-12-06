# Intonation Analysis Visualizer

A web-based application for analyzing and comparing musical intonation between scores (MusicXML) and audio performances.

## Features

- **File Selection**: Choose from available MusicXML scores and audio files via dropdown menus
- **Automatic Caching**: Analysis results are cached for faster repeated access
- **Side-by-Side Comparison**: View score and audio intonation plots together
- **Interactive Region Selection**: Select specific time regions in the audio waveform for detailed analysis
- **Timestamped Data**: All pitch data includes timestamps for precise temporal analysis

## Installation

### Prerequisites

```bash
pip install flask numpy pandas matplotlib seaborn librosa music21 scipy synctoolbox
```

### Directory Structure

Create the following directories in your project root:

```
esmuc_demo_lied/
├── app.py
├── analyzer.py
├── plotter.py
├── templates/
│   └── index.html
├── data/
│   ├── scores/          # Place MusicXML (.xml) files here
│   └── audio/           # Place audio files (.wav, .mp3, .flac) here
├── cache/               # Auto-created for cached analysis
└── plots/               # Auto-created for plot images
```

## Usage

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5001`

### 2. Using the Web Interface

1. **Select Files**:
   - Choose a MusicXML score from the first dropdown
   - Choose an audio file from the second dropdown

2. **Run Analysis**:
   - Click the "Analyze" button
   - Wait for the analysis to complete (progress shown in status box)

3. **View Results**:
   - Full analysis shows side-by-side plots of score and audio intonation
   - Information panel displays tonic frequency, duration, and data counts

4. **Analyze Regions**:
   - Click and drag on the waveform to select a time region
   - Drag the edges to adjust the selection boundaries
   - Drag the middle to move the entire selection
   - The region plot updates automatically showing intonation for that time segment

## How It Works

### Analysis Pipeline

1. **Score Analysis**:
   - Parses MusicXML file using `music21`
   - Extracts pitches as MIDI numbers
   - Determines key and tonic frequency
   - Converts to cents relative to tonic
   - Computes normalized KDE (Kernel Density Estimation)

2. **Audio Analysis**:
   - Loads audio file with `librosa`
   - Estimates tuning offset
   - Extracts fundamental frequency using pYIN
   - Filters by confidence and range
   - Converts to cents relative to tonic
   - Generates timestamps for each pitch frame
   - Computes normalized KDE

3. **Caching**:
   - Score analysis cached by file hash: `cache/score_{hash}.json`
   - Audio analysis cached by file hash and tonic: `cache/audio_{hash}_{tonic}.csv`
   - Includes timestamps for temporal analysis

4. **Visualization**:
   - Combined plots show full score vs audio comparison
   - Region plots dynamically generated for selected time segments
   - All plots normalized to show probability density

## Cache Format

### Score Cache (JSON)
```json
{
  "tonic_frequency": 440.0,
  "cents": [0, 200, 400, ...]
}
```

### Audio Cache (CSV)
```csv
timestamp,cents
0.011609,523.45
0.017414,525.12
...
```

## API Endpoints

- `GET /` - Main web interface
- `GET /api/list_files` - List available XML and audio files
- `GET /api/audio_waveform?path=<path>` - Get audio waveform data
- `POST /api/analyze` - Start background analysis
- `GET /api/get_results` - Poll for analysis results
- `POST /api/update_region` - Generate plot for time region
- `GET /plots/<filename>` - Serve plot images

## Customization

### Adjust Analysis Parameters

Edit `analyzer.py`:
- `fmin_note` / `fmax_note`: Pitch detection range
- `prob_thresh`: Voice probability threshold
- `hop_length`: Time resolution (samples)
- `medfilt_kernel`: Smoothing kernel size

### Modify Plot Appearance

Edit functions in `app.py`:
- `generate_combined_plot()`: Full analysis plots
- `generate_region_plot()`: Region-specific plots

### Change Port

In `app.py`, modify the last line:
```python
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
```

## Troubleshooting

### "No data in region"
- The selected time region may not contain valid pitch data
- Try selecting a longer region or adjusting analysis thresholds

### Analysis Taking Too Long
- First run analyzes both files (may take 30-60 seconds)
- Subsequent runs use cached data (much faster)
- Check console output for progress messages

### Files Not Appearing in Dropdowns
- Ensure files are in `data/scores/` or `data/audio/` directories
- Supported formats: `.xml` for scores, `.wav/.mp3/.flac` for audio
- Check file permissions

### Cache Not Working
- Ensure `cache/` directory exists and is writable
- Cache is based on file path and tonic frequency
- Delete cache files to force re-analysis

## Performance Tips

1. **Use WAV files**: Faster to load than MP3/FLAC
2. **Clear old cache**: Delete unused files in `cache/` directory
3. **Adjust hop_length**: Larger values = faster processing but lower time resolution
4. **Downsample audio**: Use lower sample rates for faster analysis

## Credits

Built using:
- Flask (web framework)
- librosa (audio analysis)
- music21 (music notation)
- matplotlib/seaborn (visualization)
- synctoolbox (tuning estimation)
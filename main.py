import glob
import os
from plotter import plot_intonation_histogram_in_cents_from_musicxml

def main():
    xml_files_base_dir = 'data/scores'
    xml_files = glob.glob(os.path.join(xml_files_base_dir, "**/*.xml"), recursive=True)
    print(xml_files)
    # Plot intonation histogram from the MusicXML file
    for xml_file in xml_files:
        print(f"Processing file: {xml_file}")
        plot_intonation_histogram_in_cents_from_musicxml(xml_file)
    # Plot intonation histogram from the performance given the key or tonic frequency


if __name__ == "__main__":
    main()

import sys
import os

# Ajouter le dossier parent au PYTHONPATH pour trouver le package "grand"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grand_analysis.data_processor import DataProcessor
from grand_analysis.visualizer import Visualizer
from grand_analysis.spectrum_analyzer import SpectrumAnalyzer

def main():
    # Set the data directory and file list.
    data_dir = "/home/nlebas/grand/examples/grandlib_classes/"
    file_list = [data_dir + "GP80_20250309_235256_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root"]
    
    # Process the data.
    processor = DataProcessor(file_list)
    processor.process_files()
    
    # Visualize different aspects of the data.
    viz = Visualizer(processor)
    # viz.plot_geolocation()
    viz.plot_du_histogram()
    # viz.plot_multiplicity_histogram()
    # viz.plot_event_timing(r_exp=2.0)
    
    # Perform spectrum analysis.
    spec = SpectrumAnalyzer(processor, dt=2e-9)  # dt de 2 ns pour 500 MHz
    spec.visualize_mean_fft(channels=[1], xlim=(0, 250), min_spectra=100,
                          apply_notch=True, f_sample=500e6)

    
if __name__ == "__main__":
    main()

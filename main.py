"""
Problems to solve: - The choose of the right channel should be an array for all methode
                   - The methode to see the baseline doesn't have enough time to see

"""

import sys
import os

import sys
sys.path.append("/home/nlebas/grand")


# Ajouter le dossier parent au PYTHONPATH pour trouver le package "grand"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grand_analysis.data_processor import DataProcessor
from grand_analysis.visualizer import Visualizer
from grand_analysis.spectrum_analyzer import SpectrumAnalyzer
from grand_analysis.trigger_analyzer import TriggerAnalyzer
from grand_analysis.cosmic_ray_reconstructor import CosmicRayReconstructor
import time


def main():
    start_time = time.time()
    
    # Set the data directory and file list.

    # file_list_dir = "/home/nlebas/Documents/LPNHE/GRAND/Local_Data/"
    # file_list = [data_dir + "GP80_20250228_233748_RUN10069_CD_20dB_7DUs_GP80_t3Test-CD-10000-22.root"]
    # Répertoire où se trouve le fichier texte contenant la liste.
    file_list_dir = "/sps/grand/nlebas/grand/Grand_analysis/"
    
    # Répertoire contenant les fichiers de données.
    data_files_dir = "/sps/grand/data/gp80/GrandRoot/2025/02/"
    # data_files_dir = "/home/nlebas/Documents"
    # Chemin complet du fichier texte.
    file_list_path = os.path.join(file_list_dir, "data_202502_20_28.txt")

    with open(file_list_path, "r") as f:
        sorted_file_list = [
            os.path.normpath(os.path.join(data_files_dir, line.strip()))
            for line in f if line.strip()
        ]

    valid_file_list = []
    for fname in sorted_file_list:
        if os.path.exists(fname):
            valid_file_list.append(fname)
        else:
            print("Fichier introuvable :", fname)

    if not valid_file_list:
        raise RuntimeError("Aucun fichier valide à traiter !")

                
    # Process the data.
    processor = DataProcessor(file_list = valid_file_list)
    processor.process_files()
    
    # Visualize different aspects of the data.
    viz = Visualizer(processor, dt=2e-9)
    # viz.plot_geolocation()
    # viz.visualize_event(target_du=1046, evtid=23, channels=[1,2])
    # viz.plot_du_histogram()
    # viz.plot_multiplicity_histogram()
    viz.plot_time_trigger()
    
    # Perform spectrum analysis.
    spec = SpectrumAnalyzer(processor, dt=2e-9)  # dt de 2 ns pour 500 MHz
    # spec.visualize_mean_fft(channels=[1], xlim=(0, 250), min_spectra=100, apply_notch=False, only_galacti_noise=True , f_sample=500e6, kadc=1.8/16384, R=50)
    # spec.analyze_baseline_vs_time(channel=1, freq_band=(60, 80), du=1049 ,  apply_notch=False, galactic_noise=True, f_sample=500e6, kadc=1.8/16384, R=50, fit_sine=True)

    # Perform trigger analysis.
    # trigpat = TriggerAnalyzer(processor, viz, dt=2e-9)
    # trigpat.plot_histograms_trigger_counts()
    # trigpat.plot_trigger_rate_map()

    
    # Visualiser les histogrammes de chi2
    end_time = time.time()

    print("Temps d'exécution : {:.3f} secondes".format(end_time - start_time))
    
if __name__ == "__main__":
    main()

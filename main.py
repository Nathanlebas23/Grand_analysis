import sys
import os

sys.path.append("/home/nlebas/grand")
sys.path.append("sps/grand/nlebas/grand/")

# Ajouter le dossier parent au PYTHONPATH pour trouver le package "grand"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grand_analysis.data_processor import DataProcessor
from grand_analysis.visualizer import Visualizer
from grand_analysis.spectrum_analyzer import SpectrumAnalyzer
from grand_analysis.trigger_analyzer import TriggerAnalyzer
from grand_analysis.reconstructor import Reconstructor
import time
import numpy as np


def main():
    start_time = time.time()
    
    # Répertoire où se trouve le fichier texte contenant la liste.
    file_list_dir = "/home/nlebas/Documents/LPNHE/GRAND/Local_Data/list_files/"
    # file_list_dir = "/sps/grand/nlebas/grand/Grand_analysis/"
    
    # Répertoire contenant les fichiers de données.
    # data_files_dir = "/sps/grand/data/gp80/GrandRoot/2025/03/"
    data_files_dir = "/home/nlebas/Documents/LPNHE/GRAND/Local_Data/data_files/MD_145_files/"
    
    # Chemin complet du fichier texte.
    file_list_path = os.path.join(file_list_dir, "data_test.txt")

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
            pass
            # print("Fichier introuvable :", fname)

    if not valid_file_list:
        raise RuntimeError("Aucun fichier valide à traiter !")


    test_file = ['/home/nlebas/grand/Grand_analysis/GP80_20250204_152548_RUN145_MD_RAW-ChanXYZ-20dB-GP43-20hz-0024(1).root']
                
    # Process the data.
    processor = DataProcessor(file_list = test_file)
    processor.process_files()
    
    # Visualize different aspects of the data.
    viz = Visualizer(processor, dt=2e-9)
    # viz.plot_geolocation()
    # viz.visualize_event(target_du=1046, evtid=12, channels=[0,1])
    # viz.plot_du_histogram()
    # viz.plot_multiplicity_histogram()
    # # viz.plot_event_timing()

    
    # Perform spectrum analysis.
    spec = SpectrumAnalyzer(processor, viz, dt=2e-9)  # dt de 2 ns pour 500 MHz
    spec.visualize_PSD(channels=[1], xlim=(0, 250), min_spectra=100, apply_notch=False, only_galacti_noise=False , f_sample=500e6, kadc=1.8/16384, R=50)
    # spec.analyze_baseline_vs_time(channel=0, freq_band=(60, 80), du=1046 ,  apply_notch=False, galactic_noise=False, f_sample=500e6, kadc=1.8/16384, R=50, fit_sine=True)
    # spec.histo_amplitude_all_event_norm(amplitude_type='hilbert', bins=100)
    # spec.diag_average_amplitude_per_du(amplitude_type='hilbert')
    # spec.diag_std_per_du(amplitude_type='hilbert')
    # spec.amplitude_map_X()
    # spec.normalized_average_amplitude_map(channel=[0,1,2,3])
    # spec.normalized_std_map(channel=[0,1,2,3])
    

    # Perform trigger analysis.
    # trigpat = TriggerAnalyzer(processor, viz, dt=2e-9)
    # trigpat.plot_histograms_trigger_counts()
    # trigpat.plot_trigger_rate_map()


    # Perform reconstruction 
    # recon = Reconstructor(processor, viz, event_index = 8)
    # theta_pred, phi_pred = recon.reconstruct()
    # theta_pred, phi_pred = recon.reconstruct_all_events()
    # print('θ_pred =', np.round(theta_pred*180/np.pi, 2), '°')
    # print("ϕ_pred =", np.round(phi_pred*180/np.pi, 2), '°')
    # recon.plot_3D_sphere()
    # recon.plot_2D_sphere()

    end_time = time.time()

    print("Execution time : {:.3f} seconds".format(end_time - start_time))
    
if __name__ == "__main__":
    main()
"""
- Ajout de la SWF
- Ajout de cut (theta_min et theta_max)

    # histo de la valeur par event par DU
    # qqc ---> encontion du temps 
    # Moyenne sur la carte 
"""


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
from grand_analysis.DeltaT_Analyzer import DeltaTAnalyzer
import time
import numpy as np
import time
import os


def main():
    """
    Main function for processing, visualizing, reconstructing, and analyzing GRAND data.
    
    The procedure is as follows:
      1. Define the directories and input file list.
      2. Process the data files.
      3. Visualize various data aspects.
      4. Perform spectrum analysis.
      5. Reconstruct events and display distributions of chi², theta, and phi.
      6. Analyze trigger data.
      7. Report the total execution time.

    Note: Many function calls are commented out by default. Uncomment the desired
    lines to perform specific analyses.
    """
    
    # Record the start time of execution.
    start_time = time.time()
    
    # =============================================================================
    # 1. Set Directories and Data Files
    # =============================================================================
    # Directory where the file list (text file containing a list of data file paths) is located.
    file_list_dir = "/sps/grand/nlebas/grand/Grand_analysis/"
    
    # # Directory containing the raw data files.
    data_files_dir = "/sps/grand/data/gp80/GrandRoot/2025/03/"
    
    # Optionally, you can load a list of data file paths from a text file.
    # (The following code is commented out; adjust as needed.)

    

    # file_list_path = os.path.join(file_list_dir, "data_2303.txt")
    # with open(file_list_path, "r") as f:
    #     sorted_file_list = [
    #         os.path.normpath(os.path.join(data_files_dir, line.strip()))
    #         for line in f if line.strip()
    #     ]
    
    # valid_file_list = [fname for fname in sorted_file_list if os.path.exists(fname)]
    # if not valid_file_list:
    #     raise RuntimeError("No valid data files to process!")
    
    # task_id    = int(sys.argv[1])                                    
    # n_tasks    = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    # chunks     = np.array_split(valid_file_list, n_tasks)
    # my_files   = chunks[task_id].tolist()
    # print(f"[TASK {task_id}/{n_tasks}] Processing {len(my_files)} files")


    # For testing purposes, we use a single data file.
    test_file = [
        '/home/nlebas/Documents/LPNHE/GRAND/Local_Data/data_files/GP80_20250323_000113_RUN10074_CD_20dB_GP43-14DUs-ChY-Y2FLOAT-X2Z-CD-10000-389.root'
    ]
    
    # =============================================================================
    # 2. Process the Data
    # =============================================================================
    # Create an instance of DataProcessor using the test file(s) and process the data.
    processor = DataProcessor(file_list= test_file)
    processor.process_files()
    # processor.verify(print_every=100)  


    # =============================================================================
    # 3. Visualize Data Aspects
    # =============================================================================
    # Create a Visualizer with a time resolution of 2 ns.
    viz = Visualizer(processor, dt=2e-9)
    # viz.get_event_indices()
    # Uncomment the following lines to perform various visualizations:
    viz.plot_geolocation()                # Plot the geographic locations of detectors.
    # viz.build_distance_matrix()           # Build and display a distance matrix between detectors.
    # viz.visualize_event(target_du=1046, evtid=12, channels=[0, 1])
    # viz.plot_du_histogram()               # Histogram of Digital Units.
    # viz.plot_du_histogram_duplicate()               # Histogram of Digital Units.
    # viz.plot_time_trigger()


    # viz.plot_multiplicity_histogram()     # Histogram of event multiplicities.
    # viz.plot_trigger_vs_time_comparison() # Compare trigger times against time.
    # viz.plot_deltaT_histogram(bins=100) # Histogram of time differences.
    # viz.get_deltaT(bins=100)
    # viz.get_causal_event()
    
    # =============================================================================
    # 4. Spectrum Analysis
    # =============================================================================
    # Instantiate the SpectrumAnalyzer for frequency analysis.
    # spec = SpectrumAnalyzer(processor, viz, dt=2e-9) 
    # # Uncomment the following lines for spectrum analysis:
    # spec.visualize_PSD(channels=[1], xlim=(0, 250), min_spectra=100, apply_notch=False, 
    #                    only_pre_trigger=False, f_sample=500e6, kadc=1.8/16384, R=50)
    # spec.analyze_baseline_vs_time(channel=[1], freq_band=(0, 250), du=None, apply_notch=False, 
    #                              only_pre_trigger=False, f_sample=500e6, kadc=1.8/16384, R=50, fit_sine=True)
    # spec.histo_amplitude_all_event_norm(amplitude_type='hilbert', bins=100)
    # spec.diag_average_amplitude_per_du(amplitude_type='hilbert')
    # spec.diag_std_per_du(amplitude_type='hilbert')
    # spec.amplitude_map_X()
    # spec.normalized_average_amplitude_map(channel=[0, 1, 2, 3])
    # spec.normalized_std_map(channel=[0, 1, 2, 3])
    # # spec.amplitude_vs_time(channel=[0, 1, 2, 3], bin_width=100)
    # spec.analyse_mean_amplitude_vs_time(axis=1, only_pre_trigger=False)
    # spec.analyse_std_baseline_vs_time(axis=1, only_pre_trigger=False)
    
    # =============================================================================
    # 5. Perform Reconstruction
    # =============================================================================
    # Create a Reconstructor instance. Set rec_model to 'PWF' or 'SWF' and event_index to None
    # to process all events.
    # recon = Reconstructor(processor, viz, rec_model='PWF', nb_min_antennas=4, event_index=None)
    
    # Uncomment one of the following options:
    # results = recon.reconstruct()  # Process and return reconstruction results for all events.
    # recon.distrib_chi2_thetaphi()     # Plot distributions of chi², theta, and phi.
    # recon.plot_3D_sphere()         # 3D scatter plot of reconstructed directions on a sphere.
    # recon.plot_2D_sphere()         # 2D polar projection of the reconstructed directions.
    # recon.plot_2D_sphere_chi2()      # 2D polar plot with chi² color mapping.
    # recon.histo_thetaphi()         # 2D histogram of theta and phi values.
    # recon.distrib_chi2_thetaphi()  # Plot distributions of chi², theta, and phi.
    # =============================================================================
    # 6. Analyze Trigger Data
    # =============================================================================
    # Instantiate the TriggerAnalyzer for trigger-related analysis.
    # trigpat = TriggerAnalyzer(processor, recon, viz, dt=2e-9)
    
    # # Uncomment the desired trigger analysis functions:
    # trigpat.plot_histograms_trigger_counts()
    # trigpat.plot_trigger_rate_map()
    # trigpat.trigger_vs_time(bin_width=5)
    # trigpat.trigger_count_vs_time(bin_width=2)
    # trigpat.analyze_trigger_delays(event_index=None)  # Use None to analyze and animate all events.
    # # trigpat.histo_chi2()
    


    # =============================================================================
    # 7. DeltaT Analysis
    # =============================================================================

    # ana = DeltaTAnalyzer(processor, viz, dt=2e-9)
    #     # Charge votre Parquet et trace les histogrammes
    # ana = DeltaTAnalyzer('all_events_byfilename.parquet')
    # ana.verify()
    # # ana.plot_deltaT_zoom(bins=1000, window_ns=3)
    # # ana.plot_duplicate_rate_vs_distance()
    # # ana.plot_mse_duplicates()
    # ana.plot_duplicate_histogram(bins=100)
    # ana.visualize_event(1082, 274964, channels=[1, 2], f_sample=None)
    # ana.plot_deltaT_histogram(bins=1000)


    # =============================================================================
    # 8. Report Execution Time
    # =============================================================================
    end_time = time.time()
    print("Execution time: {:.3f} seconds".format(end_time - start_time))
    
if __name__ == "__main__":
    main()

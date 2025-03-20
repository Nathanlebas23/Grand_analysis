import os
import numpy as np
import grand.dataio as rt

class DataProcessor:
    """
    Class to process ROOT data files and extract event information.
    """
    def __init__(self, file_list):
        self.file_list = file_list
        
        # Initialize containers for event data.
        self.trigger_times = []   # in seconds
        self.trigger_nanos = []   # in nanoseconds
        self.traces = []          # list of traces per event: each trace is a list of [raw, x, y, z]
        self.du_ids = []          # detection unit IDs (one per trigger)
        self.du_long = []         # GPS longitude per trigger
        self.du_lat = []          # GPS latitude per trigger
        self.du_alt = []          # GPS altitude per trigger
        self.multiplicities = []  # event multiplicity (number of DU triggers in the event)
        self.trigger_pattern_ch = []  # trigger pattern for each channel
        
    def process_files(self):
        """
        Process each file in the file list, load the ROOT data,
        and extract the necessary information for each event.
        
        For each event, loop over all activated detection units (DUs) so that no information is lost.
        """
        for fname in self.file_list:
            root_file = rt.DataFile(fname)
            n_entries = root_file.tadc.get_number_of_entries()
            
            for i in range(n_entries):
                # Retrieve ADC and raw voltage data for the current event.
                root_file.tadc.get_entry(i)
                root_file.trawvoltage.get_entry(i)
                
                # Check trigger conditions:
                # - At least one False in trigger_pattern_10s.
                # - At least one True in trigger_pattern_ch.
                if ~np.all(root_file.tadc.trigger_pattern_10s) and np.any(root_file.tadc.trigger_pattern_ch):
                    
                    # Skip events with abnormal GPS time (using first element pour le test)
                    if root_file.tadc.du_seconds[0] > 2045518457:
                        continue
                    
                    # Determine the number of triggered detection units for the event.
                    multiplicity = len(root_file.tadc.du_id)
                    
                    # Pour chaque DU activée dans l'événement, on enregistre toutes les informations.
                    for j in range(multiplicity):
                        self.multiplicities.append(multiplicity)
                        self.trigger_times.append(root_file.tadc.du_seconds[j])
                        self.trigger_nanos.append(root_file.tadc.du_nanoseconds[j])
                        self.du_ids.append(root_file.tadc.du_id[j])
                        self.du_long.append(root_file.trawvoltage.gps_long[j])
                        self.du_lat.append(root_file.trawvoltage.gps_lat[j])
                        self.du_alt.append(root_file.trawvoltage.gps_alt[j])
                        self.trigger_pattern_ch.append(root_file.tadc.trigger_pattern_ch[j])

                        
                        # Extraction des traces pour le DU j.
                        # On suppose ici que root_file.tadc.trace_ch est organisé tel que le premier indice
                        # correspond à l'indice de DU et le second indice aux différents canaux [raw, x, y, z].
                        event_traces = []
                        for ch in range(4):
                            event_traces.append(root_file.tadc.trace_ch[j][ch])
                        self.traces.append(event_traces)
                        
        # Conversion des listes en tableaux NumPy pour faciliter le traitement ultérieur.
        self.trigger_times = np.array(self.trigger_times)
        self.trigger_nanos = np.array(self.trigger_nanos)
        self.du_ids = np.array(self.du_ids)
        self.du_long = np.array(self.du_long)
        self.du_lat = np.array(self.du_lat)
        self.du_alt = np.array(self.du_alt)
        self.multiplicities = np.array(self.multiplicities)
        self.traces = np.array(self.traces)
        self.trigger_pattern_ch = np.array(self.trigger_pattern_ch)
    
    def load_data(self):
        """
        Charge les différents fichiers de reconstruction.
        Les fichiers attendus dans recons_path sont :
          - coord_antennas.txt
          - Rec_plane_wave_recons.txt
          - Rec_sphere_wave_recons.txt
          - Rec_coinctable.txt
          - DU_id.txt
        """
        # Chargement des coordonnées des antennes
        coord_path = os.path.join(self.recons_path, "coord_antennas.txt")
        self.idant, self.xant, self.yant, self.zant = np.loadtxt(coord_path).T

        # Reconstruction par onde plane
        plane_path = os.path.join(self.recons_path, "Rec_plane_wave_recons.txt")
        plane_data = np.loadtxt(plane_path).T
        # On récupère ici theta, phi et chi2_plane (indices 2, 4 et 6)
        self.theta_rec = plane_data[2]
        self.phi_rec = plane_data[4]
        self.chi2_plane = plane_data[6]

        # Reconstruction par onde sphérique
        sphere_path = os.path.join(self.recons_path, "Rec_sphere_wave_recons.txt")
        sphere_data = np.loadtxt(sphere_path).T
        # On récupère ici : recoinc, nant, chi2_sphere, xrec, yrec, zrec, trec, rho, ground_alt
        self.recoinc   = sphere_data[0]
        self.nant      = sphere_data[1]
        self.chi2_sphere = sphere_data[2]
        self.xrec      = sphere_data[4]
        self.yrec      = sphere_data[5]
        self.zrec      = sphere_data[6]
        self.trec      = sphere_data[7]
        self.rho       = sphere_data[8]
        self.ground_alt = sphere_data[9]

        # Table de coïncidences
        coinctable_path = os.path.join(self.recons_path, "Rec_coinctable.txt")
        coinctable = np.loadtxt(coinctable_path).T
        # On récupère : ind, coincid, peakt, peaka
        self.ind = coinctable[0]
        self.coincid = coinctable[1]
        self.peakt = coinctable[2]
        self.peaka = coinctable[3]

        # Fichier DU_id (on suppose que la deuxième colonne contient l'ID)
        duid_path = os.path.join(self.recons_path, "DU_id.txt")
        self.antid = np.loadtxt(duid_path, usecols=1).T
    
    def compute_true_time(self):
        """
        Compute the true time (in nanoseconds) for each event.
        """
        return self.trigger_times * 1e9 + self.trigger_nanos

    def get_unique_du(self):
        """
        Return the unique detection unit IDs and the indices of their first occurrence.
        """
        unique_du, idx = np.unique(self.du_ids, return_index=True)
        return unique_du, idx
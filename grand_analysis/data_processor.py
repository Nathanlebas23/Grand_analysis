import os
import numpy as np
import grand.dataio as rt 
# import grand.dataio.root_trees as rt  # Version CC

# Reegarder les 

class DataProcessor:
    """
    Class to process ROOT data files and extract event information.
    """
    def __init__(self, file_list = None):
        self.file_list = file_list

        # Initialize containers for event data.
        self.trigger_times = []   # in seconds
        self.trigger_nanos = []   # in nanoseconds
        self.traces = []          # list of traces per event: each trace is a list of [raw, x, y, z]
        self.du_ids = []          # detection unit IDs (one per trigger)
        self.du_long = []         # GPS longitude per trigger
        self.du_lat = []          # GPS latitude per trigger
        self.du_alt = []          # GPS altitude per trigger
        self.multiplicities = []  # event multiplicity for each trigger  (number of DU triggers in the event) /!\ Should be diferent DU !
        self.mult = []            # multiplicity of the event
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
            print(f"Loading file:", fname)

            for i in range(int(n_entries)) :
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
                multiplicity = len(root_file.tadc.du_id) # Give the DUs triggered in the event
                self.mult.append(multiplicity)
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


                    event_traces = []
                    for ch in range(4):
                        event_traces.append(root_file.tadc.trace_ch[j][ch])
                    self.traces.append(event_traces)
                        
        self.trigger_times = np.array(self.trigger_times)
        self.trigger_nanos = np.array(self.trigger_nanos)
        self.du_ids = np.array(self.du_ids)
        self.du_long = np.array(self.du_long)
        self.du_lat = np.array(self.du_lat)
        self.du_alt = np.array(self.du_alt)
        self.multiplicities = np.array(self.multiplicities)
        
        self.traces = np.array(self.traces, dtype = object)
        self.trigger_pattern_ch = np.array(self.trigger_pattern_ch)



    def verify(self, print_every: int = 100):
        """
        Affiche des informations toutes les `print_every` itérations pour vérifier les données.

        :param print_every: fréquence d'affichage (par défaut toutes les 100 itérations)
        """
        # Calculer une seule fois le tableau des temps en ns et le temps relatif en s
        true_times = self.compute_true_time()                    # tableau en nanosecondes
        t0 = np.min(true_times)                                 # référence
        time_s = (true_times - t0) / 1e9                        # passage en secondes

        n = len(self.du_ids)
        
        if not (n == len(true_times) == len(self.multiplicities)):
            raise ValueError(
                "Incohérence de longueurs : du_ids, true_times et multiplicities doivent être identiques."
            )

        for i in range(n):
            print(
                f"Itération {i:>6d} | DU ID = {self.du_ids[i]} | "
                f"Temps = {true_times[i]} ns ({time_s[i]:.4f} s) | "
                f"Multiplicité = {self.multiplicities[i]} | "
                f"GPS = (lat={self.du_lat[i]:.2f}, lon={self.du_long[i]:.2f}, alt={self.du_alt[i]:.2f} m)"
            )


    def compute_true_time(self):
        """
        Compute the true time (in nanoseconds) for each event.
        """
        return (self.trigger_times * 1e9 + self.trigger_nanos)

    def get_unique_du(self):
        """
        Return the unique detection unit IDs and the indices of their first occurrence.
        """
        unique_du, idx = np.unique(self.du_ids, return_index=True)
        return unique_du, idx
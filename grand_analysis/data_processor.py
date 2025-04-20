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
        self._trigger_secs = []   # in seconds
        self._trigger_nanos = []   # in nanoseconds
        self._traces = []          # list of traces per event: each trace is a list of [raw, x, y, z]
        self._du_ids = []          # detection unit IDs (one per trigger)
        self._du_long = []         # GPS longitude per trigger
        self._du_lat = []          # GPS latitude per trigger
        self._du_alt = []          # GPS altitude per trigger
        self._multiplicities = []  # event multiplicity for each trigger  (number of DU triggers in the event) /!\ Should be diferent DU !
        self.mult = []            # multiplicity of the event
        self._trigger_pattern_ch = []  # trigger pattern for each channel
    
    def group_event(self, quantity):
        cum_mult = list(self.cum_mult)
        return [quantity[low:high] for low, high in zip( [0] + cum_mult[:-1], cum_mult)]
    
    @property
    def trigger_secs(self):
        return self.group_event(self._trigger_secs)
    
    @property
    def trigger_nanos(self):
        return self.group_event(self._trigger_nanos)
    
    @property
    def traces(self):
        return self.group_event(self._traces)
    
    @property
    def du_ids(self):
        return self.group_event(self._du_ids)
    
    @property
    def du_long(self):
        return self.group_event(self._du_long)
    
    @property
    def du_lat(self):
        return self.group_event(self._du_lat)
    
    @property
    def multiplicities(self):
        return self.group_event(self._multiplicities)
    
    @property
    def trigger_pattern_ch(self):
        return self.group_event(self._trigger_pattern_ch)
    
    @property
    def trigger_time(self):
        t_time = self._trigger_secs.astype(np.float64) + self._trigger_nanos.astype(np.float64) * 1e-9
        return self.group_event(t_time)
    

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
                if ~np.all(root_file.tadc.trigger_pattern_10s) and np.any(root_file.tadc._trigger_pattern_ch):
                    # Skip events with abnormal GPS time (using first element pour le test)
                    if root_file.tadc.du_seconds[0] > 2045518457:
                        continue


                    
                    
                # Determine the number of triggered detection units for the event.
                multiplicity = len(root_file.tadc.du_id) # Give the DUs triggered in the event
                self.mult.append(multiplicity)
                
                # Pour chaque DU activée dans l'événement, on enregistre toutes les informations.
                for j in range(multiplicity):
                    self._multiplicities.append(multiplicity)
                    self._trigger_secs.append(root_file.tadc.du_seconds[j])
                    self._trigger_nanos.append(root_file.tadc.du_nanoseconds[j])
                    self._du_ids.append(root_file.tadc.du_id[j])
                    self._du_long.append(root_file.trawvoltage.gps_long[j])
                    self._du_lat.append(root_file.trawvoltage.gps_lat[j])
                    self._du_alt.append(root_file.trawvoltage.gps_alt[j])
                    self._trigger_pattern_ch.append(root_file.tadc._trigger_pattern_ch[j])


                    event_traces = []
                    for ch in range(4):
                        event_traces.append(root_file.tadc.trace_ch[j][ch])
                    self._traces.append(event_traces)
                        
        self._trigger_secs = np.array(self._trigger_secs)
        self._trigger_nanos = np.array(self._trigger_nanos)
        self._du_ids = np.array(self._du_ids)
        self._du_long = np.array(self._du_long)
        self._du_lat = np.array(self._du_lat)
        self._du_alt = np.array(self._du_alt)
        self._multiplicities = np.array(self._multiplicities)
        
        self._traces = np.array(self._traces, dtype = object)
        self._trigger_pattern_ch = np.array(self._trigger_pattern_ch)
        self.cum_mult = np.cumsum(self.mult)

    def compute_true_time(self):
        """
        Compute the true time (in nanoseconds) for each event.
        """
        return (self._trigger_secs * 1e9 + self._trigger_nanos) - min(self._trigger_secs)

    def get_unique_du(self):
        """
        Return the unique detection unit IDs and the indices of their first occurrence.
        """
        unique_du, idx = np.unique(self._du_ids, return_index=True)
        return unique_du, idx
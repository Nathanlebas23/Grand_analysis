import os
import numpy as np
import grand.dataio as rt 
import grand.dataio.root_files as gdr

from scipy.signal import butter, filtfilt, lfilter, hilbert
# import grand.dataio.root_trees as rt  # Version CC

# Reegarder les 
def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """
    Apply a Butterworth bandpass filter to the input data.

    This function filters the input signal `data` using a Butterworth bandpass filter
    with the specified low and high cutoff frequencies. The filter can be configured
    as either causal or non-causal.

    Args:
        data (array-like): The input signal to be filtered.
        lowcut (float): The low cutoff frequency of the bandpass filter in Hz.
        highcut (float): The high cutoff frequency of the bandpass filter in Hz.
        fs (float): The sampling frequency of the input signal in Hz.

    Returns:
        array-like: The filtered signal.

    Notes:
        - The filter order is set to 6.
        - The function uses `lfilter` for causal filtering. Uncomment the `filtfilt`
          line to use non-causal filtering instead.
    """
    b, a = butter(6, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")  # (order, [low, high], btype)

    # return filtfilt(b, a, data)  # non-causal
    return lfilter(b, a, data)  # causal

class DataProcessor:
    """
    Class to process ROOT data files and extract event information.
    """
    def __init__(self, 
                 file_list = None, pre_trigger=False, sampling_freq = 500e6, 
                 apply_notch=False,
                 kadc=1.8/16384, VGA_gain=100, R=50, freq_band_MHz = (None, None)):
        self.file_list = file_list
        self.pre_trigger = pre_trigger
        # Initialize containers for event data.
        self._trigger_secs = []   # in seconds
        self._trigger_nanos = []   # in nanoseconds
        # self.traces = []
        self._du_ids = []          # detection unit IDs (one per trigger)
        self._du_long = []         # GPS longitude per trigger
        self._du_lat = []          # GPS latitude per trigger
        self._du_alt = []          # GPS altitude per trigger
        self._multiplicities = []  # event multiplicity for each trigger  (number of DU triggers in the event) /!\ Should be diferent DU !
        self.mult = []            # multiplicity of the event
        self._trigger_pattern_ch = []  # trigger pattern for each channel
        self._file_idx = []
        self._event_idx = []

        self._traces_std = []
        self._traces_max = []
        self.dt = 1/sampling_freq
        self.apply_notch = apply_notch
        self.kadc = kadc
        self.VGA_gain = VGA_gain
        self.R = R
        self.fre_band_MHz = freq_band_MHz

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
    def _trigger_time(self):
        return self._trigger_secs.astype(np.float128) + self._trigger_nanos.astype(np.float128) * 1e-9
    
    @property
    def trigger_time(self):
        return self.group_event(self._trigger_time)
    

    def process_files(self):
        """
        Process each file in the file list, load the ROOT data,
        and extract the necessary information for each event.
        
        For each event, loop over all activated detection units (DUs) so that no information is lost.
        """
        for f_idx, fname in enumerate(self.file_list):
            root_file = rt.DataFile(fname)
            n_entries = root_file.tadc.get_number_of_entries()
            print(f"Loading file:", fname)

            for entry in range(int(n_entries)) :
            
                # Retrieve ADC and raw voltage data for the current event.
                root_file.tadc.get_entry(entry)
                root_file.trawvoltage.get_entry(entry)
                

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


                    trace = root_file.tadc.trace_ch[j]
                    self._traces_std.append(self.compute_std(trace))
                    self._traces_max.append(self.compute_max(trace))

                    # event_traces = []
                    # for ch in range(4):
                    #     event_traces.append(root_file.tadc.trace_ch[j][ch])
                    # self._traces.append(event_traces)

                    self._file_idx.append(f_idx)
                    self._event_idx.append(entry)
            root_file.close()

        self._trigger_secs = np.array(self._trigger_secs)
        self._trigger_nanos = np.array(self._trigger_nanos)
        self._du_ids = np.array(self._du_ids)
        self._du_long = np.array(self._du_long)
        self._du_lat = np.array(self._du_lat)
        self._du_alt = np.array(self._du_alt)
        self._multiplicities = np.array(self._multiplicities)
        self._file_idx=np.array(self._file_idx)
        self._event_idx=np.array(self._event_idx)
        # self._traces = np.array(self._traces, dtype = object)
        self._trigger_pattern_ch = np.array(self._trigger_pattern_ch)
        self.cum_mult = np.cumsum(self.mult)

        self._traces_std = np.array(self._traces_std)
        self._traces_max = np.array(self._traces_max) 

        self.compute_psd()

    def compute_std(self, trace):
        if not((self.fre_band_MHz[0] is None) ^ (self.fre_band_MHz[1] is None)):
            if self.fre_band_MHz[0] is not(None):
                lower_band = self.fre_band_MHz[0]
                upper_band = self.fre_band_MHz[1]
                trace = _butter_bandpass_filter(trace, lower_band, upper_band, 1/self.dt*1e-6) 
        else:
            raise TypeError
        
        if self.pre_trigger:
            traces_std = (trace[:,:250].astype(np.float32).std(axis=1))
        else:
            traces_std = (trace[:,:].astype(np.float32).std(axis=1))
        return traces_std
    
    def compute_max(self, trace):
        if not((self.fre_band_MHz[0] is None) ^ (self.fre_band_MHz[1] is None)):
            if self.fre_band_MHz[0] is not(None):
                lower_band = self.fre_band_MHz[0]
                upper_band = self.fre_band_MHz[1]
                trace = _butter_bandpass_filter(trace, lower_band, upper_band, 1/self.dt*1e-6) 
        else:
            raise TypeError

        if self.pre_trigger:
            traces_max = (np.abs(trace[:,:250].astype(np.float32)).max(axis=1))
        else:
            traces_max = (np.abs(trace[:,:].astype(np.float32)).max(axis=1))
        return traces_max
    
    def compute_psd(self):
        # Set sampling frequency if not provided

        # Retrieve unique detection units (DUs) in the current file.
        
        psd_per_du = dict()
        nspetra_per_du = dict()
        # Process each DU in the current file.    

        for f_idx, fname in enumerate(self.file_list):
            root_file = rt.DataFile(fname)
            n_entries = root_file.tadc.get_number_of_entries()
            for entry in range(int(n_entries)) :
                root_file.tadc.get_entry(entry)

                du_ids = root_file.tadc.du_id
                for j, du_id in enumerate(du_ids):
                    trace = root_file.tadc.trace_ch[j]
                    npts = 250 if self.pre_trigger else trace.shape[-1]
                    if npts > trace.shape[-1]:
                        print(f"Warning: Requested {npts} points but only {trace.shape[-1]} available. Truncating to available size.")
                        npts = trace.shape[-1]
                                # Calculate frequency resolution and generate frequency arrays.
                    delta_nu = 1 / (npts * self.dt)  # Frequency bin width in Hz
                    freq_hz = np.fft.rfftfreq(npts, self.dt)
                    freq_MHz = freq_hz / 1e6
                    window = np.blackman(npts)
                    power_window = np.sum(window ** 2) / npts

                    if self.apply_notch:
                        for filt_id in [1, 2, 3, 4]:
                            trace = self.apply_notch_filter(trace, filt_id, 1/self.dt)

                    trace_win = trace * window
                    fft_val = np.fft.rfft(trace_win)
                    psd_event_du = np.abs(fft_val) ** 2 / power_window

                    if du_id in psd_per_du:
                        psd_per_du[du_id] += psd_event_du
                        nspetra_per_du[du_id] += 1
                    else:
                        psd_per_du[du_id] = psd_event_du
                        nspetra_per_du[du_id] = 1
            root_file.close()

        lower_band = 0 if self.fre_band_MHz[0] is None else self.fre_band_MHz[0]
        upper_band = freq_MHz[-1] if self.fre_band_MHz[1] is None else self.fre_band_MHz[1]
        in_band = (freq_MHz>=lower_band) & (freq_MHz<= upper_band)
        for key in psd_per_du:
            # Calculate the mean PSD for this file.
            psd_per_du[key] /= nspetra_per_du[key]
            psd_per_du[key] *= (self.kadc ** 2) * self.VGA_gain  # Convert ADC² to V²

            # Normalize the PSD: divide by (N² * delta_nu) and then convert V²/Hz to V²/MHz.
            psd_per_du[key] /= (npts ** 2 * delta_nu) * self.R
            psd_per_du[key] *= 1e6
            psd_per_du[key][:, 1:-1] *= 2
            psd_per_du[key][:, ~in_band] = 0
        self.freq_MHz = freq_MHz
        self.cumulative_psd = psd_per_du
        self.nspectra_per_du = nspetra_per_du


    
    def apply_notch_filter(self, trace, which_filter, f_sample):
        """
        Apply a digital notch filter to a 1D trace.

        Parameters:
        - trace (ndarray): 1D time trace (ADC counts).
        - which_filter (int): Notch filter number (1, 2, 3, or 4).
        - f_sample (float): Sampling frequency in Hz.
        
        Returns:
        - Filtered trace (ndarray).
        """
        if which_filter == 1:
            f_notch = 39e6
            r = 0.9
        elif which_filter == 2:
            f_notch = 119.4e6
            r = 0.94
        elif which_filter == 3:
            f_notch = 132e6
            r = 0.95
        elif which_filter == 4:
            f_notch = 137.8e6
            r = 0.98
        else:
            raise ValueError("which_filter must be 1, 2, 3, or 4.")
        
        nu = 2. * np.pi * f_notch / f_sample

        # Calculate filter coefficients
        a1 = 2. * (r ** 4) * np.cos(4. * nu)
        a2 = - (r ** 8)
        b1 = -2. * np.cos(nu)
        b2 = 1.
        b3 = 2. * r * np.cos(nu)
        b4 = r * r
        b5 = 2. * r * r * np.cos(2. * nu)
        b6 = r ** 4

        y = np.zeros(trace.shape[0])
        y1 = np.zeros(trace.shape[0])
        y2 = np.zeros(trace.shape[0])
        for n in range(trace.shape[0]):
            y1[n] = b2 * trace[n] + b1 * trace[n-1] + trace[n-2]
            y2[n] = y1[n] + b3 * y1[n-1] + b4 * y1[n-2]
            y[n]  = a1 * y[n-4] + a2 * y[n-8] + y2[n-2] + b5 * y2[n-4] + b6 * y2[n-6]
        return y 
                    

    def compute_true_time(self):
        """
        Compute the true time (in nanoseconds) for each event.
        """
        return ((self._trigger_secs - min(self._trigger_secs) )* 1e9 + self._trigger_nanos) 

    def get_unique_du(self):
        """
        Return the unique detection unit IDs and the indices of their first occurrence.
        """
        unique_du, idx = np.unique(self._du_ids, return_index=True)
        return unique_du, idx
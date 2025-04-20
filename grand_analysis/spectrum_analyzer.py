import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import blackman
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib.patches import Patch
from scipy.signal import hilbert

from scipy.fft import rfft, irfft, rfftfreq
import numpy as np

class SpectrumAnalyzer:
    """
    Class to perform FFT analysis and visualize the mean power spectral density (PSD).
    The PSD is computed in [V²/MHz] and plotted versus frequency in MHz.
    """
    def __init__(self, data_processor, visualizer, dt=2e-9):
        """
        Parameters:
        - data_processor: an instance of DataProcessor containing the processed data.
        - dt: time interval between data points (in seconds). 
              For example, dt=2e-9 corresponds to a 500 MHz sampling rate.
        """
        self.data = data_processor
        self.dt = dt
        self.visualizer = visualizer
    
    @staticmethod
    def apply_notch_filter(trace, which_filter, f_sample):
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

    def visualize_PSD(self, channels=[1], xlim=(0, 250), min_spectra=100,
                    apply_notch=False, f_sample=None, only_pre_trigger=False,
                    kadc=1.8/16384, VGA_gain=100, R=50):
        """
        Visualizes and updates the cumulative mean Power Spectral Density (PSD)
        for each Detection Unit (DU) and specified channels.

        The method processes the current file and computes the mean PSD for each DU.
        It then updates cumulative PSD averages stored as class attributes so that
        the overall mean is computed incrementally without keeping all raw data.

        Parameters:
        - channels: list of channel indices to analyze (default: [1]).
        - xlim: tuple, x-axis limits for frequency in MHz (default: (0, 250)).
        - min_spectra: minimum number of spectra (events) required for analysis.
        - apply_notch: bool, if True applies all four notch filters to each trace.
        - f_sample: sampling frequency in Hz. If None, set to 1/self.dt.
        - only_pre_trigger: if True, uses only channel 0 and truncates traces to 250 samples.
        - kadc: ADC conversion factor in V/ADC unit.
        - VGA_gain: voltage gain of the analog chain.
        - R: resistance used in PSD normalization.
        """

        # Set sampling frequency if not provided
        if f_sample is None:
            f_sample = 1.0 / self.dt

        # Initialize cumulative dictionaries as class attributes if not already present.
        # self.cumulative_psd stores for each DU a dictionary with key=channel and value=cumulative mean PSD array.
        # self.cumulative_events stores the total number of events processed for each DU.
        if not hasattr(self, 'cumulative_psd'):
            self.cumulative_psd = {}
        if not hasattr(self, 'cumulative_events'):
            self.cumulative_events = {}

        # Retrieve unique detection units (DUs) in the current file.
        unique_dus = np.unique(self.data._du_ids)
        plt.figure(figsize=(10, 6))

        # Process each DU in the current file.
        for du in unique_dus:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]  # Expected shape: (n_events, n_channels, n_samples)
            n_events = traces_du.shape[0]
            print(traces_du.shape)
            # Check if the DU has enough events to be processed.
            if n_events < min_spectra:
                print(f"DU {du} has only {n_events} spectra (< {min_spectra}). Skipping.")
                continue

            npts_original = traces_du.shape[2]
            npts = 250 if only_pre_trigger else npts_original
            if npts > npts_original:
                print(f"Warning: Requested {npts} points but only {npts_original} available. Truncating to available size.")
                npts = npts_original

            # Calculate frequency resolution and generate frequency arrays.
            delta_nu = f_sample / npts  # Frequency bin width in Hz
            freq_hz = rfftfreq(npts, self.dt)
            freq_MHz = freq_hz / 1e6

            # Create a Blackman window and compute its power correction factor.
            window = np.blackman(npts)
            power_window = np.sum(window ** 2) / npts

            # Initialize PSD accumulator for each channel.
            psd_accum = {ch: np.zeros(len(freq_hz)) for ch in channels}

            # Loop through all events and channels to compute PSD.
            for ev in range(n_events):
                for ch in channels:
                    # Use only pre-trigger data if specified, otherwise the full channel trace.
                    trace = traces_du[ev, 0, :npts] if only_pre_trigger else traces_du[ev, ch, :npts]

                    # Apply notch filters if requested.
                    if apply_notch:
                        for filt_id in [1, 2, 3, 4]:
                            trace = SpectrumAnalyzer.apply_notch_filter(trace, filt_id, f_sample)

                    # Apply windowing and compute the FFT.
                    trace_win = trace * window
                    fft_val = rfft(trace_win)
                    psd_event = np.abs(fft_val) ** 2 / power_window
                    psd_accum[ch] += psd_event

            # Process each channel to compute the mean PSD and update cumulative averages.
            for ch in channels:
                # Calculate the mean PSD for this file.
                mean_psd = psd_accum[ch] / n_events
                mean_psd *= (kadc ** 2) * VGA_gain  # Convert ADC² to V²

                # Normalize the PSD: divide by (N² * delta_nu) and then convert V²/Hz to V²/MHz.
                mean_psd /= (npts ** 2 * delta_nu) * R
                mean_psd *= 1e6

                # Adjust for single-sided FFT (except DC and Nyquist).
                if len(mean_psd) > 2:
                    mean_psd[1:-1] *= 2

                # Update cumulative average using weighted incremental update.
                if du in self.cumulative_psd:
                    n_prev = self.cumulative_events[du]
                    # If the channel already exists for this DU, update its cumulative average.
                    if ch in self.cumulative_psd[du]:
                        self.cumulative_psd[du][ch] = (
                            self.cumulative_psd[du][ch] * n_prev + mean_psd * n_events
                        ) / (n_prev + n_events)
                    else:
                        # If channel data is new, initialize it.
                        self.cumulative_psd[du][ch] = mean_psd.copy()
                    self.cumulative_events[du] += n_events
                else:
                    # Initialize cumulative PSD and event counter for new DU.
                    self.cumulative_psd[du] = {ch: mean_psd.copy()}
                    self.cumulative_events[du] = n_events

                # Plot the updated cumulative PSD for the current DU and channel.
                plt.semilogy(freq_MHz, self.cumulative_psd[du][ch],
                            lw=1, label=f'DU {du} - Ch{ch}')

        # Configure and save the plot.
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("PSD [V²/MHz]")
        plt.xlim(xlim)
        plt.grid(True, linestyle='--', alpha=0.7)
        title = "Cumulative Mean PSD Spectrum"
        if only_pre_trigger:
            title = "Cumulative Mean PSD - Pre-trigger Only"
        title += " (with notch filters)" if apply_notch else " (no notch)"
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig('cumulative_mean_PSD.png')
        plt.show()


    def analyze_baseline_vs_time(self, channel=[1], freq_band=None, du=None, apply_notch=False, 
                                f_sample=None, kadc=1.8/16384, R=50, VGA_gain=100, 
                                fit_sine=False, only_pre_trigger=False):
            """
            Analyze the standard deviation time within a specified frequency band.
            For each event, the PSD is computed and then integrated over the frequency band of interest 
            to obtain the variance (using Parseval's theorem). The standard deviation is then extracted 
            by taking the square root.
            
            Parameters:
            - channel: int or list. If int, only that channel is used.
            - freq_band: tuple (f_low, f_high) in MHz defining the frequency band.
            - du: int, list of int, or None. If int or list, only events from the specified DU(s) are analyzed.
                If None, all DUs present in the data are analyzed.
            - apply_notch: bool, if True, successively apply notch filters.
            - f_sample: Sampling frequency in Hz. If None, it is set to 1/self.dt.
            - kadc: Conversion factor from ADC counts to volts (default: 1.8/16384).
            - R: Impedance (default: 50 Ohm).
            - VGA_gain: VGA gain (default: 100).
            - fit_sine: bool, if True, perform a sine fit on the standard deviation vs. time.
            - only_pre_trigger: bool, if True, use only the pre-trigger part.
            """
            
            if f_sample is None:
                f_sample = 1. / self.dt

            # If du is None, get all unique DU IDs from the data; otherwise, ensure du_list is a list.
            if du is None:
                du_list = np.unique(self.data._du_ids)
            else:
                du_list = du if isinstance(du, list) else [du]

            # Create a single figure for all DUs
            plt.figure(figsize=(10, 6))
            
            # Get the default color cycle from matplotlib
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            for idx, current_du in enumerate(du_list):
                # Select color from the cycle
                color = colors[idx % len(colors)]
                
                mask = self.data._du_ids == current_du if current_du is not None else slice(None)
                traces = self.data._traces[mask]
                event_times = self.data._trigger_secs[mask] * 1e9 + self.data._trigger_nanos[mask]
                
                if traces.size == 0:
                    print(f"No event for DU {current_du}. Analysis stopped.")
                    continue

                if only_pre_trigger:
                    traces = traces[:, 0, :250]  # Shape (n_events, 250)
                    npts = traces.shape[1]
                else:
                    npts = traces.shape[2] # Should be 1024

                n_events = traces.shape[0] 
                if current_du is not None and n_events < 100:
                    print(f"DU {current_du} has {n_events} events (< 100). Analysis stopped.")
                    continue

                # Convert times to seconds relative to the first event.
                event_times = (event_times - event_times.min()) * 1e-9  # in seconds

                fit_sine_flag = fit_sine
                if event_times[-1] < 604800:
                    fit_sine_flag = False

                std_list = []
                delta_nu = f_sample / npts  # Frequency bin width in Hz

                # Create the Blackman window and compute the correction factor.
                bw = np.blackman(npts)
                pow_bw = np.sum(bw * bw) / npts

                # Compute the frequency axis in MHz.
                freq_hz = rfftfreq(npts, self.dt)
                freq_MHz = freq_hz / 1e6

                f_low, f_high = freq_band
                band_mask = (freq_MHz >= f_low) & (freq_MHz <= f_high)

                for ev in range(n_events):
                    if not only_pre_trigger:
                        trace_to_use = traces[ev, channel, :npts].astype(float)
                        if isinstance(channel, list) and len(channel) > 1:
                            trace_to_use = np.sqrt(np.sum(trace_to_use**2, axis=0))
                        else:
                            trace_to_use = trace_to_use[0]
                    else:
                        trace_to_use = traces[ev, :npts]

                    if apply_notch:
                        trace_filtered = trace_to_use
                        for filt in [1, 2, 3, 4]:
                            trace_filtered = SpectrumAnalyzer.apply_notch_filter(trace_filtered, filt, f_sample)
                        trace_to_use = trace_filtered

                    # Apply the Blackman window.
                    trace_win = trace_to_use * bw

                    # Compute the FFT and the PSD (in ADC²/Hz).
                    fft_val = rfft(trace_win)
                    psd_event = np.abs(fft_val)**2 / pow_bw

                    # ADC to V² conversion: apply (kadc)² and VGA gain.
                    psd_event *= (kadc)**2 * VGA_gain

                    # Normalization: divide by (npts² * delta_nu * R) and convert to V²/MHz.
                    psd_event /= (npts**2 * delta_nu * R)
                    psd_event *= 1e6  # Convert from V²/Hz to V²/MHz
                    psd_event[1:-1] *= 2  # Single-sided spectrum correction

                    # Integrate the PSD over the frequency band and compute the standard deviation.
                    integrated_power = np.sum(psd_event[band_mask]) * (delta_nu / 1e6)
                    std = np.sqrt(integrated_power)
                    std_list.append(std)

                std_list = np.array(std_list)
                
                # Plot the standard deviation vs. time for the current DU
                plt.plot(event_times, std_list, '.', color=color, label=f"DU {current_du}")
                
                # If sine fitting is enabled, perform the sine fit and plot the fitted curve.
                if fit_sine_flag:
                    def sine_model(t, A, f, phase, offset):
                        return A * np.sin(2 * np.pi * f * t + phase) + offset

                    A0 = (std_list.max() - std_list.min()) / 2
                    offset0 = std_list.mean()
                    f0 = 1 / 86400  # 24-hour period
                    phase0 = 0
                    p0 = [A0, f0, phase0, offset0]
                    try:
                        popt, pcov = curve_fit(sine_model, event_times, std_list, p0=p0)
                        fitted_curve = sine_model(event_times, *popt)
                        plt.plot(event_times, fitted_curve, '-', color=color, label=f"Sine Fit DU {current_du}")
                        print("DU {}: Fit parameters: A = {:.3g}, f = {:.3g} Hz, phase = {:.3g}, offset = {:.3g}".format(
                            current_du, *popt))
                    except Exception as e:
                        print(f"DU {current_du}: Error during sine fitting: {e}")

            plt.title(f"Baseline vs Time in {freq_band[0]}-{freq_band[1]} MHz (Channel {channel})")
            plt.xlabel("Time (s)")
            plt.ylabel("Standard Deviation (V)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('Baseline_202250204.png')
            plt.show()

    def analyse_mean_amplitude_vs_time(self, axis=1, only_pre_trigger=False):

        plt.figure(figsize=(10, 6))
        unique_dus = np.unique(self.data._du_ids)
        
        for du in unique_dus:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]  # forme: (n_events, n_channels, n_samples)
            n_events = traces_du.shape[0]
            if n_events == 0:
                print(f"Aucun événement pour DU {du}.")
                continue


            event_times = self.data.compute_true_time()
            event_times = event_times[mask] 
            
            # Sélection de la portion de trace à analyser
            if only_pre_trigger:
                data_to_analyze = traces_du[:, axis, :250]
            else:
                data_to_analyze = traces_du[:, axis, :]

            
            mean_amplitudes = np.mean(data_to_analyze, axis=1) # Calcule la moyenne pour chaque ligne en additionnant les valeurs de la ligne et en divisant par le nombre de colonnes.
            # Shape : (n_events,)
            # On trace le résultat : chaque événement est représenté par son trigger time et son amplitude moyenne.
            plt.plot(event_times, mean_amplitudes, '.', label=f'DU {du}')
        
        plt.xlabel("Temps de trigger (ns)")
        plt.ylabel("Amplitude moyenne")
        plt.title("Amplitude moyenne vs Temps de trigger")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()



    def analyse_std_baseline_vs_time(self, axis=1, only_pre_trigger=False):
        """
        Plots the standard deviation of the trace vs time for each Detection Unit (DU) for a chosen axis.

        Parameters:
        - axis: int, the channel index to analyze (default: 0).
        - only_pre_trigger: bool, if True, only use the pre-trigger portion of the trace.
        """
        plt.figure(figsize=(10, 6))
        unique_dus = np.unique(self.data._du_ids)

        # Loop over each detection unit
        for du in unique_dus:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]  # shape: (n_events, n_channels, n_samples)
            n_events = traces_du.shape[0]
            if n_events == 0:
                print(f"No events for DU {du}. Skipping.")
                continue
            
            event_times = self.data.compute_true_time()
            event_times = event_times[mask] 

            if only_pre_trigger:
                data_to_analyze = traces_du[:, 0, :250]
                npts = 250
            else:
                data_to_analyze = np.asarray(traces_du[:, axis, :], dtype=np.float64)



            std_trace = np.std(data_to_analyze, axis=1) # std of each event compare to the mean amplitude of the event


            plt.plot(event_times, std_trace, '.', label=f'DU {du}')

        plt.xlabel("Time (ns)")
        plt.ylabel("Standard Deviation")
        plt.title(f"Standard Deviation of Trace vs Time (Axis {axis}{', Pre-trigger' if only_pre_trigger else ''})")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    


    def histo_amplitude_all_event_norm(self, channels=None, amplitude_type='hilbert', bins=50, hist_range=None):
        """
        Calculates the signal amplitude for each antenna (channels corresponding to {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'})
        and displays a histogram of the normalized amplitude distribution with a vertical line indicating 
        the mean value.
        
        For 'hilbert' mode, the amplitude is computed as the mean of the envelope obtained via the Hilbert transform.
        
        Parameters:
        - channels: list of channel indices to analyze (default: [0, 1, 2, 3]).
        - amplitude_type: method for computing the amplitude. Options:
                - 'peak_to_peak': difference between maximum and minimum.
                - 'hilbert': mean of the absolute value of the Hilbert transform.
        - bins: number of histogram bins.
        - hist_range: tuple (min, max) for the histogram bins. If None, a global range across all channels is computed.
        """
        if channels is None:
            channels = [0, 1, 2, 3]
                    
        traces = self.data._traces  # shape: (n_events, n_channels, n_points)
        n_events, n_channels, n_points = traces.shape
        amplitude_values = np.zeros((n_events, 4))  # For storing amplitudes per channel
            
        # channel_names = {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        channel_names = {0: 'float', 1: 'X', 2: 'Y', 3: 'Z'}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
        
        # Variables to store envelope and raw signal for channel 1 (if needed)
        envelope_channel_fltX = None
        trace_channel_fltX = None
        
        # Dictionary to store normalized amplitude arrays per channel
        norm_data = {}
            
        plt.figure(figsize=(8, 6))
            
        for ch in channels:
            if ch >= n_channels:
                print(f"Channel {ch} out of bounds (only {n_channels} channels available).")
                continue
                    
            # Extract signals for all events from current channel
            signal_data = traces[:, ch, :]  # shape: (n_events, n_points)
                    
            if amplitude_type == 'peak_to_peak':
                amplitudes = np.ptp(signal_data, axis=1)
            elif amplitude_type == 'hilbert':
                analytic_signal = hilbert(signal_data, axis=1)
                amplitudes = np.mean(np.abs(analytic_signal), axis=1) # max ? 
                if ch == 1:
                    envelope_channel_fltX = np.abs(analytic_signal)
                    trace_channel_fltX = signal_data
            else:
                print("Unknown amplitude type. Use 'peak_to_peak' or 'hilbert'.")
                return
                    
            # Compute the standard deviation (noise_std) of the raw amplitudes
            noise_std = np.std(amplitudes)
            norm_amplitudes = amplitudes / noise_std if noise_std != 0 else amplitudes
            
            # Store normalized amplitudes for global range computation
            norm_data[ch] = norm_amplitudes
            
            
            if ch > 0:
                amplitude_values[:, ch-1] = norm_amplitudes
            else:
                amplitude_values[:, 3] = norm_amplitudes
            
        # Compute a global range for the histograms if not provided
        if hist_range is None:
            all_norm_values = np.concatenate([norm_data[ch] for ch in norm_data])
            global_min = np.min(all_norm_values)
            global_max = np.max(all_norm_values)
            hist_range = (global_min, global_max)
            
        # Now plot the histogram for each channel using the common range
        for ch in channels:
            if ch not in norm_data:
                continue
            mean_amp = np.mean(norm_data[ch])
            print(f"Channel {ch} ({channel_names.get(ch)}): Mean normalized amplitude ({amplitude_type}) = {mean_amp:.3g}")
            plt.hist(norm_data[ch], bins=bins, range=hist_range, alpha=0.5, 
                     label=f"Channel {ch} ({channel_names.get(ch)})", 
                     color=colors.get(ch))
            plt.axvline(mean_amp, color=colors.get(ch), linestyle='--', 
                        linewidth=2, label=f"Mean {channel_names.get(ch)} = {mean_amp:.3g}")
                
        plt.xlabel("Normalized Amplitude")
        plt.ylabel("Number of events")
        plt.title("Normalized Amplitude Distribution for all events per Channel")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
            
        # If using the 'hilbert' method, additionally display the envelope for channel 1 for a specific event (event index 12)
        if amplitude_type == 'hilbert' and envelope_channel_fltX is not None and trace_channel_fltX is not None:
            event_index = 12
            if event_index < n_events:
                plt.figure(figsize=(10, 4))
                plt.plot(trace_channel_fltX[event_index, :], label="Raw Signal", linewidth=1)
                plt.plot(envelope_channel_fltX[event_index, :], label="Hilbert Envelope", linewidth=2)
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Channel X - Event 12: Raw Signal and Hilbert Envelope")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Event index {event_index} is out of bounds (n_events = {n_events}).")




    def diag_average_amplitude_per_du(self, amplitude_type='hilbert'):
        """
        Plots a diagram showing the average normalized amplitude for each axis {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        for each Detection Unit (DU).
        
        For each DU:
          - For each channel (typically channels 0, 1, 2, and 3 corresponding to {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}), 
            the amplitude is computed for all events using the chosen method:
              - 'peak_to_peak': amplitude = max(signal) - min(signal)
              - 'hilbert': amplitude = mean(|Hilbert(signal)|)
          - The standard deviation (noise_std) of these raw amplitudes is computed.
          - The normalized average amplitude is defined as: 
                mean(raw amplitudes) / noise_std   (if noise_std ≠ 0)
        """
        unique_dus = np.unique(self.data._du_ids)
    
        # Define channel names and colors for all channels
        # channel_names = {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        channel_names = {0: 'float', 1: 'X', 2: 'Y', 3: 'Z'}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
    
        avg_amplitudes = {du: {} for du in unique_dus}
    
        for du in unique_dus:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]  # shape: (n_events, n_channels, n_points)
            n_events = traces_du.shape[0]
            for ch in [0, 1, 2, 3]:
                if ch >= traces_du.shape[1]:
                    print(f"Channel {ch} is out of bounds for DU {du}.")
                    continue
                raw_amps = []
                for ev in range(n_events):
                    trace = traces_du[ev, ch, :]
                    if amplitude_type == 'hilbert':
                        analytic_signal = hilbert(trace)
                        amp = np.mean(np.abs(analytic_signal))
                    else:  # Default 'peak_to_peak'
                        amp = np.max(trace) - np.min(trace)
                    raw_amps.append(amp)
                raw_amps = np.array(raw_amps)
                noise_std = np.std(raw_amps)
                norm_avg_amp = np.mean(raw_amps) / noise_std if noise_std != 0 else np.mean(raw_amps)
                avg_amplitudes[du][channel_names[ch]] = norm_avg_amp
    
        # Prepare data for a grouped bar chart
        du_labels = [str(du) for du in unique_dus]
        n_du = len(unique_dus)
        x = np.arange(n_du)
        bar_width = 0.2
    
        fig, ax = plt.subplots(figsize=(10, 6))
        channels_order = [0, 1, 2, 3]
        # For four channels, use offsets to center the groups: offsets = (i - 1.5) * bar_width
        for i, ch in enumerate(channels_order):
            values = [avg_amplitudes[du].get(channel_names[ch], np.nan) for du in unique_dus]
            ax.bar(x + (i - 1.5) * bar_width, values, width=bar_width, color=colors[ch],
                   label=channel_names[ch])
            
        ax.set_xlabel("Detection Unit (DU)")
        ax.set_ylabel("Normalized Average Amplitude")
        ax.set_title("Normalized Average Amplitude per DU for each Axis")
        ax.set_xticks(x)
        ax.set_xticklabels(du_labels)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()


    def diag_std_per_du(self, amplitude_type='hilbert'):
        """
        Displays a grouped bar chart showing the normalized standard deviation (dispersion) 
        of the amplitudes for each axis (X, Y, and Z) for each Detection Unit (DU).
        
        For each DU:
          - For each channel (typically channels 1, 2, and 3), the amplitude is computed for all events 
            using the chosen method:
              - 'peak_to_peak': amplitude = max(signal) - min(signal)
              - 'hilbert': amplitude = mean(|Hilbert(signal)|)
          - The standard deviation (noise_std) of these raw amplitudes is computed.
          - Each event's amplitude is then normalized by dividing by noise_std (if noise_std ≠ 0), 
            and the dispersion is computed as the standard deviation of these normalized amplitudes.
        """
        unique_dus = np.unique(self.data._du_ids)
    
        # channel_names = {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        channel_names = {0: 'float', 1: 'X', 2: 'Y', 3: 'Z'}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
    
        std_amplitudes = {du: {} for du in unique_dus}
    
        for du in unique_dus:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]  # shape: (n_events, n_channels, n_points)
            n_events = traces_du.shape[0]
            for ch in [0, 1, 2, 3]:
                if ch >= traces_du.shape[1]:
                    print(f"Channel {ch} is out of bounds for DU {du}.")
                    continue
                raw_amps = []
                for ev in range(n_events):
                    trace = traces_du[ev, ch, :]
                    if amplitude_type == 'hilbert':
                        analytic_signal = hilbert(trace)
                        amp = np.mean(np.abs(analytic_signal))
                    else:
                        amp = np.max(trace) - np.min(trace)
                    raw_amps.append(amp)
                raw_amps = np.array(raw_amps)
                std_amplitudes[du][channel_names[ch]] = np.std(raw_amps)
      
        du_labels = [str(du) for du in unique_dus]
        n_du = len(unique_dus)
        x = np.arange(n_du)
        bar_width = 0.2
    
        fig, ax = plt.subplots(figsize=(10, 6))
        channels_order = [0, 1, 2, 3]
        for i, ch in enumerate(channels_order):
            values = [std_amplitudes[du].get(channel_names[ch], np.nan) for du in unique_dus]
            ax.bar(x + (i - 2) * bar_width, values, width=bar_width, color=colors[ch],
                   label=channel_names[ch])
            
        ax.set_xlabel("Detection Unit (DU)")
        ax.set_ylabel("Normalized Standard Deviation of Amplitude")
        ax.set_title("Normalized Amplitude Dispersion per DU for each Axis")
        ax.set_xticks(x)
        ax.set_xticklabels(du_labels)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()


    def get_geolocation_data(self):
        """
        Retrieve the geolocation data (dus_feb) from the Visualizer.

        Returns:
        - dus_feb: Geodetic object containing the geolocation data.
        """
        dus_feb = self.visualizer.get_geolocation()

        return dus_feb




# Chelou a reprendre 

    def amplitude_map_X(self, amplitude_type='hilbert'):
        """
        Displays a geolocation map of the Detection Units (DUs) using their GPS data.
        For each DU, a filled circle is drawn at the position (-loc.y, loc.x) with a radius
        proportional to the normalized average amplitude in X (channel 1) for that DU.
        The color of the circle follows a gradient from the "OrRd" colormap.
        The amplitude is calculated using either the Hilbert method or the peak-to-peak method.
        """
        # Compute the normalized average amplitude in X (channel 1) for each DU.
        unique_du, idx = self.data.get_unique_du()
        avg_norm_amp = {}  # Dictionary to store normalized amplitude for each DU on channel 1
        
        for du in unique_du:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]
            n_events = traces_du.shape[0]
            if n_events == 0:
                continue
            raw_amps = []
            for ev in range(n_events):
                trace = traces_du[ev, 1, :]
                if amplitude_type == "hilbert":
                    analytic_signal = hilbert(trace)
                    amp = np.mean(np.abs(analytic_signal))
                else:  # Default: peak-to-peak method
                    amp = np.max(trace) - np.min(trace)
                raw_amps.append(amp)
            raw_amps = np.array(raw_amps)
            noise_std = np.std(raw_amps)
            norm_amp = np.mean(raw_amps) / noise_std if noise_std != 0 else np.mean(raw_amps)
            avg_norm_amp[du] = norm_amp
        
        # Retrieve geolocation data (assumed to be a Geodetic object with attributes x and y)
        loc = self.get_geolocation_data()
        
        fig, ax = plt.subplots()
        ax.plot(-loc.y, loc.x, 'ob', label="Detector Units")
        ax.plot(0, 0, 'or', label="Center Station")
        
        # Extract normalized amplitudes for all DUs (for channel 1)
        norm_amp_values = [avg_norm_amp[du] for du in unique_du if du in avg_norm_amp]
        max_amp = max(norm_amp_values) if norm_amp_values else 0
        min_amp = min(norm_amp_values) if norm_amp_values else 0
        scale = 300 / max_amp if max_amp != 0 else 1
        
        # Loop over each DU to add labels and circles
        for i, du in enumerate(unique_du):
            if du not in avg_norm_amp:
                continue
            center = (-loc.y[i], loc.x[i])
            ax.text(center[0] - 200, center[1] + 100, str(du), fontsize=12)
            amp = avg_norm_amp[du]
            # Compute the scaled radius based on the normalized amplitude
            r = scale * amp
            # Normalize the amplitude for the colormap "OrRd"
            normalized = (amp - min_amp) / (max_amp - min_amp) if max_amp != min_amp else 0
            color = plt.cm.OrRd(normalized)
            circle = Circle(center, r, facecolor=color, edgecolor=None)
            ax.add_patch(circle)
        
        # Create and add the colorbar
        norm_obj = colors.Normalize(vmin=min_amp, vmax=max_amp)
        sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Normalized Amplitude in X')
    
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Geolocation of Detection Units with Normalized Average Amplitude in X")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()




    def normalized_average_amplitude_map(self, channel=[0, 1, 2, 3], amplitude_type="hilbert"):
        """
        Displays a geographical map of the Detection Units (DUs) with circles representing the 
        normalized average amplitude for each axis (X, Y, and Z).
        
        For each DU:
          - For each channel (0, 1, 2, and 3), the amplitude is computed for all events 
            using the chosen method:
              - "peak_to_peak": amplitude = max(signal) - min(signal)
              - "hilbert": amplitude = mean(|Hilbert(signal)|)
          - The standard deviation (noise_std) of these amplitudes is then computed.
          - The normalized amplitude for that channel is defined as:
              normalized_amplitude = mean(raw amplitudes) / noise_std  (if noise_std != 0)
        
        Finally, using geolocation data, circles are drawn at each DU's position (only channels 1 and 2 are displayed),
        with an offset to avoid overlap.
        """
        unique_du, idx = self.data.get_unique_du()  
        avg_norm_amp = {du: {} for du in unique_du}
        
        for du in unique_du:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]
            n_events = traces_du.shape[0]
            if n_events == 0:
                continue
            
            # Compute raw amplitudes and noise_std for each channel
            for ch in [0, 1, 2, 3]:
                if ch >= traces_du.shape[1]:
                    continue
                raw_amps = []
                for ev in range(n_events):
                    trace = traces_du[ev, ch, :]
                    if amplitude_type == "hilbert":
                        analytic_signal = hilbert(trace)
                        amp = np.mean(np.abs(analytic_signal))
                    else:  # Default "peak_to_peak"
                        amp = np.max(trace) - np.min(trace)
                    raw_amps.append(amp)
                raw_amps = np.array(raw_amps)
                noise_std = np.std(raw_amps)
                norm_amp = np.mean(raw_amps) / noise_std if noise_std != 0 else np.mean(raw_amps)
                avg_norm_amp[du][ch] = norm_amp
    
        loc = self.get_geolocation_data()
        
        fig, ax = plt.subplots()
        ax.plot(-loc.y, loc.x, 'ob', label="Detector Units")
        ax.plot(0, 0, 'or', label="Center Station")
        
        # Global scaling: the DU with the maximum normalized amplitude will have a circle 
        # with the desired maximum radius.
        global_max_norm_amp = max(max(avg_norm_amp[du].get(ch, 0) for du in unique_du) for ch in [0, 1, 2, 3])
        desired_max_radius = 300.0
        global_scale = desired_max_radius / global_max_norm_amp if global_max_norm_amp != 0 else 1
        
        # Offsets to avoid overlapping circles for different axes (here X and Y)
        offset_map = {0: (-20, 0), 1: (0, 0), 2: (20,0), 3: (0, -20), 4: (0, -20)}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
        
        for i, du in enumerate(unique_du):
            center = (-loc.y[i], loc.x[i])
            ax.text(center[0] - 200, center[1] + 100, str(du), fontsize=12)
            # Plot for channels 1 and 2 (X and Y axes)
            for ch in channel:
                if ch not in avg_norm_amp[du]:
                    continue
                disp_val = avg_norm_amp[du][ch] 
                r = global_scale * disp_val  
                offset = offset_map[ch]
                center_offset = (center[0] + offset[0], center[1] + offset[1])
                circle = Circle(center_offset, r, facecolor=colors[ch], edgecolor='k', alpha=0.7)
                ax.add_patch(circle)
        
        # legend_elements = [
        #     Patch(facecolor='lightskyblue', edgecolor='k', label='Filtered Y'),
        #     Patch(facecolor='coral', edgecolor='k', label='Raw X'),
        #     Patch(facecolor='cornflowerblue', edgecolor='k', label='Raw Y'),
        #     Patch(facecolor='darkorange', edgecolor='k', label='Filtered X')
        # ]

        legend_elements = [
            Patch(facecolor='lightskyblue', edgecolor='k', label='Float'),
            Patch(facecolor='coral', edgecolor='k', label='X'),
            Patch(facecolor='cornflowerblue', edgecolor='k', label='Y'),
            Patch(facecolor='darkorange', edgecolor='k', label='Z')
        ]

        ax.legend(handles=legend_elements, loc='best')
        
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Geolocation of DUs with Normalized Average Amplitude per Axis")
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
    
        return avg_norm_amp




    def normalized_std_map(self, channel=[0, 1, 2, 3], amplitude_type="hilbert"):
        """
        Displays a geographical map of the Detection Units (DUs) with circles representing the 
        dispersion (standard deviation) of the normalized amplitudes for each axis (X, Y, and Z).
        
        For each DU, the method works as follows:
          - For each channel (0, 1, 2, and 3), extract the amplitude for all events using the chosen method:
              - "peak_to_peak": amplitude = max(signal) - min(signal)
              - "hilbert": amplitude = mean(|Hilbert(signal)|)
          - Compute the standard deviation (noise_std) of these raw amplitudes.
          - Compute the normalized amplitudes for each event by dividing the raw amplitude by noise_std.
          - Finally, compute the standard deviation of these normalized amplitudes.
        
        The resulting normalized dispersion is then used to scale circles on the geolocation map (only channels 1 and 2 are displayed).
        """
        unique_du, idx = self.data.get_unique_du()
    
        std_norm_amp = {du: {} for du in unique_du}
        for du in unique_du:
            mask = self.data._du_ids == du
            traces_du = self.data._traces[mask]
            n_events = traces_du.shape[0]
            if n_events == 0:
                continue
            
            for ch in channel:
                if ch >= traces_du.shape[1]:
                    print(f"Channel {ch} is out of bounds for DU {du}.")
                    continue
                raw_amps = []
                for ev in range(n_events):
                    trace = traces_du[ev, ch, :]
                    if amplitude_type == "hilbert":
                        analytic_signal = hilbert(trace)
                        amp = np.mean(np.abs(analytic_signal))
                    else:
                        amp = np.max(trace) - np.min(trace)
                    raw_amps.append(amp)
                raw_amps = np.array(raw_amps)
                std_amp = np.std(raw_amps)
                std_norm_amp[du][ch] = std_amp
      
        loc = self.get_geolocation_data()
        
        fig, ax = plt.subplots()
        ax.plot(-loc.y, loc.x, 'ob', label="Detector Units")
        ax.plot(0, 0, 'or', label="Center Station")
        
        # Global scaling: the DU with the maximum normalized dispersion will have a circle 
        # with the desired maximum radius.
        global_max_std = max(max(std_norm_amp[du].get(ch, 0) for du in unique_du) for ch in [0, 1, 2, 3])
        desired_max_radius = 300.0  # Maximum radius in meters
        global_scale = desired_max_radius / global_max_std if global_max_std != 0 else 1
        
        offset_map = {0: (-20, 0), 1: (0, 0), 2: (20,0), 3: (0, -20), 4: (0, -20)}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
        
        for i, du in enumerate(unique_du):
            center = (-loc.y[i], loc.x[i])
            ax.text(center[0] - 200, center[1] + 100, str(du), fontsize=12)
            # Plot for channels 1 and 2 (X and Y axes)
            for ch in channel:
                if ch not in std_norm_amp[du]:
                    continue
                disp_val = std_norm_amp[du][ch]  
                r = global_scale * disp_val  
                offset = offset_map[ch]
                center_offset = (center[0] + offset[0], center[1] + offset[1])
                circle = Circle(center_offset, r, facecolor=colors[ch], edgecolor='k', alpha=0.7)
                ax.add_patch(circle)
        
        # legend_elements = [
        #     Patch(facecolor='lightskyblue', edgecolor='k', label='Filtered Y'),
        #     Patch(facecolor='coral', edgecolor='k', label='Raw X'),
        #     Patch(facecolor='cornflowerblue', edgecolor='k', label='Raw Y'),
        #     Patch(facecolor='darkorange', edgecolor='k', label='Filtered X')
        # ]

        legend_elements = [
            Patch(facecolor='lightskyblue', edgecolor='k', label='Float'),
            Patch(facecolor='coral', edgecolor='k', label='X'),
            Patch(facecolor='cornflowerblue', edgecolor='k', label='Y'),
            Patch(facecolor='darkorange', edgecolor='k', label='Z')
        ]
    
        ax.legend(handles=legend_elements, loc='best')
        
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Geolocation of DUs with Normalized Standard Deviation (Dispersion) per Axis")
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

        

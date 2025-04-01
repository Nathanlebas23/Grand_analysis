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

    def visualize_PSD(self, channels=[0, 1, 2, 3], xlim=(0, 250), min_spectra=100,
                        apply_notch=False, f_sample=None, only_galacti_noise=False, kadc=1.8/16384, VGA_gain=100, R=50):
        """
        Visualizes the mean Power Spectral Density (PSD) for the specified channels and each Detection Unit (DU).
        Applies a Blackman window and averages over all events for each DU with at least `min_spectra` events.
        
        If apply_notch is True, all four notch filters are applied sequentially.
        The PSD is normalized as:
            PSD = |FFT|² / (N² · Δν_bin · R)
        and converted to units of [V²/MHz].

        Parameters:
        - channels: list of channel indices to analyze (default: [1,2,3]).
        - xlim: tuple, x-axis limits for the frequency in MHz (default: (0, 250)).
        - min_spectra: minimum number of spectra (events) required to compute the mean PSD.
        - apply_notch: bool, whether to apply all four notch filters to each trace.
        - f_sample: sampling frequency in Hz. If None, it is set to 1/self.dt.
        - only_galacti_noise: if True, uses only channel 0 and truncates traces to 250 samples.
        - kadc: ADC conversion factor in V/ADC unit (default: 1.8/16384 for a 14-bit ADC).
        - VGA_gain: voltage gain of the analog chain (unitless multiplicative factor).
        """

        if f_sample is None:
            f_sample = 1. / self.dt

        unique_dus = np.unique(self.data.du_ids)
        plt.figure(figsize=(10, 6))        
        i = 0
        for du in unique_dus:
            i+= 1
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]  # Expected (n_events, n_channels, n_samples)
            
            # Verify the dimension of traces_du
            if traces_du.ndim < 3:
                print(f"No event found for DU {du}.")
                continue

            n_events = traces_du.shape[0]

            if n_events < min_spectra:
                print(f"DU {du} has only {n_events} spectra (< {min_spectra}). Skipping.")
                continue

            npts_original = traces_du.shape[2]
            npts = 250 if only_galacti_noise else npts_original

            if npts > npts_original:
                print(f"Warning: Requested {npts} points but only {npts_original} available. Truncating to available size.")
                npts = npts_original

            delta_nu = f_sample / npts  # Frequency bin width in Hz
            freq_hz = rfftfreq(npts, self.dt)
            freq_MHz = freq_hz / 1e6

            window = np.blackman(npts)
            power_window = np.sum(window ** 2) / npts  # Power correction factor due to windowing

            psd_accum = {ch: np.zeros(len(freq_hz)) for ch in channels}

            for ev in range(n_events):
                for ch in channels:
                    # If galactic noise only mode, always use channel 0
                    trace = traces_du[ev, 0, :npts] if only_galacti_noise else traces_du[ev, ch, :npts]

                    if apply_notch:
                        for filt_id in [1, 2, 3, 4]:
                            trace = SpectrumAnalyzer.apply_notch_filter(trace, filt_id, f_sample)

                    trace_win = trace * window
                    fft_val = rfft(trace_win)
                    psd_event = np.abs(fft_val) ** 2 / power_window
                    psd_accum[ch] += psd_event

            for ch in channels:
                mean_psd = psd_accum[ch] / n_events
                mean_psd *= (kadc ** 2) * VGA_gain  # Convert from ADC² to V²

                # Normalize by (N² · Δν_bin), then convert from V²/Hz to V²/MHz
                mean_psd /= (npts ** 2 * delta_nu) * R # Division per R is weird maybe not homogeneous
                mean_psd *= 1e6  # Convert to V²/MHz

                # Compensate for single-sided FFT (except DC and Nyquist)
                if len(mean_psd) > 2:
                    mean_psd[1:-1] *= 2

                plt.semilogy(freq_MHz, mean_psd, lw=1, label=f'DU {du} - Ch{ch}')

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("PSD [V²/MHz]")
        plt.xlim(xlim)
        plt.grid(True, linestyle='--', alpha=0.7)

        if only_galacti_noise:
            title = "Mean PSD - Galactic Noise Only"
        else:
            title = "Mean PSD Spectrum"

        if apply_notch:
            title += " (with notch filters)"
        else:
            title += " (no notch)"

        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.show()


    def analyze_baseline_vs_time(self, channel, freq_band, du=None, apply_notch=False, 
                            f_sample=None, kadc=1.8/16384, R=50, VGA_gain=100, 
                            fit_sine=False, galactic_noise=False):
        """
        Analyse l'écart-type du bruit galactique versus le temps dans une bande de fréquence donnée.
        
        Pour chaque événement, on calcule la PSD puis on l'intègre sur la bande de fréquence 
        d'intérêt pour obtenir la variance (via le théorème de Parseval), et enfin on en extrait 
        l'écart-type en prenant la racine carrée.
        
        Parameters:
        - channel: int ou liste de deux ints. Si int, seul ce canal est utilisé.
                Si liste, la tension combinée est calculée comme V = sqrt(V1² + V2²).
        - freq_band: tuple (f_low, f_high) en MHz définissant la bande de fréquence.
        - du: int ou None, si fourni, seuls les événements de cet ID DU sont analysés.
        - apply_notch: bool, si True, applique successivement les quatre filtres notch à chaque trace avant FFT.
        - f_sample: Fréquence d'échantillonnage en Hz. Si None, est définie à 1/self.dt.
        - kadc: Facteur de conversion des ADC counts en volts (défaut : 1.8/16384).
        - R: Impédance (défaut : 50 Ohm).
        - VGA_gain: Gain du VGA (défaut : 100).
        - fit_sine: bool, si True, ajuste sinusoïdalement l'écart-type versus le temps.
        - galactic_noise: bool, si True, utilise seulement les 250 premiers échantillons du canal 0.
        """
        
        if f_sample is None:
            f_sample = 1. / self.dt

        # Définir le masque selon que 'du' est fourni ou non.
        mask = self.data.du_ids == du if du is not None else slice(None)

        # Sélectionner les traces et calculer les temps d'événement.


        # Sélectionner les traces et calculer les temps d'événement.
        traces = self.data.traces[mask]
        event_times = self.data.trigger_times[mask] * 1e9 + self.data.trigger_nanos[mask]

        # Vérifier que des événements sont présents et que le tableau a la bonne forme.
        if traces.ndim < 3 or traces.size == 0:
            print(f"Aucun événement trouvé pour DU {du}. Analyse interrompue.")
            return

        # Si le mode galactic_noise est activé, on conserve uniquement les 250 premiers échantillons du canal 0.
        if galactic_noise:
            traces = traces[:, 0, :250]  # Résultat de forme (n_events, 250)
            npts = traces.shape[1]
        else:
            npts = traces.shape[2]

        # Vérifier que le DU choisi possède suffisamment d'événements.
        n_events = traces.shape[0]
        if du is not None and n_events < 300:
            print(f"DU {du} ne possède que {n_events} événements (< 300). Analyse interrompue.")
            return

        # Convertir les temps en secondes relatifs au premier événement.
        event_times = (event_times - event_times.min()) * 1e-9  # en secondes

        # Vérifier que la fenêtre temporelle est d'au moins une semaine (604800 s)
        if event_times[-1] < 604800:
            print(f"La fenêtre temporelle ({event_times[-1]:.0f} s) est inférieure à une semaine. "
                "L'ajustement sinusoïdal ne sera pas réalisé.")
            fit_sine = False

        # Liste pour stocker l'écart-type pour chaque événement.
        std_list = []

        delta_nu = f_sample / npts  # largeur du bin en Hz

        # Créer la fenêtre Blackman et calculer le facteur de correction.
        bw = blackman(npts)
        pow_bw = np.sum(bw * bw) / npts

        # Calculer l'axe des fréquences en MHz.
        freq_hz = rfftfreq(npts, self.dt)
        freq_MHz = freq_hz / 1e6

        f_low, f_high = freq_band
        band_mask = (freq_MHz >= f_low) & (freq_MHz <= f_high)

        # Pour chaque événement, calculer la PSD et en extraire l'écart-type dans la bande d'intérêt.
        for ev in range(n_events):
            # Sélection de la trace selon le mode galactic_noise.
            if not galactic_noise:
                if isinstance(channel, list) and len(channel) == 2:
                    trace1 = traces[ev, channel[0], :npts]
                    trace2 = traces[ev, channel[3], :npts]
                    # Combinaison des deux canaux: V = sqrt(trace1² + trace2²)   pas sur de ca encore
                    trace_to_use = np.sqrt(trace1**2 + trace2**2)
                else:
                    trace_to_use = traces[ev, channel, :npts]
            else:
                trace_to_use = traces[ev, :npts]

            if apply_notch:
                trace_filtered = trace_to_use
                for filt in [1, 2, 3, 4]:
                    trace_filtered = SpectrumAnalyzer.apply_notch_filter(trace_filtered, filt, f_sample)
                trace_to_use = trace_filtered

            # Appliquer la fenêtre Blackman.
            trace_win = trace_to_use * bw

            # Calcul de la FFT et de la PSD (en unités ADC²/Hz).
            fft_val = rfft(trace_win)
            psd_event = np.abs(fft_val) ** 2 / pow_bw

            # Conversion ADC -> V² : appliquer (kadc)² et le gain VGA.
            psd_event *= (kadc)**2 * VGA_gain

            # Normalisation : division par (npts² * Δν_bin * R) puis conversion en V²/MHz.
            psd_event /= (npts * npts * delta_nu * R)
            psd_event *= 1e6  # Passage de V²/Hz à V²/MHz
            # Pour un spectre à un seul côté, multiplier les bins intermédiaires par 2.
            psd_event[1:-1] *= 2

            # Intégration de la PSD sur la bande d'intérêt pour obtenir la variance,
            # puis extraction de l'écart-type (std).
            integrated_power = np.sum(psd_event[band_mask]) * (delta_nu / 1e6)
            std = np.sqrt(integrated_power)
            std_list.append(std)

        std_list = np.array(std_list)

        plt.figure(figsize=(10, 6))
        plt.plot(event_times, std_list, 'o', label="Standard Deviation")
        
        if fit_sine:
            # Modèle sinusoïdal pour ajuster les données (périodicité sur 24h).
            def sine_model(t, A, f, phase, offset):
                return A * np.sin(2 * np.pi * f * t + phase) + offset

            A0 = (std_list.max() - std_list.min()) / 2
            offset0 = std_list.mean()
            f0 = 1 / 86400  # Périodicité de 24h (~1.16e-5 Hz)
            phase0 = 0
            p0 = [A0, f0, phase0, offset0]
            popt, pcov = curve_fit(sine_model, event_times, std_list, p0=p0)
            fitted_curve = sine_model(event_times, *popt)
            plt.plot(event_times, fitted_curve, '-', label="Sine Fit")
            print("Paramètres du fit: A = {:.3g}, f = {:.3g} Hz, phase = {:.3g}, offset = {:.3g}".format(*popt))
        
        if du is not None:
            plt.title(f"Galactic Noise Standard Deviation vs Time in {f_low}-{f_high} MHz (Channel {channel}, DU {du})")
        else:
            plt.title(f"Galactic Noise Standard Deviation vs Time in {f_low}-{f_high} MHz (Channel {channel})")
        plt.xlabel("Time (s)")
        plt.ylabel("Standard Deviation (V)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
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
                    
        traces = self.data.traces  # shape: (n_events, n_channels, n_points)
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
                amplitudes = np.mean(np.abs(analytic_signal), axis=1)
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
            
            # Optionally store in amplitude_values (if needed elsewhere)
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
        unique_dus = np.unique(self.data.du_ids)
    
        # Define channel names and colors for all channels
        # channel_names = {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        channel_names = {0: 'float', 1: 'X', 2: 'Y', 3: 'Z'}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
    
        avg_amplitudes = {du: {} for du in unique_dus}
    
        for du in unique_dus:
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]  # shape: (n_events, n_channels, n_points)
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
        unique_dus = np.unique(self.data.du_ids)
    
        # channel_names = {0: 'filtered Y', 1: 'raw X', 2: 'raw Y', 3: 'filtered X'}
        channel_names = {0: 'float', 1: 'X', 2: 'Y', 3: 'Z'}
        colors = {0: 'lightskyblue', 1: 'coral', 2: 'cornflowerblue', 3: 'darkorange'}
    
        std_amplitudes = {du: {} for du in unique_dus}
    
        for du in unique_dus:
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]  # shape: (n_events, n_channels, n_points)
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
            ax.bar(x + (i - 1.5) * bar_width, values, width=bar_width, color=colors[ch],
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
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]
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
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]
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
            mask = self.data.du_ids == du
            traces_du = self.data.traces[mask]
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

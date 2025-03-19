import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import blackman
from scipy.optimize import curve_fit

from scipy.fft import rfft, irfft, rfftfreq
import numpy as np

class SpectrumAnalyzer:
    """
    Class to perform FFT analysis and visualize the mean power spectral density (PSD).
    The PSD is computed in [V²/MHz] and plotted versus frequency in MHz.
    """
    def __init__(self, data_processor, dt=2e-9):
        """
        Parameters:
        - data_processor: an instance of DataProcessor containing the processed data.
        - dt: time interval between data points (in seconds). 
              For example, dt=2e-9 corresponds to a 500 MHz sampling rate.
        """
        self.data = data_processor
        self.dt = dt

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


    def visualize_mean_fft(self, channels=[1, 2, 3], xlim=(0, 250), min_spectra=100,
                       apply_notch=False, f_sample=None, only_galacti_noise=False, kadc=1.8/16384, R=50, VGA_gain=100):
        """
        Visualize the mean power spectral density (PSD) for the specified channels for each detection unit (DU),
        using a Blackman window. Only DUs with at least `min_spectra` events are processed.
        If apply_notch is True, all four notch filters are applied in sequence.

        The PSD is normalized according to:
            PSD = |FFT|² / (N² · Δν_bin)
        and then converted to [V²/MHz] (Δν_bin is in Hz).

        Parameters:
        - channels: list of channel indices to analyze (default: [1,2,3]).
        - xlim: tuple, x-axis limits for the frequency in MHz (default: (0, 250)).
        - min_spectra: minimum number of spectra (events) required to compute the mean PSD (default: 100).
        - apply_notch: bool, if True, apply all four notch filters in sequence to each trace before FFT.
        - f_sample: Sampling frequency in Hz. If None, it is set to 1/self.dt.
        - only_galacti_noise: bool, if True, use uniquement le canal 0 et tronquer les traces à 250 points.
        """

        if f_sample is None:
            f_sample = 1. / self.dt

        unique_dus = np.unique(self.data.du_ids)
        plt.figure(figsize=(10, 6))
        
        for du in unique_dus:
            mask = self.data.du_ids == du
            # Expected shape: (n_events, n_channels, n_points)
            traces_du = self.data.traces[mask]
            n_events = traces_du.shape[0]
            if n_events < min_spectra:
                print(f"DU {du} has only {n_events} spectra (< {min_spectra}). Skipping.")
                continue

            # Définir le nombre de points effectif pour la FFT
            npts_original = traces_du.shape[2]
            npts = 250 if only_galacti_noise else npts_original
            
            # Δν_bin (bin width) en Hz
            delta_nu = f_sample / npts  
            # Créer l'axe des fréquences en MHz.
            freq_hz = rfftfreq(npts, self.dt)
            freq_MHz = freq_hz / 1e6

            # Créer la fenêtre Blackman et calculer le facteur de correction de puissance.
            bw = np.blackman(npts)
            pow_bw = np.sum(bw * bw) / npts

            # Accumuler le PSD pour chaque canal sélectionné.
            psd_accum = {ch: np.zeros(freq_hz.shape) for ch in channels}

            # Traitement de chaque événement pour les canaux sélectionnés.
            for ev in range(n_events):
                for ch in channels:
                    # Si only_galacti_noise est activé, on utilise toujours le canal 0 et on tronque la trace.
                    if only_galacti_noise:
                        trace = traces_du[ev, 0, :npts]
                    else:
                        trace = traces_du[ev, ch, :npts]
                    if apply_notch:
                        # Appliquer en séquence les 4 filtres notch.
                        trace_filtered = trace
                        for filt in [1, 2, 3, 4]:
                            trace_filtered = SpectrumAnalyzer.apply_notch_filter(trace_filtered, filt, f_sample)
                        trace = trace_filtered
                    # Appliquer la fenêtre Blackman.
                    trace_win = trace * bw
                    # Calculer la FFT et le spectre de puissance.
                    fft_val = rfft(trace_win)
                    psd_event = np.abs(fft_val) ** 2 / pow_bw  
                    psd_accum[ch] += psd_event

            # Moyennage du PSD sur tous les événements et normalisation.
            for ch in channels:
                mean_psd = psd_accum[ch] / n_events 
                mean_psd = mean_psd * kadc**2 * VGA_gain  # Conversion en V² (en tenant compte du gain VGA)
                # Normalisation : division par (N² * Δν_bin * R) avec N = npts,
                # puis conversion de V²/Hz à V²/MHz par multiplication par 1e6.
                mean_psd /= (npts * npts * delta_nu * R)
                mean_psd *= 1e6
                # Pour un spectre à un seul côté, multiplier toutes les bins (sauf DC et Nyquist) par 2.
                mean_psd[1:-1] *= 2
                plt.semilogy(freq_MHz, mean_psd, lw=1, label=f'DU {du} - Ch{ch}')
        
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("PSD [V²/MHz]")
        plt.xlim(xlim)
        plt.grid(True, linestyle='--', alpha=0.7)
        if only_galacti_noise:
            if apply_notch:
                plt.title("Mean PSD Spectrum Galactic Noise Only (with all notch filters applied)")
            else:
                plt.title("Mean PSD Spectrum Galactic Noise Only (without notch filters)")
        else:    
            if apply_notch:
                plt.title("Mean PSD Spectrum (with all notch filters applied)")
            else:
                plt.title("Mean PSD Spectrum (without notch filters)")
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
        traces = self.data.traces[mask]
        event_times = self.data.trigger_times[mask] * 1e9 + self.data.trigger_nanos[mask]

        # Si le mode galactic_noise est activé, on conserve uniquement les 250 premiers échantillons du canal 0.
        if galactic_noise:
            traces = traces[:, 0, :250]  # Résultat de forme (n_events, 250)
            npts = traces.shape[1]
        else:
            npts = traces.shape[2]

        # Vérifier que le DU choisi possède suffisamment d'événements.
        n_events = traces.shape[0]
        if du is not None and n_events < 300:
            print(f"DU {du} ne possède que {n_events} événements (< 100). Analyse interrompue.")
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
                    trace2 = traces[ev, channel[1], :npts]
                    # Combinaison des deux canaux: V = sqrt(trace1² + trace2²)
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

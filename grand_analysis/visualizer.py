import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from grand import Geodetic, GRANDCS

class Visualizer:
    """
    Class to visualize various aspects of the processed data.
    """
    def __init__(self, data_processor, dt=2e-9):
        self.data = data_processor
        self.dt = dt

    def plot_geolocation(self):
        """
        Plot the geolocation of the detection units using their GPS data.
        """
        unique_du, idx = np.unique(self.data.du_ids, return_index=True)
        
        
        # Define the center station (example coordinates). /!\ Change this to the actual DAQ coordinate to correspond to the data !
        daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)
        dus_feb = Geodetic(
            latitude=self.data.du_lat[idx],
            longitude=self.data.du_long[idx],
            height=self.data.du_alt[idx]
        )
        dus_feb = GRANDCS(dus_feb, obstime="2024-09-15", location=daq)
        
        plt.figure()
        plt.plot(-dus_feb.y, dus_feb.x, 'ob', label="Detector Units")
        plt.plot(0, 0, 'or', label="Center Station")
        for i, du in enumerate(unique_du):
            plt.text(-dus_feb.y[i] + 50, dus_feb.x[i], str(du), fontsize=12)
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.title("Geolocation of Detection Units")
        plt.legend()
        plt.show()

    def get_geolocation(self):

        unique_du, idx = np.unique(self.data.du_ids, return_index=True)
        
        # Define the center station (example coordinates). /!\ Change this to the actual DAQ coordinate to correspond to the data !
        daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)
        dus_feb = Geodetic(
            latitude=self.data.du_lat[idx],
            longitude=self.data.du_long[idx],
            height=self.data.du_alt[idx]
        )
        dus_feb = GRANDCS(dus_feb, obstime="2024-09-15", location=daq)

        return dus_feb

    def visualize_event(self, target_du, evtid, channels=[1, 2], f_sample=None):
        """
        Visualize the time trace and FFT for a specified detection unit (DU) and event.
        
        For the given target DU and event ID (evtid), this method:
        - Extracts the traces corresponding to the DU.
        - Plots the time-domain trace for each specified channel.
        - Computes and plots the absolute FFT (spectral amplitude) of the trace
            for the last channel in the channels list.
        
        Parameters:
        - target_du: int, the DU ID to analyze.
        - evtid: int, the index of the event to visualize within the selected DU.
        - channels: list of ints, channel indices to display (default: [1, 2]).
        - f_sample: Sampling frequency in Hz. If None, it is set to 1/self.dt.
        """

        if f_sample is None:
            f_sample = 1. / self.dt

        # Création du masque pour sélectionner les événements du DU cible.
        dumask = self.data.du_ids == target_du
        # Extraction des traces pour ce DU.
        traces_du = self.data.traces[dumask]  # shape: (n_events, n_channels, n_points)

        # Vérifier que des événements existent pour ce DU.
        if traces_du.shape[0] == 0:
            print(f"Aucun événement trouvé pour DU {target_du}.")
            return

        # Vérification que l'événement demandé existe.
        if evtid >= traces_du.shape[0]:
            raise IndexError(f"Event ID {evtid} is out of range for DU {target_du} (only {traces_du.shape[0]} events available).")

        # --- Première figure : affichage des traces temporelles pour les canaux spécifiés ---
        plt.figure()
        for ch in channels:
            lab = "Ch" + str(ch)
            # Extraction de la trace pour l'événement evtid et le canal ch.
            trace_event = traces_du[evtid, ch, :]
            plt.plot(trace_event, label=lab, linewidth=1)
        plt.legend(loc='best')
        tit = f"DU {target_du} - Event {evtid}"
        plt.title(tit)

        # --- Deuxième figure : calcul et affichage de la FFT de la trace du dernier canal sélectionné ---
        # Ici, on prend la trace du dernier canal de la liste 'channels'
        selected_channel = channels[-1]
        trace_for_fft = traces_du[evtid, selected_channel, :]
        # Calcul de la FFT et du module
        afft = abs(rfft(trace_for_fft))
        # Calcul de l'axe des fréquences en Hz, puis conversion en MHz.
        freq = rfftfreq(len(trace_for_fft), self.dt)
        freq_MHz = freq / 1e6

        plt.figure()
        # Tracer une ligne verticale à 118.9 MHz (si cela correspond à une fréquence d'intérêt)
        plt.axvline(118.9, ls='--')
        plt.semilogy(freq_MHz, afft, linewidth=1)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("abs(FFT)")
        plt.title(f"FFT for DU {target_du} - Event {evtid} - Ch {selected_channel}")
        plt.tight_layout()
        plt.show()


    def plot_du_histogram(self):
        """
        Affiche un histogramme horizontal montrant le nombre de déclenchements par DU,
        trié de manière croissante (le DU ayant le plus grand nombre de déclenchements apparaît en haut).
        """
        du_ids, counts = np.unique(self.data.du_ids, return_counts=True)
        total_antennes = len(du_ids)
        total_declenchements = len(self.data.du_ids)
        
        # Création d'une liste de tuples (DU, count) et tri en ordre croissant par count.

        items = list(zip(du_ids, counts))
        sorted_items = sorted(items, key=lambda x: x[1])
        du_ids_sorted, counts_sorted = zip(*sorted_items)
        
        plt.figure(figsize=(10, 6))
        y_positions = np.arange(len(du_ids_sorted))
        plt.barh(y_positions, counts_sorted, color='blue')
        plt.xlabel("Number of triggers")
        plt.ylabel("Detector Unit")
        plt.title(
            f"Histogram of triggers per DU\n"
            f"Total triggered antennas: {total_antennes} | Total triggers: {total_declenchements}"
        )
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.yticks(y_positions, du_ids_sorted)
        
        for i, (du, count) in enumerate(zip(du_ids_sorted, counts_sorted)):
            plt.text(count + 0.005 * max(counts_sorted), i, str(count), va='center', fontsize=12)
        plt.tight_layout()
        plt.show()


    def plot_multiplicity_histogram(self):
        """
        Plot the histogram of event multiplicities (i.e. number of detection units triggered per event).
        """
        bins = np.arange(0.5, np.max(self.data.multiplicities) + 1.5, 1)
        plt.figure(figsize=(10, 6))
        plt.hist(self.data.multiplicities, bins=bins, color='lightgreen', edgecolor='black')
        plt.xlabel("Multiplicity (number of triggered antennas)")
        plt.ylabel("Coincidence index")
        plt.title("Histogram of coincidence multiplicity")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Highlight abnormal multiplicities (1, 2, 3).
        for mult in range(1, 4):
            count = np.sum(self.data.multiplicities == mult)
            if count > 0:
                plt.text(mult, count + 1, f"Anormal: {count}", ha='center', color='red', fontsize=12)
        plt.axvline(4, color='red', linestyle='--', label="Minimal multiplicity expected : 4")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_event_timing(self, r_exp=0.5):
        """
        Plot the relative timing of events for each detection unit.
        """
        true_time = self.data.compute_true_time()
        duration = (np.max(true_time) - np.min(true_time)) / 1e9
        
        plt.figure()
        unique_du = np.unique(self.data.du_ids)
        for du in unique_du:
            mask = self.data.du_ids == du
            t_du = (true_time[mask] - np.min(true_time)) / 1e9
            plt.plot(t_du, np.arange(np.sum(mask)), label=f'DU {du}')
        
        plt.xlabel('Event time (s)')
        plt.ylabel('Event #')
        # Plot an expected rate line for reference.
        plt.plot(
            [0, (np.max(true_time) - np.min(true_time)) / 1e9],
            [0, r_exp * (np.max(self.data.trigger_times) - np.min(self.data.trigger_times))],
            '--', label='expected'
        )
        plt.legend(loc='best')
        plt.title("Event Timing per Detection Unit")
        plt.show()

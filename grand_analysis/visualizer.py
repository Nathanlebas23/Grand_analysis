import numpy as np
import re
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
        self.c = 299792458  # Speed of light in m/s

    def plot_geolocation(self):
        """
        Plot the geolocation of the detection units using their GPS data.
        """
        unique_du, idx = np.unique(self.data.du_ids, return_index=True)
        
        
        # Define the center station (example coordinates). 
        daq   = Geodetic(latitude=40.99434, longitude=93.94177, height=1262) 
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

        return trace_event

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
            plt.text(count + 0.002 * max(counts_sorted), i, str(count), va='center', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_du_histogram_duplicate(self):
        """
        Affiche un histogramme horizontal montrant le nombre de déclenchements par DU,
        trié de manière croissante (le DU ayant le plus grand nombre de déclenchements apparaît en haut).
        
        Deux barres sont affichées :
        - La barre bleue indique le nombre total de triggers par DU.
        - La barre rouge indique le nombre de triggers par DU sans compter les doublons
            (c'est-à-dire, on compte une seule fois le trigger d'un même DU par événement).
        """
        # Calcul des totaux (nombre de triggers total par DU)
        du_ids, counts = np.unique(self.data.du_ids, return_counts=True)
        total_antennes = len(du_ids)
        total_declenchements = len(self.data.du_ids)

        # On trie les DU par ordre croissant de nombre de triggers total
        items = list(zip(du_ids, counts))
        sorted_items = sorted(items, key=lambda x: x[1])
        du_ids_sorted, counts_sorted = zip(*sorted_items)

        # Calcul des triggers "uniques" par événement (sans doublons)
        # Pour cela, on découpe les événements selon self.data.mult
        event_indices = self.get_event_indices()
        unique_counts_per_du = {}   # dictionnaire pour accumuler par DU
        for start, end in event_indices:
            # On récupère les DU uniques dans l'événement (même DU présent plusieurs fois est compté une seule fois)
            unique_du = np.unique(self.data.du_ids[start:end])
            for du in unique_du:
                unique_counts_per_du[du] = unique_counts_per_du.get(du, 0) + 1

        # Génère la liste des triggers "uniques" dans le même ordre que les DU triés précédemment
        unique_counts_sorted = [unique_counts_per_du.get(du, 0) for du in du_ids_sorted]

        # Création du graphique
        plt.figure(figsize=(10, 6))
        y_positions = np.arange(len(du_ids_sorted))
        bar_height = 0.4

        # Barres pour le nombre total de triggers (barres bleues)
        plt.barh(y_positions - bar_height/2, counts_sorted, height=bar_height,
                color='blue', label='Total Triggers')
        # Barres pour le nombre de triggers uniques par événement (barres rouges)
        plt.barh(y_positions + bar_height/2, unique_counts_sorted, height=bar_height,
                color='red', label='Unique Triggers (per event)')

        plt.xlabel("Number of triggers")
        plt.ylabel("Detector Unit")
        plt.title(
            f"Histogram of triggers per DU\n"
            f"Total triggered antennas: {total_antennes} | Total triggers: {total_declenchements}"
        )
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.yticks(y_positions, du_ids_sorted)

        # Affichage des valeurs numériques à côté des barres.
        for i, (du, tot_count, uniq_count) in enumerate(zip(du_ids_sorted, counts_sorted, unique_counts_sorted)):
            plt.text(tot_count + 0.002 * max(counts_sorted), y_positions[i] - bar_height/2,
                    str(tot_count), va='center', fontsize=12, color='blue')
            plt.text(uniq_count + 0.002 * max(counts_sorted), y_positions[i] + bar_height/2,
                    str(uniq_count), va='center', fontsize=12, color='red')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_multiplicity_histogram(self):
        """
        Plot the histogram of event multiplicities (i.e. number of detection units triggered per event).
        """
        total_declenchements = len(self.data.du_ids)
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
                plt.text(mult, count + 1, f"Anormal: {count} / {total_declenchements}", ha='center', color='red', fontsize=12)
        for mult in range(4, max(self.data.multiplicities) + 1):
            count = np.sum(self.data.multiplicities == mult)
            if count > 0:
                plt.text(mult, count + 1, f"{count} / {total_declenchements}", ha='center', color='red', fontsize=12)
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

    def plot_time_trigger(self):

        trigger_times_ns = self.data.compute_true_time()
        
        sorted_trigger_times = np.sort(trigger_times_ns)
        t_nsec = (sorted_trigger_times - sorted_trigger_times[0]) 
        cumulative_triggers = np.arange(1, len(t_nsec) + 1)
        
        plt.figure()
        
        plt.plot(t_nsec, cumulative_triggers, drawstyle='steps-post')
        plt.xlabel("Time since the first trigger (ns)")
        plt.ylabel("Number of trigger")
        plt.title("Evolution of the number of triggers over time")
        plt.grid(True)
        plt.show()


    def build_distance_matrix(self):
        """
        Build a distance matrix from detection unit positions.
        
        Returns:
        - distance_matrix: 2D numpy array in meters with shape (N, N) where N is the number of unique DUs.
        - du_to_index: a dictionary mapping a DU id to its row/column index in the matrix.
        """
        du_ids = np.unique(self.data.du_ids)
        print("Unique DU IDs:", du_ids)

        dus_feb = self.get_geolocation()

        M = len(du_ids)
        distance_matrix = np.zeros((M, M))
        
        du_to_index = {du: i for i, du in enumerate(du_ids)}
        
        for i in range(M):
            pos_i = np.array([-dus_feb.y[i], dus_feb.x[i]])
            for j in range(M):
                pos_j = np.array([-dus_feb.y[j], dus_feb.x[j]])
                distance_matrix[i, j] = np.linalg.norm(pos_i - pos_j) # metres

        return distance_matrix, du_to_index


    def get_causal_event(self):
       
        times_ns = (self.data.compute_true_time() - min(self.data.compute_true_time())) / 1e9
        event_indices = self.get_event_indices() 
        distance_matrix, du_to_index = self.build_distance_matrix()

        all_causal_events = []  
        all_causal_times = []   
        solo = 0  

    
        for start, end in event_indices:
            if (end - start) < 2:
                solo += 1
                continue

            times_event = times_ns[start:end]
            du_event = self.data.du_ids[start:end]

            sort_idx = np.argsort(times_event)
            times_sorted = times_event[sort_idx]
            du_sorted = du_event[sort_idx]

            du_indices = np.array([du_to_index[du] for du in du_sorted])
            M = len(times_sorted)

            causal_indices = [0]

            for i in range(1, M):
                for j in range(0, i):
                    d = distance_matrix[du_indices[i], du_indices[j]]
                    propagation_delay = d / self.c
                    deltaT = times_sorted[i] - times_sorted[j] - propagation_delay
                    print(f'couple ;', (i,j), 'deltaT :', deltaT)
                if deltaT >= 0:
                    # Tester le Chi² avec i ou j 
                    trigger_causal = True

                    # break  # On n'a pas besoin de tester les autres triggers antérieurs
            if trigger_causal:
                causal_indices.append(i)
            else:
                print("Trigger non causal retiré :", du_sorted[i], "à temps", times_sorted[i])

       
        event_causal_du = du_sorted[causal_indices]
        event_causal_times = times_sorted[causal_indices]

        all_causal_events.append(event_causal_du)
        all_causal_times.append(event_causal_times)
    
        return all_causal_events, all_causal_times


        
            
    def plot_trigger_vs_time_comparison(self):

        times_ns = self.data.compute_true_time()
        times_s = times_ns / 1e9
        total_triggers = len(times_s)

        clusters, num_raw_events, bad_indices = self.check_clusters_events_non_split()
        
        print(f"Number of flagged (bad) triggers: {len(bad_indices)}")

        raw_sorted_times = np.sort(times_s)
        raw_cum_count = np.arange(1, len(raw_sorted_times) + 1)


        clean_indices = [i for i in range(total_triggers) if i not in bad_indices]
        clean_sorted_times = np.sort(times_s[clean_indices])
        clean_cum_count = np.arange(1, len(clean_sorted_times) + 1)

        plt.figure(figsize=(10, 6))
        plt.step(raw_sorted_times, raw_cum_count, where='post', label='Raw triggers', color='k')
        plt.step(clean_sorted_times, clean_cum_count, where='post', label='Clean triggers (outliers removed)', color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Trigger Count")
        plt.title("Cumulative Number of Triggers vs Time")
        plt.legend()
        plt.grid(True)
        plt.show()



    def get_event_indices(self):
        """
        Computes start and end indices for each event from the flat data arrays,
        using the event multiplicities stored in the DataProcessor's mult array.
        
        :return: List of (start_index, end_index) tuples—one per event.
        """
        event_indices = []
        start = 0
        for m in self.data.mult:
            event_indices.append((start, start + m))
            start += m
        return event_indices



    def plot_deltaT_histogram(self, bins=50):
        times_ns = (self.data.compute_true_time() - min(self.data.compute_true_time())) / 1e9
        event_indices = self.get_event_indices() 
        distance_matrix, du_to_index = self.build_distance_matrix()
        all_deltaT = []
        solo = 0
        duplicatate = 0
        duplicates_per_DU = {}
        
        for start, end in event_indices:
            if (end - start) < 2:
                solo += 1
                continue



            times_event = times_ns[start:end]
            du_event = self.data.du_ids[start:end]
            
            
            sort_idx = np.argsort(times_event)
            times_sorted = times_event[sort_idx]
            du_sorted = du_event[sort_idx]
            
            
            du_indices = np.array([du_to_index[du] for du in du_sorted]) # pour traiter la matrice de distance ex: 1044:0, 1087:1
            
            unique_du, counts_du = np.unique(du_sorted, return_counts=True)
            for du, count in zip(unique_du, counts_du):
                if count >= 2:
                    duplicatate += 1
                    duplicates_per_DU[du] = duplicates_per_DU.get(du, 0) + (count - 1)


            M = len(times_sorted)
            dt_matrix = times_sorted[:, None] - times_sorted[None, :]
            

            distances = distance_matrix[du_indices[:, None], du_indices[None, :]]
            
            propagation_delays = distances / self.c
            deltaT_matrix = dt_matrix - propagation_delays
            
            lower_triangle_indices = np.tril_indices(M, k=-1) 

            event_deltaT = deltaT_matrix[lower_triangle_indices]
            print("event deltaT", event_deltaT)

            all_deltaT.append(event_deltaT)
        
        if all_deltaT:
            all_deltaT = np.concatenate(all_deltaT)
        else:
            all_deltaT = np.array([])
        
        print(solo, "events with only one trigger.")
        print(duplicatate, "duplicate triggers detected across events.")
        print("Duplicates per DU:", duplicates_per_DU)
        
        plt.figure()
        plt.hist(all_deltaT, bins=bins, edgecolor='black')
        plt.xlabel("ΔT (s) = T_i - T_j - d_{ij}/c (s)")
        plt.ylabel("Numbre of counts")
        plt.title("Histogram of ΔT")
        plt.grid(True)
        plt.show()


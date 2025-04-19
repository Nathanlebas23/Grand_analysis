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
    
        times_ns = (self.data.compute_true_time() - min(self.data.compute_true_time())) / 1e9 # REgarder dif ici
        event_indices = self.get_event_indices()
        distance_matrix, du_to_index = self.build_distance_matrix()

        all_problematic_indices = [] 
        solo = 0  

        for event_id, (start, end) in enumerate(event_indices):
            if (end - start) < 2:
                solo += 1
                all_problematic_indices.append([])
                continue

            times_event = times_ns[start:end]
            du_event = self.data.du_ids[start:end]

            sort_idx = np.argsort(times_event)
            times_sorted = times_event[sort_idx]
            du_sorted = du_event[sort_idx]

            M = len(times_sorted)
            # print(f"Event {event_id} - Nombre de triggers (M):", M)

            event_problematic = []  

            for i in range(1, M - 1):
                current_du = du_sorted[i]
                next_du = du_sorted[i + 1]

                current_index = du_to_index[current_du]
                next_index = du_to_index[next_du]
                d = distance_matrix[current_index, next_index]

                propagation_delay = d / self.c

                deltaT = times_sorted[i] - times_sorted[i + 1] - propagation_delay
                # print(f"Event {event_id}, paire ({i}, {i+1}) - deltaT: {deltaT}")

                if deltaT >= 0:
                    event_problematic.append((event_id,i, i + 1))
                    print(f"Event {event_id}, paire ({i}, {i+1}) - deltaT: {deltaT} (problematic)") 
            all_problematic_indices.append(event_problematic)
        print(solo)
        return all_problematic_indices



        
            
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
        print(event_indices, 'event indices')
        return event_indices


    def get_deltaT(self, bins=100):
        times_s = (self.data.compute_true_time() - min(self.data.compute_true_time())) / 1e9
        event_indices = self.get_event_indices() 
        distance_matrix, du_to_index = self.build_distance_matrix()
        all_deltaT = []
        solo = 0
        duplicate = 0
        duplicates_per_DU = {}
        duplicate_deltaT_by_du = {}
        duplicate_times_per_du = {}
        all_duplicate_times = []

        for start, end in event_indices:
            if (end - start) < 2:
                solo += 1
                continue

            times_event = times_s[start:end]
            du_event = self.data.du_ids[start:end]
            
            sort_idx = np.argsort(times_event)
            times_sorted = times_event[sort_idx]
            # print(len(times_sorted), "times sorted")
            du_sorted = du_event[sort_idx]   
            
            
            du_indices = np.array([du_to_index[du] for du in du_sorted]) # pour traiter la matrice de distance ex: 1044:0, 1087:1
            # print(du_indices)
            unique_du, counts_du = np.unique(du_sorted, return_counts=True)
            for du, count in zip(unique_du, counts_du):
                if count >= 2:
                    duplicate += 1
                    duplicates_per_DU[du] = duplicates_per_DU.get(du, 0) + (count - 1)



            M = len(times_sorted)
            # print(f"Nombre de triggers (M):", M)
            
            dt_matrix = times_sorted[:, None] - times_sorted[None, :]
            

            distances = distance_matrix[du_indices[:, None], du_indices[None, :]]
            
            
            propagation_delays = distances / self.c
            # print("propagation delays", propagation_delays)

            deltaT_matrix = dt_matrix - propagation_delays
            
            # lower_triangle_indices = np.tril_indices(M, k=-1) 
            # event_deltaT = deltaT_matrix[lower_triangle_indices]

            sub_diag = np.diag(deltaT_matrix, k=-1)
            # print("event deltaT", sub_diag)
            
            all_deltaT.append(sub_diag)


            for i in range(1, M):
                if du_sorted[i] == du_sorted[i-1]:
                    du_val = du_sorted[i]
                    du_time = times_sorted[i]
                    # print("du_val", du_val)  
                    dt_val = sub_diag[i-1]
                    # print("dt_val", dt_val)
                    duplicate_times_per_du[du_val] = [] 
                    if du_val in duplicate_deltaT_by_du:
                        duplicate_deltaT_by_du[du_val].append(dt_val)
                    else:
                        duplicate_deltaT_by_du[du_val] = [dt_val]
                    
                    duplicate_times_per_du[du_val].append(du_time)
                    all_duplicate_times.append(du_time)

                    
        # print("all deltaT", all_deltaT)
        if all_deltaT:
            all_deltaT = np.concatenate(all_deltaT)
        else:
            all_deltaT = np.array([])

        du_list = sorted(duplicate_deltaT_by_du.keys())
        data_list = [duplicate_deltaT_by_du[du] for du in du_list]


        
        # print(du_to_index)
        # print(distance_matrix)
        # print(solo, "events with only one trigger.")
        # print(duplicate, "duplicate triggers detected across events.")
        # print("Duplicates per DU:", duplicates_per_DU)
        # print(len(all_deltaT), "all deltaT")

        return all_deltaT, data_list, du_list, duplicate_deltaT_by_du, duplicate_times_per_du, all_duplicate_times, duplicate


    def plot_deltaT_histogram(self, bins=50):
        
        all_deltaT, data_list, du_list, duplicate_deltaT_by_du, duplicate_times_per_du, all_duplicate_times, duplicate = self.get_deltaT(bins=bins)
        # print(duplicate_deltaT_by_du, "duplicate deltaT by DU")

        du_ids, counts = np.unique(self.data.du_ids, return_counts=True)
        total_counts = dict(zip(du_ids, counts))

        plt.figure(figsize=(10, 6))

        global_min = np.min(all_deltaT)
        global_max = np.max(all_deltaT)
        hist_range = (global_min, global_max)

        plt.hist(all_deltaT, bins=bins, range=hist_range, color='gray', alpha=0.5, label="All events", edgecolor='black')
        
        labels = []
        for du in du_list:
            dup = len(duplicate_deltaT_by_du.get(du, []))
            total = total_counts.get(du, 1)
            pct_du = dup / total * 100
            pct_tot = dup / len(self.data.du_ids) * 100
    
            labels.append(f"DU {du}: {dup}/{total} ({pct_du:.1f}% DU, {pct_tot:.1f}% tot)")

        pct_tot_dup = duplicate / len(self.data.du_ids)  * 100   
        plt.hist(data_list, bins=bins, range=hist_range, stacked=True, 
                label=labels)

        plt.xlabel("ΔT (s)")
        plt.ylabel("Number of duplicate")
        plt.title(f"Histogram all ΔT and per duplicate DUs\n - Number of events: {len(self.data.du_ids)} -\n - Number of duplicates: {duplicate} ({pct_tot_dup:.1f}% of the total) - ")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() 
        plt.savefig("deltaT_histogram.png")
        plt.show()
        

        plt.figure(figsize=(10, 6))

        # Concaténer toutes les valeurs ΔT issues des doublons afin de calculer le min et max global
        all_duplicate_values = np.concatenate(list(duplicate_deltaT_by_du.values()))
        global_min_duplicate = np.min(all_duplicate_values)
        global_max_duplicate = np.max(all_duplicate_values)
        hist_range_duplicate = (global_min_duplicate, global_max_duplicate)

        for du in du_list:
            plt.hist(duplicate_deltaT_by_du[du], bins=bins, range=hist_range_duplicate,
                    alpha=0.5, label=f"DU {du}", edgecolor='black')

        plt.xlabel("ΔT (s)")
        plt.ylabel("Number of duplicates")
        plt.title("Histogram of ΔT for the duplicates per DU")
        plt.legend()
        plt.grid(True)
        plt.savefig("duplicate_deltaT.png")
        plt.show()
        

        all_trigger_times = (self.data.compute_true_time() - np.min(self.data.compute_true_time())) / 1e9
        all_trigger_times = np.array(all_trigger_times)

        all_duplicate_times = np.array(all_duplicate_times)
        global_min = min(all_trigger_times.min(), all_duplicate_times.min())
        global_max = max(all_trigger_times.max(), all_duplicate_times.max())
        bins_time = np.linspace(global_min, global_max, bins+1)



        bin_width = bins_time[1] - bins_time[0]

        plt.figure(figsize=(10, 6))

        
        plt.hist(
            all_trigger_times,
            bins=bins_time,
            histtype='stepfilled',
            linewidth=2,
            alpha=0.5,
            label="All triggers",
            weights=np.ones_like(all_trigger_times) / bin_width,
            edgecolor='black'
        )

        
        plt.hist(
            all_duplicate_times,
            bins=bins_time,
            histtype='stepfilled',
            linewidth=2,
            label="All duplicates",
            weights=np.ones_like(all_duplicate_times) / bin_width
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Rate of duplicates DUs (Hz)")                     
        plt.title("Differential number of duplicates vs trigger time")
        plt.legend()
        plt.grid(True)
        plt.savefig("duplicate_time.png")
        plt.show()
        

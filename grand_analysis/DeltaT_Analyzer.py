import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import math
from collections import Counter
from itertools import combinations
from grand import Geodetic, GRANDCS


class DeltaTAnalyzer:
    
    def __init__(self, data_precessor, c=299792458.0):
        
        self.data_processor = data_precessor




        mask_pos    = self.data.compute_true_times() > 0
        min_nonzero = self.data.compute_true_times[mask_pos].min()
        self.true_times_ns = self.compute_true_times() - min_nonzero

    def verify(self):
        n = len(self.du_ids)
        for i in range(10):
            print(
                f"Itération {i:>6d} | DU ID = {self.du_ids[i]} | "
                f"Temps = {self.true_times_ns[i]} ns ({self.true_times_s[i]:.4f} s) | "
                f"Multiplicité = {self.multiplicities[i]} | "
                f"GPS = ({self.lat[i]:.2f}, {self.lon[i]:.2f}, alt={self.alt[i]:.2f} m)"
            )

    def _get_event_indices(self):
        indices = []
        i = 0
        N = len(self.multiplicities)
        while i < N:
            m = self.multiplicities[i]
            if m <= 0 or i + m > N:
                raise ValueError(f"Invalid multiplicity at index {i}: {m}")
            indices.append((i, i + m))
            i += m
        return indices
    
    def _get_geolocation(self):

        unique_du, idx = np.unique(self.du_ids, return_index=True)
        
        daq   = Geodetic(latitude=40.99434, longitude=93.94177, height=1262) 
        dus_feb = Geodetic(
            latitude=self.lat[idx],
            longitude=self.lon[idx],
            height=self.alt[idx]
        )
        dus_feb = GRANDCS(dus_feb, obstime="2024-09-15", location=daq)

        return dus_feb

    def _build_distance_matrix(self):
        du_ids = np.unique(self.du_ids)
        print("Unique DU IDs:", du_ids)

        dus_feb = self._get_geolocation()

        M = len(du_ids)
        distance_matrix = np.zeros((M, M))
        
        du_to_index = {du: i for i, du in enumerate(du_ids)}
        
        for i in range(M):
            pos_i = np.array([-dus_feb.y[i], dus_feb.x[i]])
            for j in range(M):
                pos_j = np.array([-dus_feb.y[j], dus_feb.x[j]])
                distance_matrix[i, j] = np.linalg.norm(pos_i - pos_j) # metres

        return distance_matrix, du_to_index


    def compute_deltaT(self):

        all_deltaT = []
        all_deltaT_d0 = []
        dup_by_du = {}
        times_by_du = {}
        all_dup_times = []
        valid_times = []
        dup_mse_by_du = {}
        dup_evt_by_du = {}
        dt_pos = 0
        duplicate_count = 0
        solo = 0
        
        
        for idx, (start, end) in enumerate(self.indices):
            length = end - start
            if length < 2:
                solo += 1
                continue


            traces_y = self.traces[start:end, 1, :]
            if traces_y.shape[0] != length:
                # Shape incompatible, skip this event
                continue
            # Extract the event data
            t_evt = self.true_times_s[start:end]
            du_evt = self.du_ids[start:end]
            
            
            # Sort the events by time
            order = np.argsort(t_evt)

            t_sorted = t_evt[order]
            du_sorted = du_evt[order]
            idxs = [self.du_to_idx[d] for d in du_sorted]
            

            M = len(t_sorted)

            # Time matrix
            dt = t_sorted[:, None] - t_sorted[None, :]
            
            # Propagation matrix
            dist_mat = self._build_distance_matrix()
            prop = dist_mat[np.ix_(idxs, idxs)] / self.c
            
            dT = dt - prop
            
            # Following events deltaT
            sub = np.diag(dT, k=-1)
            prop_d = np.diag(prop,  k=-1)


            # We only keep the events where the propagation time is not zero
            mask_nonzero = (prop_d != 0)

            filtered = sub[(mask_nonzero) & (sub < 1e-4)]   
            all_deltaT.append(sub)
            all_deltaT_d0.append(filtered)

            
            uniq, cnts = np.unique(du_sorted, return_counts=True)
            for du, cnt in zip(uniq, cnts):
                if cnt > 1:
                    duplicate_count += cnt - 1
        

            for k in range(1, M):
                dtv = sub[k-1]            
                if dtv > 0:
                    dt_pos += 1

                if dtv > 1 :
                    continue
                
                if du_sorted[k] == du_sorted[k-1]:
                    duv = du_sorted[k]
                    dtv = sub[k-1]
                    y_i = traces_y[k]
                    y_j = traces_y[k-1]
                    mse = np.mean((y_i - y_j) ** 2)
                    
                    dup_mse_by_du.setdefault(duv, []).append(mse)
                    dup_evt_by_du.setdefault(duv, []).append(idx)
                    dup_by_du.setdefault(duv, []).append(dtv)
                    times_by_du.setdefault(duv, []).append(t_sorted[k])
                    all_dup_times.append(t_sorted[k])
                
        all_deltaT = np.concatenate(all_deltaT) if all_deltaT else np.array([], dtype=float)
        all_deltaT_d0 = np.concatenate(all_deltaT_d0) if all_deltaT_d0 else np.array([], dtype=float)
        
        uniq_du, counts_du = np.unique(self.du_ids, return_counts=True)
        total_counts = dict(zip(uniq_du, counts_du))

        real_non_causal_ev = dt_pos - duplicate_count

        
        return all_deltaT, all_deltaT_d0, dup_by_du, times_by_du, all_dup_times, duplicate_count, total_counts, real_non_causal_ev, dup_mse_by_du, dup_evt_by_du


################################ Plots ###############################


    ############################
    # Plot histogram of deltaT #
    ############################  

    def plot_deltaT_histogram(self, bins=100000):
        
        all_deltaT, all_deltaT_d0, dup_by_du, _, all_dup_times, dup_count, total_counts, real_non_causal_ev, dup_mse_by_du, dup_evt_by_du = self.compute_deltaT()
        
        plt.figure(figsize=(10, 6))
        
        if all_deltaT.size:
            mn, mx = all_deltaT_d0.min(), all_deltaT_d0.max()

            plt.hist(all_deltaT_d0, bins=bins, range=(mn, mx), alpha=0.5,
                    label='All non duplicate DUs', edgecolor='black')
    
            stacked = [dup_by_du[d] for d in sorted(dup_by_du)]
            
            labels = []
            for du in sorted(dup_by_du):
                dup = len(dup_by_du.get(du, []))
                total = total_counts.get(du, 1)
                pct_du = dup / total * 100
                pct_tot = dup / len(self.du_ids) * 100
                labels.append(f"duplicate DUs {du}: {dup}/{total} ({pct_du:.1f}% DU, {pct_tot:.1f}% tot)")

            
            pct_tot_dup = dup_count / len(self.du_ids) * 100
            pct_tot_non_causal = real_non_causal_ev / len(self.du_ids) * 100

            plt.hist(stacked, bins=bins, range=(mn, mx), stacked=True,
                     label=labels)
        
        plt.xlabel('ΔT (s)')
        plt.ylabel('Count')
        plt.title(f"Histogram all ΔT and per duplicate DUs\n - Number of events: {len(self.du_ids)} -\n - Number of duplicates: {dup_count} ({pct_tot_dup:.1f}% of the total) -\n - Number of non-causal events: {real_non_causal_ev} ({pct_tot_non_causal:.1f}% of the total) -")
        plt.legend(loc='best'); 
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.tight_layout()
        plt.savefig('deltaT_histogram.png'); 
        # plt.show()
        
        
       
        
        
    ###########################
    # Plot MSE for duplicates #
    ###########################  
    
    def plot_mse_duplicates(self):
        all_deltaT, all_deltaT_d0, dup_by_du, _, all_dup_times, dup_count, total_counts, real_non_causal_ev, dup_mse_by_du, dup_evt_by_du = self.compute_deltaT()
           
        n = len(dup_mse_by_du)
        fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), sharex=False, constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, du in zip(axes, sorted(dup_mse_by_du)):
            evts = dup_evt_by_du[du]
            mses = dup_mse_by_du[du]
            ax.plot(evts, mses, 'o-', label=f'DU {du}')
            ax.set_title(f'MSE (Ch 1) for DU {du}')
            ax.set_xlabel("Event Index")
            ax.set_ylabel("MSE")
            ax.grid(True)
        
        evt_counts = Counter(evt for evts in dup_evt_by_du.values() for evt in evts)
        total_pairs = sum(evt_counts.values())
        print(f"Total pairs of duplicates: {total_pairs}")
        plt.tight_layout()
        plt.savefig('mse_duplicates.png')
        # plt.show()



    ################################
    # Plot histogram of duplicates #
    ################################
    def plot_duplicate_histogram(self, bins=1000):
        all_deltaT, all_deltaT_d0, dup_by_du, _, all_dup_times, dup_count, total_counts, real_non_causal_ev, dup_mse_by_du, dup_evt_by_du = self.compute_deltaT()
           
        t_s = self.true_times_s
        
        min_t = t_s[ t_s>0 ].min()
        max_t = t_s.max()       
        max_t = t_s.max()
    
        bins_t = np.linspace(min_t, max_t, bins+1)
        w = bins_t[1] - bins_t[0]

        counts_all, _ = np.histogram(t_s, bins=bins_t)
        counts_dup, _ = np.histogram(all_dup_times, bins=bins_t)

        bin_centers = (bins_t[:-1] + bins_t[1:]) / 2

        eps = 1e-8
        ratio = counts_all  / (counts_dup + eps)
        

        plt.figure(figsize=(10, 6))
        plt.hist(t_s, bins=bins_t, histtype='stepfilled', linewidth=2,
                 alpha=0.5, label='All', weights=np.ones_like(t_s)/w)
        plt.hist(all_dup_times, bins=bins_t, histtype='stepfilled', linewidth=2,
                 alpha=0.5, label='Duplicates', weights=np.ones_like(all_dup_times)/w)
        
        mask = ratio > 0
        plt.plot(bin_centers[mask], ratio[mask],color='red',lw=2,label='All/Duplicates Ratio')



        plt.xlabel("Time (s)")
        plt.ylabel("Rate of duplicates DUs (Hz)")                     
        plt.title("Differential number of duplicates vs trigger time")
        plt.legend(loc='best') 
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('duplicate_time.png')
        plt.show()





    ####################
    # plot_deltaT_zoom #
    ####################  
        
    def plot_deltaT_zoom(self, bins=10000, window_ns=4):
        """
        Pour chaque DU :
        - on calcule l'histogramme complet de ses ΔT
        - on trouve le bin de fréquence max, on en déduit le centre peak_ns
        - on trace un histogramme zoomé dans [peak_ns-window_ns, peak_ns+window_ns]
        """
        dup_by_du = self.compute_deltaT()[2]
        window = window_ns  # en ns
        

        all_vals_ns = np.concatenate([np.array(v)*1e9 for v in dup_by_du.values()])
        global_min, global_max = all_vals_ns.min(), all_vals_ns.max()
        global_edges = np.linspace(global_min, global_max, 1000)

        n = len(dup_by_du)
        fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), sharex=False, constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (du, vals_s) in zip(axes, dup_by_du.items()):
            vals_ns = np.array(vals_s) * 1e9

            counts, edges = np.histogram(vals_ns, bins=global_edges)
            peak_idx = np.argmax(counts)
            peak_ns  = 0.5*(edges[peak_idx] + edges[peak_idx+1])

            low_ns, high_ns = peak_ns - window, peak_ns + window

            zoom_edges = np.linspace(low_ns, high_ns, bins+1)
            bin_width_ns = (zoom_edges[1] - zoom_edges[0]) 

            weights = np.ones_like(vals_ns) / bin_width_ns

            ax.hist(vals_ns, bins=zoom_edges, weights=weights,
                    edgecolor='black', alpha=0.7)
            ax.set_title(f"DU {du} — pic à {peak_ns:.2f} ns")
            ax.set_xlabel("ΔT (ns)")
            ax.set_ylabel("Rate (Hz)")
            ax.set_xlim(low_ns, high_ns)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('deltaT_zoom.png')
        plt.show()



    ###################################
    # plot_duplicate_rate_vs_distance #
    ################################### 

    def plot_duplicate_rate_vs_distance(self):
        """
        Trace le taux de duplications par DU en fonction de sa distance au DAQ.

        - taux interne (duplications / total DU)
        - taux global   (duplications / total triggers)
        """
        # Récupération des résultats
        all_deltaT, all_deltaT_d0, dup_by_du, times_by_du, all_dup_times, duplicate_count, total_counts, real_non_causal_ev, dup_mse_by_du, dup_evt_by_du = self.compute_deltaT()

        # Coordonnées DU dans le repère local et distance au DAQ
        dus = self._get_geolocation()
        # dus.x, dus.y sont des arrays alignés sur self.unique_du
        dists = np.hypot(dus.x, dus.y)  # shape (n_du,)

        du_ids = np.array(self.unique_du)
        # Tri par distance pour un affichage ordonné
        order = np.argsort(dists)
        du_ids = du_ids[order]
        dists  = dists[order]

        # Taux interne : nombre de duplications pour ce DU / total triggers de ce DU
        rates_per_du = np.array([
            len(dup_by_du.get(du, [])) / total_counts.get(du, 1)
            for du in du_ids
        ])

        # Taux global : nombre de duplications pour ce DU / nombre total de triggers
        total_triggers = len(self.du_ids)
        rates_global = np.array([
            len(dup_by_du.get(du, [])) / total_triggers
            for du in du_ids
        ])

        # Plot
        plt.figure(figsize=(9, 5))
        plt.scatter(dists, rates_per_du, s=60, label='DU Duplicate rate ', edgecolor='black')
        plt.scatter(dists, rates_global, s=60, label='Global Duplicate rate',
                    facecolors='none', edgecolor='red', linewidth=1)

        # Annotations des points avec l'ID DU
        for du, x, y in zip(du_ids, dists, rates_per_du):
            plt.text(x, y, str(du), fontsize=8, va='bottom', ha='right')

        plt.xlabel("Distance to DAQ (m)")
        plt.ylabel("Duplication Rate")
        plt.title("Duplication rates per DU vs distance au DAQ")
        plt.legend(loc='upper right', frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('duplicate_rate_vs_distance.png')
        plt.show()


    ######################################
    # Visualize event time trace and FFT #
    ######################################

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
        dumask = self.du_ids == target_du
        # Extraction des traces pour ce DU.
        traces_du = self.traces[dumask]  # shape: (n_events, n_channels, n_points)

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
        plt.savefig('Event display')
        plt.show()

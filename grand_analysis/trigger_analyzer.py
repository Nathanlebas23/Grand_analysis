import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Circle

c_light = 2.997924580e8

class TriggerAnalyzer:
    """
    Class for analyzing trigger patterns across detection units (DUs) and channels.
    
    This class provides methods to:
      - Compute the number of triggers per DU and per channel using trigger_pattern_ch.
      - Plot histograms of trigger counts for individual channels (X only, Y only) 
        and for events where both X and Y are triggered simultaneously.
      - Compute and plot the trigger rate (in Hz) total and per DU.
      
    It assumes that the data_processor object (accessible via self.data) provides:
      - self.data._du_ids: array of DU IDs per event.
      - self.data._trigger_pattern_ch: array of shape (n_events, n_channels) with boolean trigger data.
      - self.data._trigger_secs: array of trigger times (in seconds or other unit).
      - self.data._trigger_nanos: array of trigger nanoseconds.
    """
    def __init__(self, data_processor, reconstructor,  visualizer,  dt=2e-9):
        """
        Parameters:
        - data_processor: instance of DataProcessor containing the processed data.
        - dt: sampling interval in seconds (e.g., dt=2e-9 for a 500 MHz sampling rate).
        """
        self.data = data_processor
        self.dt = dt
        self.visualizer = visualizer
        self.reconstructor = reconstructor

    def get_trigger_counts(self):
        """
        Compute and return the trigger counts per DU using trigger_pattern_ch.
        
        For each event, on considère :
          - "X only" : canal X déclenché et canal Y non déclenché.
          - "Y only" : canal Y déclenché et canal X non déclenché.
          - "X & Y" : les deux canaux déclenchés simultanément.
        
        Returns:
          - unique_du_ids: 1D array of unique DU IDs.
          - n_dus: number of unique DUs.
          - trigger_counts: 2D array of shape (n_dus, 3) with counts for [X only, Y only, X & Y].
        """
        unique_du_ids = np.unique(self.data._du_ids)
        n_dus = len(unique_du_ids)
        n_event = len(self.data._du_ids)
        # 3 bins : X only, Y only, X & Y
        trigger_counts = np.zeros((n_dus, 3), dtype=int)
        
        for i, du in enumerate(unique_du_ids):
            mask = self.data._du_ids == du
            tp = self.data._trigger_pattern_ch[mask]  # shape: (n_events_for_du, n_channels)
            x_only = np.sum(tp[:, 1] & ~tp[:, 2])
            y_only = np.sum(tp[:, 2] & ~tp[:, 1])
            both   = np.sum(tp[:, 1] & tp[:, 2])
            trigger_counts[i, :] = [x_only, y_only, both]
        return n_event, unique_du_ids, n_dus, trigger_counts

    def plot_histograms_trigger_counts(self):
        """
        Affiche pour chaque DU un histogramme comparant les triggers sur les canaux:
          - 'X only' : seul le canal X a déclenché.
          - 'Y only' : seul le canal Y a déclenché.
          - 'X & Y'  : les deux canaux ont déclenché simultanément.
          
        On suppose ici que l'index 1 correspond au canal X et l'index 2 au canal Y.
        """
        n_event, unique_du_ids, n_dus, trigger_counts = self.get_trigger_counts()

        fig, axes = plt.subplots(4, 6, figsize=(20, 12))
        axes = axes.flatten()
        
        # Pour chaque DU, on trace un histogramme à trois barres.
        for i, du in enumerate(unique_du_ids):
            count_x_only = trigger_counts[i, 0]
            count_y_only = trigger_counts[i, 1]
            count_both   = trigger_counts[i, 2]
            axes[i].bar(['X only', 'Y only', 'X & Y'], [count_x_only, count_y_only, count_both],
                        color=['skyblue', 'salmon', 'forestgreen'])
            axes[i].set_title(f'DU {du}')
            axes[i].set_ylabel("Nombre of triggers")
            axes[i].set_ylim(0, n_event/n_dus * 2) 
        
        # Masquage des sous-plots non utilisés.
        for ax in axes[n_dus:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def compute_trigger_rate(self):
        """
        Compute the total trigger rate (in Hz) and the trigger rate per DU,
        using the global duration of the dataset.
        
        The event times are computed using:
            true_time = self.data._trigger_secs * 1e9 + self.data._trigger_nanos
        which are then converted to seconds.
        
        Returns:
          total_rate (float): Total trigger rate (Hz), computed as total triggers / global duration.
          per_du_rate (dict): Dictionary with DU IDs as keys and trigger rate (Hz) as values,
                              where each DU's rate is computed as (number of triggers for that DU) / (global duration).
        """
        # Calculer le temps vrai (en ns) pour chaque événement, puis le convertir en secondes.
        event_times_ns = self.data._trigger_secs * 1e9 + self.data._trigger_nanos
        event_times_s = event_times_ns / 1e9
        
        # Calculer la durée globale de la prise de données (en secondes).
        duration_total = event_times_s.max() - event_times_s.min()
        
        # Calcul du taux global : nombre total de déclenchements / durée totale.
        total_triggers = len(self.data._du_ids)
        total_rate = total_triggers / duration_total if duration_total > 0 else np.nan
        
        # Calcul du taux par DU en utilisant la même durée globale.
        unique_du_ids = np.unique(self.data._du_ids)
        per_du_rate = {}
        for du in unique_du_ids:
            mask = self.data._du_ids == du
            count = np.sum(mask)
            rate = count / duration_total if duration_total > 0 else np.nan
            per_du_rate[du] = rate
                
        return total_rate, per_du_rate

        
    def plot_trigger_rate(self):
        """
        Affiche un histogramme horizontal montrant le trigger rate (en Hz) pour chaque DU,
        trié par ordre croissant de trigger rate (le DU ayant le trigger rate maximal se trouve en haut).
        Le taux global est également indiqué dans le titre.
        """
        # Calculer le taux total et le taux par DU.
        total_rate, per_du_rate = self.compute_trigger_rate()
        
        # Trier les DU par trigger rate croissant.
        # sorted_items: liste de tuples (du, rate) triée par rate.
        sorted_items = sorted(per_du_rate.items(), key=lambda x: x[1])
        du_ids_sorted = [item[0] for item in sorted_items]
        rates_sorted = [item[1] for item in sorted_items]
        
        # Création de l'histogramme horizontal
        y_positions = np.arange(len(du_ids_sorted))
        plt.figure(figsize=(10, 6))
        plt.barh(y_positions, rates_sorted, color='skyblue')
        plt.yticks(y_positions, du_ids_sorted)
        plt.xlabel("Trigger Rate (Hz)")
        plt.ylabel("Detection Unit")
        plt.title(f"Trigger Rate par DU\n Global rate: {total_rate:.2f} Hz")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Annoter chaque barre avec sa valeur
        for i, rate in enumerate(rates_sorted):
            plt.text(rate + 0.005 * max(rates_sorted), i, f"{rate:.2f}", va='center', fontsize=12)
        
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


    def plot_trigger_rate_map(self):
        """
        Affiche la carte des déclenchements en utilisant les données de géolocalisation.
        Pour chaque DU, un cercle rempli est dessiné à la position (-loc.y, loc.x) avec un rayon
        proportionnel au trigger rate de ce DU. La couleur suit un gradient du colormap "Oranges".
        """
        total_rate, per_du_rate = self.compute_trigger_rate()
        unique_du, idx = self.data.get_unique_du()
        
        loc = self.get_geolocation_data() 
        
        fig, ax = plt.subplots()
        ax.plot(-loc.y, loc.x, 'ob', label="Detector Units")
        ax.plot(0, 0, 'or', label="Center Station")
        
        # Choix d'un facteur d'échelle : le DU avec le trigger rate maximal aura un cercle de rayon 200 m.
        max_rate = max(per_du_rate.values())
        min_rate = min(per_du_rate.values())
        scale = 300 / max_rate if max_rate != 0 else 1
        
        # Parcours des DUs pour ajouter les étiquettes et les cercles
        for i, du in enumerate(unique_du):
            center = (-loc.y[i], loc.x[i])
            ax.text(center[0] - 200, center[1] + 100, str(du), fontsize=12)
            rate = per_du_rate[du]
            # Calcul du rayon mis à l'échelle
            r = scale * rate
            # Normalisation pour le colormap "OrRd"
            normalized = (rate - min_rate) / (max_rate - min_rate) if max_rate != min_rate else 0
            color = plt.cm.OrRd(normalized)
            circle = Circle(center, r, facecolor=color, edgecolor=None)
            ax.add_patch(circle)
        
        norm = colors.Normalize(vmin=min_rate, vmax=max_rate)
        sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
        sm.set_array([])  
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Trigger Rate (Hz)')
    
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Geolocation of Detection Units with Trigger Rate")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def trigger_vs_time(self, bin_width=10):
        
        event_times_ns = self.data._trigger_secs * 1e9 + self.data._trigger_nanos
        event_times = event_times_ns / 1e9  
        
        unique_du_ids = np.unique(self.data._du_ids)
        
        t_min = event_times.min()
        t_max = event_times.max()
        bins = np.arange(t_min, t_max + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        plt.figure(figsize=(12, 8))
        
        for du in unique_du_ids:
            mask = self.data._du_ids == du
            du_times = event_times[mask]
            if len(du_times) == 0:
                continue
            counts, _ = np.histogram(du_times, bins=bins)
            rate = counts / bin_width
            plt.plot(bin_centers, rate, marker='o', label=f"DU {du}")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Trigger Rate (Hz)")
        plt.title("Evolution du taux de triggers par DU")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


    def trigger_count_vs_time(self, bin_width=10):
        """
        Plot the cumulative number of triggers per DU as a function of time.

        Parameters:
        - bin_width: Width of the time bin in seconds used for the histogram.
        """
        # Compute event times in seconds by combining trigger_times (assumed in seconds)
        # with trigger_nanos (assumed in nanoseconds).
        event_times = self.data._trigger_secs + self.data._trigger_nanos / 1e9

        # Get unique DU IDs from the data.
        unique_du_ids = np.unique(self.data._du_ids)

        # Determine the time range and create bins.
        t_min, t_max = event_times.min(), event_times.max()
        bins = np.arange(t_min, t_max + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create a new figure.
        plt.figure(figsize=(12, 8))
        
        # Loop over each DU and calculate the cumulative counts.
        for du in unique_du_ids:
            mask = self.data._du_ids == du
            du_times = event_times[mask]
            if len(du_times) == 0:
                continue
            counts, _ = np.histogram(du_times, bins=bins)
            cumulative_counts = np.cumsum(counts)
            plt.plot(bin_centers, cumulative_counts, marker='o', label=f"DU {du}")

        # Set plot labels and title.
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Trigger Count")
        plt.title("Cumulative Number of Triggers per DU vs Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


    def analyze_trigger_delays(self, do_plot=True, event_index=None):
        """
        Analyze the trigger arrival times for one event or animate all events if event_index is None.

        """
        # If event_index is None, animate over all events.
        if event_index is None:
            event_indices = self.reconstructor.get_event_indices()
            num_events = len(event_indices)
            if num_events == 0:
                print("No events available for analysis.")
                return None

            plt.ion()  # enable interactive mode
            fig = plt.figure(figsize=(10, 8))  # create a single figure for all events

            for idx, (start, end) in enumerate(event_indices):
                # Clear previous plots from the figure
                fig.clf()
                print(f"Processing event {idx+1}/{num_events} ...")
                
                # Retrieve DU coordinates for this event.
                du_coords = self.reconstructor._get_coord_ants_single(idx)
                if du_coords.size == 0:
                    print(f"No DU coordinates for event {idx}. Skipping...")
                    continue

                # Retrieve trigger times (ns) for the event.
                true_times = self.data.compute_true_time()
                if len(true_times) < 4 : 
                    print(f"Not enough trigger times for event {idx}. Skipping...")
                    continue
                else :
                    t_triggers = np.array(true_times[start:end])
                    sort_idx = np.argsort(t_triggers)
                    t_triggers = t_triggers[sort_idx]
                    du_coords = du_coords[sort_idx]
                    if hasattr(self.data, 'du_ids'):
                        # Extract the DU IDs for the current event and sort them accordingly.
                        du_ids_event = np.array(self.data._du_ids[start:end])[sort_idx]

                # Reconstruct the event using PWF to get theta and phi.
                result = self.reconstructor.reconstruct(event_index=idx)

                if result is None:
                    print(f"Reconstruction failed for event {idx}. Skipping...")
                    continue
                theta = result["theta"]
                phi = result["phi"]
                
                if np.isnan(theta) or np.isnan(phi):
                    print(f"Invalid angles for event {idx}. Skipping...")
                    continue

                # Define the shower axis as a unit vector.
                shower_axis = np.array([np.sin(theta) * np.cos(phi),
                                        np.sin(theta) * np.sin(phi),
                                        np.cos(theta)])
                # Compute expected delays by projecting DU positions onto the shower axis.
                t_expected = (du_coords.dot(shower_axis) / c_light) * 1e9  # in ns
                

                # Shift experimental times so that the first trigger is at t=0.
                t0 = t_triggers.min()
                t_exp = t_triggers - t0

                # Align the reconstructed delays using the first triggered DU.
                index_min = np.argmin(t_triggers)
                time_offset = - t_expected[index_min]
                t_rec = t_expected + time_offset
                t_rec = -t_rec  

                # Compute residuals.
                residuals = t_exp - t_rec

                # Compute the normalized chi-square.
                dof = len(t_exp) - 2
                chi2 = np.sum((residuals)**2) / dof / 1e2 if dof > 0 else np.nan

                if do_plot:
                    # Subplot 1: Δt_exp vs. Δt_rec with the ideal line (y = x).
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax1.plot(t_exp, t_rec, marker='o', ls='')
                    max_val = max(t_exp.max(), t_rec.max()) * 1.1
                    ax1.plot([0, max_val], [0, max_val], ls='--', color='r')
                    ax1.set_xlabel(r"$\Delta t_{\rm exp}$ [ns]")
                    ax1.set_ylabel(r"$\Delta t_{\rm rec}$ [ns]")
                    ax1.set_title(f"Plane Wave Rec. Event n°{idx} (χ² = {chi2:.2f})")
                    if hasattr(self.data, 'du_ids'):
                        du_ids_event = np.array(self.data._du_ids[start:end])
                        for j in range(len(t_exp)):
                            ax1.text(t_exp[j] + 100, t_rec[j], str(du_ids_event[j]), fontsize=12)
                    ax1.grid(True)

                    # Subplot 2: Residuals with fixed error bars (10 ns).
                    ax2 = fig.add_subplot(2, 1, 2)
                    ax2.errorbar(t_exp, residuals, yerr=10 * np.ones_like(residuals), marker='o', ls='')
                    ax2.set_xlabel(r"$\Delta t_{\rm exp}$ [ns]")
                    ax2.set_ylabel(r"$\Delta t_{\rm exp} - \Delta t_{\rm rec}$ [ns]")
                    if hasattr(self.data, 'du_ids'):
                        for j in range(len(t_exp)):
                            ax2.text(t_exp[j] + 10, residuals[j], str(np.array(self.data._du_ids)[start+j]), fontsize=12)
                    ax2.grid(True)
                    fig.tight_layout()
                    plt.draw()         # update the current figure
                    plt.pause(4)       # pause for 4 seconds before the next update

            plt.ioff()  # disable interactive mode
            # plt.show()  # display the final figure if needed
            return

        # If event_index is provided, analyze that specific event.

        # Get the list of event indices (each as a (start, end) tuple)
        event_indices = self.reconstructor.get_event_indices()
        if event_index >= len(event_indices):
            raise IndexError("Event index out of range.")
        start, end = event_indices[event_index]

        # Retrieve the antenna coordinates for this event using the dedicated helper.
        du_coords = self.reconstructor._get_coord_ants_single(event_index)
        if du_coords.size == 0:
            print("No DU coordinates available for event {}. Aborting analysis.".format(event_index))
            return None

        # Retrieve trigger arrival times (in ns) for all triggers and extract those for this event.
        true_times = self.data.compute_true_time()

        # Extract the trigger times for the event of interest
        t_triggers = np.array(true_times[start:end])

        # Sort the trigger times and adjust the corresponding DU coordinates (and DU IDs if available)
        sort_idx = np.argsort(t_triggers)
        t_triggers = t_triggers[sort_idx]
        du_coords = du_coords[sort_idx]

        if hasattr(self.data, 'du_ids'):
            # Extract the DU IDs for the current event and sort them accordingly.
            du_ids_event = np.array(self.data._du_ids[start:end])[sort_idx]
            

        if t_triggers.size == 0:
            print("No trigger times available for event {}. Aborting analysis.".format(event_index))
            return None

        # Reconstruct the event using plane-wave reconstruction to obtain theta and phi (in radians).
        result = self.reconstructor.reconstruct(event_index=event_index)
        if result is None:
            print("Reconstruction failed for event {}. Aborting analysis.".format(event_index))
            return None
        theta = result["theta"]
        phi = result["phi"]
        print('θ =', np.round(theta * 180 / np.pi, 2), '°')
        print('ϕ =', np.round(phi * 180 / np.pi, 2), '°')

        if np.isnan(theta) or np.isnan(phi):
            print("Invalid reconstructed angles (NaN encountered) for event {}. Aborting analysis.".format(event_index))
            return None

        # Define the shower axis as a unit vector: (sinθ*cosφ, sinθ*sinφ, cosθ)
        shower_axis = np.array([np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(theta)])

        # Compute expected delays by projecting DU positions onto the shower axis.
        # t_expected = (position dot shower_axis) / c, converted to ns.
        t_expected = du_coords.dot(shower_axis) / c_light 
        print("t_expected", t_expected)

        # Shift experimental times so that the first trigger is at t=0.
        t0 = t_triggers.min()
        t_exp = t_triggers - t0
        print("t_exp", t_exp)
        # Use the first triggered DU (the one with the minimum experimental time) as the reference.  /!\ a vérifier ici si ca match 
        index_min = np.argmin(t_triggers)   # index of the minimum trigger time shloud alwas be zero
        time_offset = - t_expected[index_min] # t0
        t_rec = t_expected + time_offset
        t_rec = - t_rec 

        # Compute the residuals and normalized chi-square.
        residuals = t_exp - t_rec
        dof = len(t_exp) - 2
        chi2 = np.sum((residuals)**2) / dof / 1e2 if dof > 0 else np.nan  # Voir si ca c'est ok aussi 

        if do_plot:
            plt.figure(figsize=(10, 8))
            # Subplot 1: Δt_exp vs. Δt_rec with ideal line y = x.
            plt.subplot(2, 1, 1)
            plt.plot(t_exp, t_rec, marker='o', ls='')
            max_val = max(t_exp.max(), t_rec.max()) * 1.1
            plt.plot(np.array([0, max_val]), np.array([0, max_val]), ls='--', color='r')
            plt.xlabel(r"$\Delta t_{\rm exp}$ [ns]")
            plt.ylabel(r"$\Delta t_{\rm rec}$ [ns]")
            plt.title(f"Plane Wave Rec. Event n°{event_index} of the file")
            # Annotate each point with DU ID if available.
            if hasattr(self.data, 'du_ids'):
                du_ids_event = self.data._du_ids[start:end]
                for j in range(len(t_exp)):
                    plt.text(t_exp[j] + 100, t_rec[j], str(du_ids_event[j]), fontsize=12)
            plt.grid(True)

            # Subplot 2: Residuals with error bars (10 ns).
            plt.subplot(2, 1, 2)
            plt.errorbar(t_exp, residuals, yerr=10 * np.ones_like(residuals), marker='o', ls='')
            plt.xlabel(r"$\Delta t_{\rm exp}$ [ns]")
            plt.ylabel(r"$\Delta t_{\rm exp} - \Delta t_{\rm rec}$ [ns]")
            if hasattr(self.data, 'du_ids'):
                for j in range(len(t_exp)):
                    plt.text(t_exp[j] + 10, residuals[j], str(self.data._du_ids[start + j]), fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # print("Event", event_index)
        # print("Chi2 plane =", chi2, " (theta, phi) =", (np.degrees(theta), np.degrees(phi)))
        return chi2, t_exp, t_rec, residuals




    def histo_chi2(self):
        """
        Compute χ² values for all events using plane-wave reconstruction (PWF)
        and plot the event index vs. χ²₍PWF₎.

        If event_index is None, the method processes all events.
        If event_index is provided, a single-event analysis is performed (but here we require None for histogram).

        Returns:
            None. (Displays a plot.)
        """

        chi2_list = []
        event_indices = self.reconstructor.get_event_indices()
        n_events = len(event_indices)
        
        # Loop over all events and compute χ² for each event.
        for i in range(n_events):
            result = self.analyze_trigger_delays(do_plot=False, event_index=i)
            # print(f"Processing event {i}...")
            if result is None:
                # print(f"Skipping event {i} due to missing data.")
                continue
            chi2, t_exp, t_rec, residuals = result
            chi2_list.append(chi2)
        
        chi2_arr = np.array(chi2_list)
        
            
        # Create a single plot: event index vs. χ²₍PWF₎.
        plt.figure(figsize=(10, 6))
        plt.hist(chi2_arr,  np.logspace(-2, 6, 100), label=r'$\chi^2_{PWF}$')
        plt.ylabel("Number of Events")
        plt.xlabel(r"$\chi^2_{PWF}$")
        plt.title("χ² (PWF) vs. Event Index")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

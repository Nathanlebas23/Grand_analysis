import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Circle

class TriggerAnalyzer:
    """
    Class for analyzing trigger patterns across detection units (DUs) and channels.
    
    This class provides methods to:
      - Compute the number of triggers per DU and per channel using trigger_pattern_ch.
      - Plot histograms of trigger counts for individual channels (X only, Y only) 
        and for events where both X and Y are triggered simultaneously.
      - Compute and plot the trigger rate (in Hz) total and per DU.
      
    It assumes that the data_processor object (accessible via self.data) provides:
      - self.data.du_ids: array of DU IDs per event.
      - self.data.trigger_pattern_ch: array of shape (n_events, n_channels) with boolean trigger data.
      - self.data.trigger_times: array of trigger times (in seconds or other unit).
      - self.data.trigger_nanos: array of trigger nanoseconds.
    """
    def __init__(self, data_processor, visualizer,  dt=2e-9):
        """
        Parameters:
        - data_processor: instance of DataProcessor containing the processed data.
        - dt: sampling interval in seconds (e.g., dt=2e-9 for a 500 MHz sampling rate).
        """
        self.data = data_processor
        self.dt = dt
        self.visualizer = visualizer

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
        unique_du_ids = np.unique(self.data.du_ids)
        n_dus = len(unique_du_ids)
        n_event = len(self.data.du_ids)
        # 3 bins : X only, Y only, X & Y
        trigger_counts = np.zeros((n_dus, 3), dtype=int)
        
        for i, du in enumerate(unique_du_ids):
            # Sélectionne les événements correspondant à ce DU.
            mask = self.data.du_ids == du
            tp = self.data.trigger_pattern_ch[mask]  # shape: (n_events_for_du, n_channels)
            # Calcul des triggers :
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

        # Création d'une grille de sous-plots : ici 4 lignes x 6 colonnes pour 22 DUs.
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
            true_time = self.data.trigger_times * 1e9 + self.data.trigger_nanos
        which are then converted to seconds.
        
        Returns:
          total_rate (float): Total trigger rate (Hz), computed as total triggers / global duration.
          per_du_rate (dict): Dictionary with DU IDs as keys and trigger rate (Hz) as values,
                              where each DU's rate is computed as (number of triggers for that DU) / (global duration).
        """
        # Calculer le temps vrai (en ns) pour chaque événement, puis le convertir en secondes.
        event_times_ns = self.data.trigger_times * 1e9 + self.data.trigger_nanos
        event_times_s = event_times_ns / 1e9
        
        # Calculer la durée globale de la prise de données (en secondes).
        duration_total = event_times_s.max() - event_times_s.min()
        
        # Calcul du taux global : nombre total de déclenchements / durée totale.
        total_triggers = len(self.data.du_ids)
        total_rate = total_triggers / duration_total if duration_total > 0 else np.nan
        
        # Calcul du taux par DU en utilisant la même durée globale.
        unique_du_ids = np.unique(self.data.du_ids)
        per_du_rate = {}
        for du in unique_du_ids:
            mask = self.data.du_ids == du
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

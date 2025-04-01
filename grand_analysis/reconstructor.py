import numpy as np
import matplotlib.pyplot as plt
from PWF_reconstruction.recons_PWF import PWF_semianalytical
from mpl_toolkits.mplot3d import Axes3D  # Nécessaire pour la projection 3D
from grand import Geodetic, GRANDCS


class Reconstructor:
    def __init__(self, data_processor, visualizer, event_index=0):
        """
        :param data_processor: Instance de DataProcessor avec les données traitées.
        :param visualizer: Instance de Visualizer, utilisée ici pour la conversion des coordonnées.
        :param event_index: L'index de l'événement à analyser (0 par défaut).
        """
        self.data = data_processor
        self.visualizer = visualizer
        self.event_index = event_index

    def _get_event_indices(self):
        """
        Regroupe les indices de triggers par événement en se basant sur la multiplicité.
        """
        event_indices = []
        i = 0
        multiplicities = self.data.multiplicities  # number of trigger per event 
        while i < len(multiplicities):
            m = multiplicities[i]
            event_indices.append((i, i + m))  # index events are : 1-4 fo exemple 
            i += m
        return event_indices
    

    def get_coord_ants(self):
        """
        Extrait les coordonnées GPS des antennes déclenchées pour l'événement choisi,
        puis les convertit en coordonnées cartésiennes via Geodetic et GRANDCS.
        
        Renvoie un tableau de forme (n_ants, 3), avec n_ants = nombre de déclenchements dans l'événement.
        """
        event_indices = self._get_event_indices()
        if self.event_index >= len(event_indices):
            raise IndexError("Le numéro d'événement demandé est hors limites.")
        start, end = event_indices[self.event_index]  # for exemple 1-4
        
        # Extraction des coordonnées GPS pour l'événement
        event_lat = self.data.du_lat[start:end]
        event_lon = self.data.du_long[start:end]
        event_alt = self.data.du_alt[start:end]
        
        # Définition de la station centrale (à adapter selon vos données réelles)
        daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)
        event_geo = Geodetic(latitude=event_lat, longitude=event_lon, height=event_alt)
        event_geo_cs = GRANDCS(event_geo, obstime="2024-09-15", location=daq)
        
        # Constitution du tableau de coordonnées : chaque ligne correspond à une antenne (n_ants, 3)
        coords = np.column_stack((event_geo_cs.x, event_geo_cs.y, event_geo_cs.z))
        return np.array(coords)  
    
    def get_t_ants(self):
        """
        Extrait les temps d'arrivée (en nanosecondes) des triggers pour l'événement choisi.
        
        Renvoie un tableau de forme (n_ants,).
        """
        event_indices = self._get_event_indices()
        if self.event_index >= len(event_indices):
            raise IndexError("Le numéro d'événement demandé est hors limites.")
        start, end = event_indices[self.event_index]
        
        true_times = self.data.compute_true_time()  # Temps pour tous les triggers
        t_ants = np.array(true_times[start:end])
        return t_ants

    def reconstruct(self):
        '''
        
        '''
        x_ants = self.get_coord_ants()  # (n_ants, 3)
        t_ants = self.get_t_ants()        # (n_ants,)
        theta, phi = PWF_semianalytical(x_ants, t_ants)
        return theta, phi

    
    def reconstruct_all_events(self):
        event_indices = self._get_event_indices()
        thetas = []
        phis = []

        for i, (start, end) in enumerate(event_indices):
            self.event_index = i
            theta, phi = self.reconstruct()
            
            # Let's filtered problematic events
            if np.isnan(theta).all() or (hasattr(phi, '__len__') and len(phi) == 0):
                continue
            thetas.append(theta)
            phis.append(phi)

        return np.array(thetas), np.array(phis)
    

    def plot_3D_sphere(self):
        """
        Plot the direction of thetas et phis on a 3D sphere.
        Thetas et Phis must be in radiant.
        """
        thetas, phis = self.reconstruct_all_events()

        xs = np.sin(thetas) * np.cos(phis)
        ys = np.sin(thetas) * np.sin(phis)
        zs = np.cos(thetas)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Tracé de la sphère unitaire (optionnel, pour contexte)
        # On peut tracer une "wireframe" pour représenter la sphère :
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        sphere_x = np.outer(np.sin(v), np.cos(u))
        sphere_y = np.outer(np.sin(v), np.sin(u))
        sphere_z = np.outer(np.cos(v), np.ones_like(u))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='lightgray', alpha=0.3)

        # Tracé des points directionnels
        ax.scatter(xs, ys, zs, color='red', s=10, alpha=0.8)

        # Réglages divers
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Distribution on the direction of the sphere")

        # On fixe les limites pour voir la sphère entière
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.show()


    def plot_2D_sphere(self):
        """
        Affiche une projection 2D sur une "sphère" en utilisant une projection polaire.
        
        Paramètres :
        - thetas : tableau des angles zénith (en radians) à convertir en degrés pour le rayon.
        - phis   : tableau des angles azimutaux (en radians) pour l'angle de la projection.
        """

        thetas, phis = self.reconstruct_all_events()


        # Conversion de theta en degrés pour l'affichage
        thetas_deg = np.degrees(thetas)
        
        # Optionnel : on peut limiter (clipper) theta à 100° pour le rayon
        thetas_deg = np.clip(thetas_deg, 0, 100)
        
        # Création de la figure en projection polaire
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        
        # Affichage des points
        ax.scatter(phis, thetas_deg, s=10, color='red', alpha=0.8)
        
        # Configuration de la projection :
        ax.set_theta_zero_location("N")   # 0° (phi = 0) en haut (direction Nord)
        ax.set_theta_direction(1)        # les angles augmentent dans le sens horaire
        
        # Définition des limites du rayon (pour theta, en degrés)
        ax.set_rlim(0, 100)
        
        # Optionnel : personnaliser les ticks angulaires en degrés
        tick_angles = np.arange(0, 360, 45)
        ax.set_xticks(np.radians(tick_angles))
        ax.set_xticklabels([f"{angle}°" for angle in tick_angles])
        
        ax.set_title("Azimut (φ) et zenith (θ)", va='bottom')
        plt.show()

    # def histo_thetaphi(self):
        
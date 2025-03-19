import os
import numpy as np
import matplotlib.pyplot as plt

class CosmicRayReconstructor:
    """
    Classe de reconstruction permettant de reconstruire le rayon cosmique à partir des données.
    """
    def __init__(self, recons_path, daq=None, daqm=None):
        """
        Paramètres:
          - recons_path: chemin vers le dossier contenant les fichiers de reconstruction.
          - daq: tuple (xdaq, ydaq, zdaq) pour la position de la DAQ. (Par défaut: (490, 501, 1320))
          - daqm: tuple (xdaqm, ydaqm, zdaqm) pour une position alternative de la DAQ. (Par défaut: (406, 456, 1220))
        """
        # Constantes
        self.kc = 299792458.  # Vitesse de la lumière en m/s
        self.kc_ns = self.kc * 1e-9  # en m/ns
        self.kn = 1.0001  # Indice de réfraction (peut être ajusté)
        self.kcn_ns = self.kc_ns / self.kn
        self.sig_t = 15  # Résolution temporelle en ns

        self.recons_path = recons_path
        
        # Positions DAQ par défaut si non fournies
        self.daq = daq if daq is not None else (490, 501, 1320)
        self.daqm = daqm if daqm is not None else (406, 456, 1220)
        
        # Chargement et traitement des données de reconstruction.
        # self.load_data()
        # self.process_reconstruction()


    def process_reconstruction(self):
        """
        Effectue le traitement des données de reconstruction :
          - Correction de la convention angulaire pour phi_rec.
          - Calcul de quelques distributions (ex. : distance entre la source reconstruite et la DAQ).
          - Pour chaque coïncidence, calcule les délais (tsphere_rec, tsphere_daq, peakt_sel).
          - Calcule un ch² par événement.
        """
        # Correction de phi_rec (ajouter 180° et ramener dans [0, 360))
        self.phi_rec = (self.phi_rec + 180.) % 360.

        # Distance entre la source reconstruite et la DAQ
        xdaq, ydaq, zdaq = self.daq
        self.dist_daq = np.linalg.norm(
            np.array([self.xrec - xdaq, self.yrec - ydaq, self.zrec - zdaq]), axis=0)

        # Initialisation des listes pour stocker les délais et autres paramètres
        self.tsphere_rec = []  # délais calculés depuis la position reconstruite
        self.tsphere_daq = []  # délais calculés depuis la DAQ
        self.peakt_sel = []    # temps de pic en ns (après conversion)
        # Boucle sur les coïncidences uniques
        sel = np.arange(len(self.xrec))
        unique_recoinc = np.unique(self.recoinc[sel])
        for icoinc in unique_recoinc:
            sel_coinc = np.where(self.coincid == icoinc)[0]
            xant_coinc = self.xant[sel_coinc]
            yant_coinc = self.yant[sel_coinc]
            zant_coinc = self.zant[sel_coinc]
            peakt_coinc = self.peakt[sel_coinc]
            # On prend le premier indice pour la reconstruction (sel_rec_idx)
            sel_rec_idx = np.where(self.recoinc == icoinc)[0][0]
            # Calcul des distances entre les antennes et la source reconstruite
            dist_ant_rec = np.linalg.norm(
                np.array([xant_coinc - self.xrec[sel_rec_idx],
                          yant_coinc - self.yrec[sel_rec_idx],
                          zant_coinc - self.zrec[sel_rec_idx]]), axis=0)
            # Calcul des distances entre les antennes et la DAQ
            dist_ant_daq = np.linalg.norm(
                np.array([xant_coinc - xdaq,
                          yant_coinc - ydaq,
                          zant_coinc - zdaq]), axis=0)
            self.tsphere_rec.append(dist_ant_rec / self.kcn_ns)
            self.tsphere_daq.append(dist_ant_daq / self.kcn_ns)
            self.peakt_sel.append(peakt_coinc * 1e9)

        # Calcul du chi2 par événement (pour les délais entre t_model et t_data)
        self.chi2ndf_delays = []
        # Initialisation de matrices pour stocker d'autres délais (pour 7 antennes par exemple)
        self.ds_daq = np.zeros((len(self.nant), 7))
        self.t_datastack = np.zeros((len(self.nant), 7))
        for i in range(len(self.nant)):
            sel_rec = np.where(self.coincid == i)[0]
            if len(sel_rec) == 0:
                self.chi2ndf_delays.append(np.nan)
                continue
            t_data = self.peakt_sel[i] - np.min(self.peakt_sel[i])
            t_model = self.tsphere_rec[i] - np.min(self.tsphere_rec[i])
            t_daq = self.tsphere_daq[i] - np.min(self.tsphere_daq[i])
            d_model = t_model - t_data
            # Ajustement en retirant la moyenne
            t_model = t_model - np.mean(d_model)
            d_model = d_model - np.mean(d_model)
            d_daq = t_daq - t_data
            t_daq = t_daq - np.mean(d_daq)
            d_daq = d_daq - np.mean(d_daq)
            if len(t_data) > 4:
                chi2 = np.sum(d_model**2) / (self.sig_t**2) / (len(t_data) - 4)
            else:
                chi2 = np.nan
            self.chi2ndf_delays.append(chi2)
            # Sauvegarde des délais si 7 antennes sont impliquées
            if len(d_daq) == 7:
                isort = np.argsort(self.antid[sel_rec])
                self.ds_daq[i] = d_daq[isort]
                self.t_datastack[i] = t_data[isort]


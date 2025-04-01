#!/usr/bin/python
# Ce script permet de visualiser de manière organisée et interactive les données d'un fichier GP13.
# Il utilise l'interface orientée analyse de GRANDlib pour charger et afficher les traces.
#
# Usage: ./aoi.py [chemin_du_fichier]

# from grand.grandlib_classes.grandlib_classes import *
from grand.aoi import *
import matplotlib.pyplot as plt
import sys
import ROOT
import numpy as np
import time

############# Useful interface functions #############
def arrays2canvas(x, y=None, dx=None, dy=None, drawoption="A*", z=None, dz=None, x_shift=0):
    """
    Crée un TGraph à partir des tableaux numpy et l'affiche dans un canvas ROOT.
    """
    g = arrays2graph(x, y, dx, dy, z, dz, x_shift)
    c = ROOT.TCanvas()
    g.Draw(drawoption)
    c.graph = g
    return c

def arrays2graph(x, y=None, dx=None, dy=None, z=None, dz=None, x_shift=0):
    """
    Crée un TGraph à partir de deux tableaux numpy (ou plus, pour TGraphErrors ou TGraph2D).
    """
    if y is not None:
        x = np.array(x)
        y = np.array(y)
    else:
        y = np.arange(len(x))
        x = np.array(x)
        x, y = y, x
    x += x_shift

    if z is None:
        if dx is not None:
            dx = np.array(dx)
        if dy is not None:
            dy = np.array(dy)
        if dx is None and dy is None:
            g = ROOT.TGraph(len(x), x.astype(np.float64), y.astype(np.float64))
        elif dx is None and dy is not None:
            dx = np.zeros(len(x))
            g = ROOT.TGraphErrors(len(x), x.astype(np.float64), y.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64))
        elif dy is None and dx is not None:
            dy = np.zeros(len(x))
            g = ROOT.TGraphErrors(len(x), x.astype(np.float64), y.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64))
    else:
        z = np.array(z)
        if dz is not None:
            dz = np.array(dz).astype(np.float64)
        if dx is None and dy is None and dz is None:
            g = ROOT.TGraph2D(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64))
        elif dx is None and dy is not None:
            dx = np.ones(len(x))
            g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64),
                                      dx.astype(np.float64), dy.astype(np.float64), dz)
        elif dy is None and dx is not None:
            dy = np.ones(len(x))
            g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64),
                                      dx.astype(np.float64), dy.astype(np.float64), dz)
        elif dx is not None and dy is not None:
            g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64),
                                      dx.astype(np.float64), dy.astype(np.float64), dz)
        else:
            dx = np.ones(len(x))
            dy = np.ones(len(x))
            g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64),
                                      dx, dy, dz)
    g.SetTitle("")
    return g

def wait4key():
    """
    Attend que l'utilisateur entre 'c' pour continuer ou 'q' (ou 'x') pour quitter.
    """
    rep = ''
    while rep.lower() not in ['q', 'c', 'x']:
        rep = input('Entrer "c" pour continuer, "q" pour quitter : ')
        if rep.lower() in ['q', 'x']:
            sys.exit()

    return


############# Results functions #############


def plot_shower(event_list, event_index=3):
    """
    Reconstruit et affiche la géométrie d'une shower à partir d'un événement donné.
    """
    # Récupérer l'événement
    event = event_list.get_event(entry_number = event_index)
    if event is None:
        print(f"L'événement {event_index} n'existe pas.")
        return

    index = event.event_number

    # Récupérer l'objet shower associé à l'événement
    shower = event.Shower
    if shower is None:
        print(f"L'événement {event_index} ne contient pas d'information de shower.")
        return

    # Maintenant, on peut accéder aux propriétés de la shower
    core = shower.core_ground_pos      # position du cœur au sol (ex: (x, y))
    xmax = shower.Xmaxpos               # position de Xmax (ex: (x, y))
    origin = shower.origin_geoid        # direction d'origine (ex: (dx, dy))
    arrow_scale = 100.0                 # facteur d'échelle pour la flèche

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(core[0], core[1], 'ro', markersize=8, label="Core ground position")
    ax.plot(xmax[0], xmax[1], 'bo', markersize=8, label="Xmax position")
    
    ax.arrow(core[0], core[1], arrow_scale * origin[0], arrow_scale * origin[1],
             head_width=10, head_length=15, fc='green', ec='green', label="Shower origin direction")
    
    textstr = (
        f"Energy (EM): {shower.energy_em:.1f} GeV\n"
        f"Energy (primary): {shower.energy_primary:.1f} GeV\n"
        f"Xmax: {shower.Xmax:.1f} g/cm²\n"
        f"Azimuth: {shower.azimuth:.1f}°\n"
        f"Zenith: {shower.zenith:.1f}°"
    )
    ax.text(core[0] + 20, core[1] + 20, textstr, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Reconstruction de la Shower")
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_trigger_histogram(el):
    """
    Parcourt l'EventList pour compter, pour chaque du_id,
    le nombre de fois où une trace (Voltage) déclenchée est enregistrée.
    Affiche ensuite un histogramme matplotlib (vertical) où l'axe des y représente les DU et l'axe des x le nombre de triggers.
    """
    counts = {}
    compteur = 0
    ne = el.get_number_of_events()
    

    start_event = time.time_ns()

    for i, e in enumerate(el):
        if i > 99 : 
            break

        for voltage in e.voltages:  # Voltages from different antennas
            if voltage.is_triggered:  # is_triggered: bool = True (appartenant à la classe Voltage)
                du = voltage.du_id
                counts[du] = counts.get(du, 0) + 1
                # print(counts)
        # Affiche la progression toutes les 10 itérations
    end_event = time.time_ns()
    print("Temps d'exécution : {:.3f} nano_secondes".format(end_event - start_event))

        # if i % 10 == 0:
        #     print("i_fin :", i)
        #     end_event = time.time_ns()
        #     print("Temps d'exécution : {:.3f} nano_secondes".format(end_event - start_event))
    
    if not counts:
        print("Aucun trigger trouvé.")
        return

    # Tri par nombre de triggers décroissant
    sorted_by_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    # Préparation des données pour le graphique : DU en y et le nombre de triggers en x
    du_ids = [str(item[0]) for item in sorted_by_count]
    counts_values = [item[1] for item in sorted_by_count]

    # Création du graphique avec matplotlib

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(du_ids, counts_values, color='lightgreen')
    ax.set_xlabel("Nombre de triggers")
    ax.set_ylabel("DU ID")
    ax.set_title("Triggers par DU")
    ax.invert_yaxis()  # Le DU avec le plus grand nombre de triggers en haut
    plt.tight_layout()
    plt.show()


def affiche_event(e):
    """
    Affiche les informations d'un événement et trace les courbes de tension (composantes X, Y, Z).
    """
    index = e.event_number
    print(f"\n=== Evénement {index} ===")
    print(f"Run number : {e.run_number}, Event number : {e.event_number}")
    if e.voltages and len(e.voltages) > 0:
        voltage = e.voltages[0]
        print(f"du_id : {voltage.du_id}")
        print(f"t0 : {voltage.t0}")
        print(f"Nombre de points de la trace : {voltage.n_points}")
    else:

        print("Aucune information de tension disponible.")
        return

    # Récupération du vecteur temps et des traces (X, Y, Z)
    t = voltage.t_vector
    try:
        x = voltage.trace.x
        y = voltage.trace.y
        z = voltage.trace.z
    except Exception as exc:
        print("Erreur lors de l'accès aux données de trace :", exc)
        return


    # Création des TGraph pour chaque composante
    g_z = arrays2graph(t, z) # noir
    g_z.SetTitle(f"Evnement {index} - du_id {voltage.du_id}\n")
    g_z.GetXaxis().SetTitle("Temps [ns]")
    g_z.GetYaxis().SetTitle("Tension brute [V]")

    g_x = arrays2graph(t, x)
    g_x.SetLineColor(2)  # rouge
    g_x.SetTitle("")
    g_y = arrays2graph(t, y)
    g_y.SetLineColor(3)  # vert
    g_y.SetTitle("")

    # Création d'un canvas pour afficher les traces superposées
    c_trace = ROOT.TCanvas(f"c_trace_{index}", f"Traces de l'événement {index}", 1200, 600)
    # On dessine la trace Z, puis on superpose X et Y
    g_z.Draw("AL")
    g_x.Draw("same L")
    g_y.Draw("same L")
    c_trace.Update()

    # On attend que l'utilisateur appuie sur Entrée pour passer à l'événement suivant
    input("Appuyez sur Entrée pour passer à l'événement suivant...")
    c_trace.Close()

def main():
    start_time = time.time()
    # Lecture du fichier (passé en argument ou par défaut)
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "/home/nlebas/grand/GP80_20250312_190857_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-612.root"

    print("Lecture du fichier :", file_name)

    el = EventList(file_name, use_trawvoltage=True)
    
    # Affichage de quelques infos sur les événements 
    ne = el.get_number_of_events()
    print(f"Nombre total d'événements : {ne}")
    
    # Affichage de l'événement 3
    e = el.get_event(entry_number=3)
    affiche_event(e)

    # Vérifier la présence d'informations de shower
    if e.shower is not None:
        plot_shower(el, event_index=3)
    else:
        print("L'événement 3 ne contient pas d'information de shower. Aucune visualisation de shower possible.")

    end_time = time.time()
    print("Temps d'exécution : {:.3f} secondes".format(end_time - start_time))

if __name__ == '__main__':
    main()
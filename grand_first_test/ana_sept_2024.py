#### Test Nathan : Prise main des outils ###


# Deco mémo ----- On regarde chacun des événements, c'est à dire lorsqu'un signal est récu par un canal (branche d'une antenne). On regarde un run, c'est 
# à dire une durée d'activité pouvant correspondre à un rayon cosmique

import numpy as np
import matplotlib.pyplot as plt
# import glob
from scipy.fft import rfftfreq, rfft, irfft
from grand import ECEF, Geodetic, GRANDCS, LTP
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
import os 
import datetime
import grand.dataio.root_trees as rt  # Permet d'extraire les attributs

from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet
from grand import Topography, Reference, geoid_undulation
from grand import topography



data_dir = "/home/nlebas/grand/examples/grandlib_classes/"


flist=[data_dir+"GP80_20250309_235256_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root"] # Voici le fichier à charger mis dans une liste



list_trigger_time = [] # Time of trigger in seconds
list_trigger_nano = [] # Time of trigger in nanoseconds
list_traces_MD = []
list_traces = []
list_time_MD = []
list_du_id_MD = []
list_du_id = []
list_du_long = []
list_du_lat = []
list_du_alt = []
list_mult = []   


for fname in flist:
        
    root_file = rt.DataFile(fname)  # DataFile charge les données root 
    n = root_file.tadc.get_number_of_entries() # Le nombre d'entries est le nombre de déclanchements pouvant correspondre à un rayon cosmique
    # root_file est une instance de DataFile, il partage toutes les méthodes de DataFile. On choisit de regarder le canal tdac 
    # number_of_entries donne le nombre de d'informations contenues dans le fichier --> Nombre d'événement

    # print(root_file.trawvoltage.du_count) # 0


    for i in range(n):  # On regarde chaque coincidence
      #if i/100 == int(i/100):   
        #print(i,n)      # Affiche la progression tous les 100 événements
      root_file.tadc.get_entry(i)   # Donne la valeur de adc de la coincidence i  ----> Remarque : Pas présent dans la doc
      root_file.trawvoltage.get_entry(i) # Donne les valeurs adc brutes de la coincidence i  ----> Remarque : Pas présent dans la doc
      # Pourquoi faire ca ????


      if ~np.all(root_file.tadc.trigger_pattern_10s) and np.any(root_file.tadc.trigger_pattern_ch):  # Vérifie qu'il y ai au moins un False dans trigger_pattern_10s
                                                                                                     # vérifie si au moins une valeur du tableau trigger_pattern_ch est True
        # UD data
        #print("0:",root_file.tadc.du_seconds[0])
        if root_file.tadc.du_seconds[0]>2045518457:  # Skip wrong GPS time, du_second[0] est le temps de déclanchement si il est abbérant on l'exclu
           continue
        

        multiplicity = len(root_file.tadc.du_id)
        list_mult.append(multiplicity)   
        #print(root_file.tadc.du_id, root_file.trawvoltage.gps_long) #  /!\ Il semble y avoir des erreurs de valeurs dans du_id
        

        # On remplit les différentes listes contenant les informations 
        # print (root_file.tadc.du_id)
        list_trigger_time.append(root_file.tadc.du_seconds[0])  # On recupére le temps de déclanchement de chaque coincidence
        list_trigger_nano.append(root_file.tadc.du_nanoseconds[0])
        list_du_id.append(root_file.tadc.du_id[0]) # On récupère l'identité de la première source déclanchée ?????
        list_du_long.append(root_file.trawvoltage.gps_long[0])  # On regarde la longitude gps associé au raw voltage pour chaque DU
        list_du_lat.append(root_file.trawvoltage.gps_lat[0])    # On regarde la latitude gps associé au raw_voltage
        list_du_alt.append(root_file.trawvoltage.gps_alt[0])    # On regarde l'altitude associé au raw_voltage
        
        # Refléxion : Ici on récuprère que les première antennes à chaque fois, on ne perd pas de l'information ?


        # print( n in root_file.tadc.trace_ch[0][0])  # On regarde si le nombre d'événement est dans les données de trace_ch[0][0]
        _traces = [] # A chaque ligne de data. _traces prend [[raw_data],[x],[y],[z]]
        _traces.append(root_file.tadc.trace_ch[0][0])  # On récupère les données brutes de l'événement
        _traces.append(root_file.tadc.trace_ch[0][1])  # On récupère les données x de l'événement
        _traces.append(root_file.tadc.trace_ch[0][2])  # On récupère les données y de l'événement
        _traces.append(root_file.tadc.trace_ch[0][3])  # On récupère les données z de l'événement
        list_traces.append(_traces) # list_traces prend [[[raw_data],[x],[y],[z]], [[raw_data],[x],[y],[z]], [[raw_data],[x],[y],[z]]]
 
# print(len(list_traces))        # Nombre de déclenchement  --> n = 891
# print(len(list_traces[0]))     # Nombre de traces par déclenchement --> 4
# print(len(list_traces[0][0]))  # Nombre de points par trace --> 1024


# list_trigger_time should already be sorted in the root file


# Convert list to np.array
list_trigger_time = np.array(list_trigger_time)  # seconds
list_trigger_nano = np.array(list_trigger_nano)  # nanseconds
list_du_id = np.array(list_du_id)
list_time_MD = np.array(list_time_MD)
list_du_id_MD = np.array(list_du_id_MD)
list_traces_MD = np.array(list_traces_MD)
list_traces = np.array(list_traces)
list_du_long = np.array(list_du_long)
list_du_lat = np.array(list_du_lat)
list_du_alt = np.array(list_du_alt)
list_mult = np.array(list_mult)

# print("trigger time", list_trigger_time)
# print("trigger nano", list_trigger_nano)
# print("list id", list_du_id)   # Me permetra d'affciher la liste de ce run
# print("list time MD", list_time_MD) # Vide ici 
# print("list traces MD", list_traces_MD) # Vide ici
# print("list traces", list_traces)
# print("list long", list_du_long)  
# print("list lat", list_du_lat)
# print("list alt", list_du_alt)




du_list,idu = np.unique(list_du_id,return_index=True)  # On récupère des valeur np.unique de la liste_du_id ainsi que l'index de l'événement avec le nom idu 
du_MD_list = np.unique(list_du_id_MD)
du_long = np.unique(list_du_long)
du_lat = np.unique(list_du_lat)
du_alt = np.unique(list_du_alt)

# print(list_du_id[idu])
# print(list_du_long[idu])
# print(list_du_lat[idu])
# print(list_du_alt[idu])






# ##### Visualisation #######


daq   =   Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)   # lat, lon of the center station (from FEB@rocket) --> point rouge 
dus_feb = Geodetic(latitude=list_du_lat[idu], longitude=list_du_long[idu], height=list_du_alt[idu])  # On rentre les coord GPS des différentes antennnes activées
dus_feb = GRANDCS(dus_feb, obstime="2024-09-15", location=daq)  
plt.plot(-dus_feb.y,dus_feb.x,'ob')  # On trace tous les cercles bleus 
plt.plot(0,0,'or')   # On trace un cercle rouge à l'origne 0,0
for i in range(len(idu)):
    plt.text(-dus_feb.y[i]+100,dus_feb.x[i],list_du_id[idu][i],fontsize=12)     # On affiche à chaque point bleu l'idententité de l'antenne
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()


# Calculer le nombre de déclenchements pour chaque DU_ID
du_ids, counts = np.unique(list_du_id, return_counts=True)
total_antennes = len(du_ids)
total_declenchements = len(list_du_id)

# Création de l'histogramme horizontal
plt.figure(figsize=(10, 6))
y_positions = np.arange(len(du_ids))
plt.barh(y_positions, counts, color='blue')

plt.xlabel("Nombre de déclenchements")
plt.ylabel("Detector unit ")
plt.title(f"Histogramme des déclenchements par DU\n"
          f"Total antennes déclenchées : {total_antennes} | Total déclenchements : {total_declenchements}", 
          fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)


# On masque les ticks de l'axe y pour ne pas afficher les id par défaut
plt.yticks([])

# Pour chaque barre, on ajoute le DU_ID à la fin
for i, (du, count) in enumerate(zip(du_ids, counts)):
    plt.text(count + 0.1, i, str(du), va='center', fontsize=12)

plt.tight_layout()
plt.show()







# Affichage de l'histogramme de la multiplicité (nombre d'antennes déclenchées par coincidence)
plt.figure(figsize=(10, 6))
# On définit des bins pour des valeurs entières
bins = np.arange(0.5, np.max(list_mult)+1.5, 1)
n_events, _, _ = plt.hist(list_mult, bins=bins, color='lightgreen', edgecolor='black')
plt.xlabel("Multiplicité (nombre d'antennes déclenchées)", fontsize=14)
plt.ylabel("Indice de coincidence", fontsize=14)
plt.title("Histogramme de la multiplicité des coincidences", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Marquage visuel pour les coincidences dont la multiplicité est inférieure à 4
# Ici, on surligne les bins pour 1, 2 et 3
for mult in range(1, 4):
    count = np.sum(list_mult == mult)
    if count > 0:
        # On place un texte au-dessus de la barre correspondante
        plt.text(mult, count + 1, f"Anormal: {count}", ha='center', color='red', fontsize=12)
# Optionnel : une ligne verticale indiquant la multiplicité minimale attendue (ici 4)
plt.axvline(4, color='red', linestyle='--', label="Multiplicité minimale attendue : 4")
plt.legend()
plt.tight_layout()
plt.show()





# On suppose que list_mult est déjà rempli et converti en array, contenant la multiplicité pour chaque coincidence
i_events = np.arange(len(list_mult))  # Indice de chaque coincidence

# Générer la couleur pour chaque événement : rouge si multiplicité < 4, sinon bleu
bar_colors = ['red' if m < 4 else 'blue' for m in list_mult]

plt.figure(figsize=(10, 6))
plt.barh(i_events, list_mult, color=bar_colors)
plt.xlabel("Multiplicité (nombre d'antennes déclenchées)", fontsize=14)
plt.ylabel("Indice de coincidence", fontsize=14)
plt.title("Multiplicité par coincidence", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()






# Calcul de la moyenne et de l'écart type
mean_long = np.mean(list_du_long)
std_long = np.std(list_du_long)
mean_lat  = np.mean(list_du_lat)
std_lat  = np.std(list_du_lat)

print("Longitude : moyenne =", mean_long, ", écart type =", std_long)
print("Latitude  : moyenne =", mean_lat,  ", écart type =", std_lat)


r_exp = 0.5 # Hz  ----> Taux de déclenchement attendu ? r_exp = 100 Hz originellement

list_true_time = list_trigger_time*1e9+list_trigger_nano # donne le temps de chaque premier déclenchement de chaque coincidence en ns
dur = (np.max(list_true_time)-np.min(list_true_time))/1e9 # donne la véritable durée en s du run

print("UD data on DUs:",du_list)
print("UD data duration (s) =",dur)

# plt.figure()


for target_du in du_list:
    dumask = list_du_id == target_du   # masque booléen qui sélectionne uniquement les entrées de list_du_id qui se sont déclanchées.
                                       # Remplit list_du_id avec des False partout sauf au endroits ou target == list_du_id
    t_du = (list_true_time[dumask]-np.min(list_true_time))/1e9  # temps pour 1 DU - min du temps pour 1 DU ( dans le run ) / 1e9 pour passer en s
                                                                # liste de l'espacement relatif des différents déclenchements d'une DU
    dur_du = t_du[-1]-t_du[0] # Durée d'activation de la DU lors d'un run
    # print(target_du,":",np.sum(dumask),'events in',dur_du,"s")
    r_true = np.sum(dumask)/dur # Nombre de déclanchement de la DU / durée d'activation de la DU
                                # Caractérise l'éfficacité des DU
    # print("Rate (Hz) =",r_true, r_true/r_exp)
    # plt.plot(list_trigger_time[dumask],label=str(target_du)) Qu'est ce que ca affiche ?
# plt.legend(loc='best')
# plt.show()




for target_du in du_list:
    dumask = list_du_id == target_du
    t_du = (list_true_time[dumask]-np.min(list_true_time))/1e9
    #plt.semilogy(t_du,np.arange(sum(dumask)),label='DU'+str(target_du))
    plt.plot(t_du,np.arange(np.sum(dumask)),label='DU'+str(target_du)) # On regarde les temps relatifs d'activation de chaque DU en fonction du nombre de déclenchement
plt.xlabel('Event time (s)')
plt.ylabel('Event #')

# a = np.arange(0.1,(np.max(list_true_time)-np.min(list_true_time))/1e9) # Départ : 0.1 à temps de fin de run (s)
plt.plot([0,(np.max(list_true_time)-np.min(list_true_time))/1e9],[0,r_exp*(np.max(list_trigger_time)-np.min(list_trigger_time))],'--',label='expected')
# plt.semilogy(a,10*a,'--',label='expected')
plt.legend(loc='best')    
plt.show()  


### Visiblement il y a une erreur dans l'évaluation de r_exp ici, ou dnas les données de déclenchement





target_du = 1029
evtid = 1

dumask = list_du_id == target_du    # masque booléen qui sélectionne uniquement les entrées de list_du_id correspondant à la DU 1029.

 
plt.figure()
for ich in range(1,3): # 1 et 2 
  lab = "Ch"+str(ich)
  plt.plot(list_traces[dumask,ich,:][evtid],label=lab,linewidth=1)   # list_traces[event impliquant 1029, On regarde les deux canaux x et y, On regarde toutes données][On selectionne un événement particulier]
  #plt.xlim(200,400)
plt.legend(loc='best')
tit = "DU"+str(target_du)+" - Event "+str(evtid)
plt.title(tit)


## Analyse fréquetielle ####
plt.figure()
afft = abs(rfft(list_traces[dumask,ich,:][evtid])) # ich = 2
freq = rfftfreq(len(list_traces[dumask,ich,:][evtid]),2e-3)  # in MHz with 2ns ??? seperation given in mus , rfftfreq retourne un tableau des fréquences correspondant aux composantes de la FFT.
plt.axvline(118.9,ls='--')

plt.semilogy(freq,afft,linewidth = 1)  # trace le spectre d'amplitude en utilisant une échelle logarithmique pour l'axe des ordonnées. Cela aide à visualiser des variations sur plusieurs ordres de grandeur.
plt.xlabel("Frequency (MHz)")
plt.ylabel("abs(FFT)")
plt.show()





### Analyse fréquentielle sur les données brutes ####
target_du = 1029


dumask = list_du_id == target_du
nevts = np.sum(list_du_id == target_du)
# 
npts = len(list_traces[dumask,0,:][0]) # On regarde le nombre de point de raw data dans le premier event de 1029  (1024 je crois)
mafft = np.zeros((4,int(npts/2)+1))
#for ev in range(nevts):
#    if int(ev/100) == ev/100:
#        print(ev,nevts)
    #for ich in range(4):
#    afft = abs(rfft(list_traces[dumask,ich,:][ev]))
#    mafft[ich] += afft
#mafft /= nevts
freq = rfftfreq(npts,2e-3)  # in MHz with 2ns ??? seperation given in mus
mfft = np.mean(abs(rfft(list_traces[dumask],axis=2)),axis=0)





### Visualisation du spectre #### 

def visualize_mean_fft(list_traces, list_du_id, dt=2e-3, xlim=(0,250)):
    """
    Visualizes the mean FFT spectrum for each detection unit (DU).

    Parameters:
    - list_traces: np.array of shape (n_events, n_channels, n_points)
      containing the trace data for each event.
    - list_du_id: np.array of shape (n_events,)
      containing the detection unit ID for each event.
    - dt: float, time interval between data points (default: 2e-3 seconds).
    - xlim: tuple, x-axis limits for the frequency (default: (0,250) MHz).
    """
    # Find all unique detection units (DUs)
    unique_dus = np.unique(list_du_id)
    
    plt.figure(figsize=(10, 6))
    
    # Loop over each unique DU
    for du in unique_dus:
        # Create a mask to select events for the current DU
        mask = list_du_id == du
        traces_du = list_traces[mask]  # shape: (n_events_du, n_channels, n_points)
        
        # Skip if no events were found for this DU
        if traces_du.size == 0:
            continue
        
        # Assume all events have the same number of data points per trace.
        npts = traces_du.shape[2]
        # Compute the frequency axis using the real FFT frequency function.
        freq = rfftfreq(npts, dt)
        
        # Compute FFT along the time axis for each event and take its absolute value.
        fft_values = np.abs(rfft(traces_du, axis=2))  # shape: (n_events_du, n_channels, n_freq)
        # Compute the mean FFT across events (axis=0) to get an average per channel.
        mean_fft = np.mean(fft_values, axis=0)  # shape: (n_channels, n_freq)
        
        # Plot the mean FFT for channels 1, 2, and 3.
        for ch in range(1, 4):
            plt.semilogy(freq, mean_fft[ch, :], lw=1, label=f'DU {du} - Ch{ch}')
    
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (FFT)")
    plt.xlim(xlim)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Mean FFT Spectrum for Each Detection Unit")
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming list_traces and list_du_id have been computed and converted to np.array
visualize_mean_fft(list_traces, list_du_id)


# plt.figure()
# tit = "Mean FFT - DU"+str(target_du)
# plt.title(tit)
# lab = ['FiltX','X','Y','FiltY']
# for ich in range(0,4):
#     #lab = "Ch"+str(ich)
#     plt.semilogy(freq,mfft[ich,:],lw = 1, label=lab[ich])
# plt.xlabel("Frequency (MHz)")
# plt.ylabel("abs(FFT)")
# plt.legend(loc='best')
# plt.xlim(100,150)
# plt.axvline(118.9,ls='--')
# plt.axvline(120.6,ls='--')
# plt.axvline(137.75,ls='--')
# plt.show()


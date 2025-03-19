
"""
Adaptation du code Test Nathan utilisant l'interface Analysis Oriented de GRAND.
Ce script lit le fichier ROOT via EventList, extrait les positions des antennes
des détecteurs (DU) et trace la carte (Easting/Northing) avec les DU (points bleus)
et la station DAQ (point rouge).
"""

from grand.grandlib_classes.grandlib_classes import *
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import sys
from grand import Geodetic, GRANDCS




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




def main():
    # Récupérer le nom du fichier depuis la ligne de commande ou utiliser un fichier par défaut
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "/home/nlebas/grand/examples/grandlib_classes/GP80_20250309_235256_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root"
    
    print("Reading file", file_name)
    
    # Créer l'instance EventList en précisant d'utiliser le TTree TRawVoltage pour les traces
    el = EventList(file_name, use_trawvoltage=True)
    
    for i, e in enumerate(el):
        print(f"Event {i}, du_id {e.voltages[0].du_id}, time {e.voltages[0].t0}")
        
        
        # Récupérer le vecteur temps et les traces x, y, z du Voltage du DU 0
        t = e.voltages[0].t_vector
        x = e.voltages[0].trace.x
        y = e.voltages[0].trace.y
        z = e.voltages[0].trace.z


        list_trigger_time.append(root_file.tadc.du_seconds[0])
        list_trigger_nano.append(root_file.tadc.du_nanoseconds[0])
        list_du_id.append(root_file.tadc.du_id[0])
        list_du_long.append(root_file.trawvoltage.gps_long[0])
        list_du_lat.append(root_file.trawvoltage.gps_lat[0])
        list_du_alt.append(root_file.trawvoltage.gps_alt[0])
        
        _traces = [] # A chaque ligne de data. _traces prend [raw_data,x,y,z]
        _traces.append(root_file.tadc.trace_ch[0][0])
        _traces.append(root_file.tadc.trace_ch[0][1])
        _traces.append(root_file.tadc.trace_ch[0][2])
        _traces.append(root_file.tadc.trace_ch[0][3])
        list_traces.append(_traces) # list_traces prend [[raw_data,x,y,z], [raw_data,x,y,z], [raw_data,x,y,z]]
        
        

if __name__ == '__main__':
    main()

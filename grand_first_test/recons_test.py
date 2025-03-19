import os

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


################################################################################
kc = 299792458. #m/s
kc_ns = 299792458.*1.e-9 #m/ns
kn = 1.0001
#kn = 1.000293
kcn_ns = kc_ns / kn
sig_t = 15 # Timing resolution (ns)
################################################################################


recons_path = "/home/nlebas/grand/examples/grandlib_classes/GP80_20250309_235256_RUN10070_CD_20dB_23DUs_GP43-ChY-X2X-Y2Y-CD-10000-22.root"

idant, xant, yant, zant = np.loadtxt(recons_path+"coord_antennas.txt").T
_, _, theta_rec, _, phi_rec, _, chi2_plane, _ = np.loadtxt(recons_path+"Rec_plane_wave_recons.txt").T
recoinc, nant, chi2_sphere, _, xrec, yrec, zrec, trec, rho, ground_alt = np.loadtxt(recons_path+"Rec_sphere_wave_recons.txt").T
ind, coincid, peakt, peaka = np.loadtxt(recons_path+"Rec_coinctable.txt").T
antid = np.loadtxt(recons_path+"DU_id.txt",usecols=1).T  # Modified DU_id.txt format for Xishui analysis


xdaqm, ydaqm, zdaqm = 406, 456, 1220  # DAQ position
xdaq, ydaq, zdaq = 490, 501, 1320  # DAQ position
tsphere_rec, tsphere_daq, peakt_sel, chi2_exp, chi2ndf_delays,ds_daq = [], [], [], [], [], []
ds_daq, t_datastack = np.zeros((len(nant),7)), np.zeros((len(nant),7)) # Ugly, works for May 2024 data

plt.figure()
plt.hist(nant)
plt.show()

#/!\ crappy angular convention correction
phi_rec +=180.
phi_rec = phi_rec%360.
################################################################################

# Buld distributions
dist_daq = np.linalg.norm(np.array([xrec - xdaq, yrec - ydaq, zrec - zdaq]), axis=0)
dist_daq_xy = np.linalg.norm(np.array([xrec - xdaq, yrec - ydaq]), axis=0)
sel = range(len(xrec))
print("Ncoincs:",len(nant))
for icoinc in np.unique(recoinc[sel]):

    sel_coinc = np.where(coincid==icoinc)[0]
    xant_coinc = xant[sel_coinc] ; yant_coinc = yant[sel_coinc] ; zant_coinc = zant[sel_coinc]
    peakt_coinc = peakt[sel_coinc]
    sel_rec = np.where(recoinc==icoinc)[0][0]
    trec_coinc = trec[sel_rec]
    dist_ant_rec = np.linalg.norm(np.array([xant_coinc - xrec[sel_rec], yant_coinc - yrec[sel_rec], zant_coinc - zrec[sel_rec]]), axis=0)
    dist_ant_daq = np.linalg.norm(np.array([xant_coinc - xdaq, yant_coinc - ydaq, zant_coinc - zdaq]), axis=0)
    tsphere_rec.append(dist_ant_rec/kcn_ns)
    tsphere_daq.append(dist_ant_daq/kcn_ns)
    peakt_sel.append(peakt_coinc*1.e9)

################################################################################
##Plots
#plot_folder = './GP13_plots/'
plot_folder = recons_path
if not os.path.exists(plot_folder): os.makedirs(plot_folder)


## Perform Spherical recons - DelayPlots
for i in range(len(nant)):
#for i in range(5):
    print("Coinc", i, "mult=",nant[i])
    sel_rec = np.where(coincid==i)
    t_data = peakt_sel[i]-min(peakt_sel[i])
    t_model = tsphere_rec[i]-min(tsphere_rec[i])
    t_daq = tsphere_daq[i]-min(tsphere_daq[i])
    d_model = t_model-t_data
    t_model = t_model-np.mean(d_model)
    d_model = d_model-np.mean(d_model)
    d_daq = t_daq-t_data
    t_daq = t_daq-np.mean(d_daq)
    d_daq = d_daq-np.mean(d_daq)
    chi2 = sum((d_model)**2)/sig_t**2/(len(t_data)-4)
    chi2ndf_delays.append(chi2)

    #fb, (b1x,b2x) = plt.subplots(2,1)
    fb = plt.figure()
    fb.set_figheight(8)
    fb.set_figwidth(8)
    b1x = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2)
    b2x = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1)
    fb.suptitle(f"Coinc {int(recoinc[sel][i]):d}")
    b1x.set_ylabel(r"$\rm t_{\rm model}\ (ns)$")
    b1x.errorbar(t_data, t_model,yerr = sig_t,fmt='.',color='b',label='fit')
    #b1x.errorbar(t_data, t_daq,yerr = sig_t,fmt='.',color='g',label='mean')

    if (len(d_daq) == 7) & (chi2<100):
        isort = np.argsort(antid[sel_rec])
        ds_daq[i] = d_daq[isort] # Save trig time difference to model (DU ID ascending order)
        t_datastack[i] = t_data[isort] # Save raw trig time (DU ID ascending order)

    s= 'Source reconstructed at'
    b1x.text(max(t_data)*0.05,max(t_model)*0.9,s,fontsize='x-small')
    s= 'SN = {0:0.2f} m, EW = {1:0.2f}m, al t ={2:0.2f}m asl'.format(xrec[i],yrec[i],zrec[i])
    b1x.text(max(t_data)*0.05,max(t_model)*0.85,s,fontsize='x-small')
    s = '$\chi^2$/ndf = {0:0.2f}'.format(chi2)
    b1x.text(max(t_data)*0.05,max(t_model)*0.8,s,fontsize='x-small')
    print(s)
    for j in range(len(t_data)):
         #print(int(antid[sel_rec][j]),int(t_data[j]))
         b1x.text(t_data[j]+150,t_model[j]-50,int(antid[sel_rec][j]),fontsize='x-small')
    b1x.plot([0,max(t_data)],[0,max(t_data)],'--r')
    #bx.legend()
    b2x.errorbar(t_data, d_model,yerr = sig_t,fmt='.',color='b',label='fit')
    #b2x.errorbar(t_data, d_daq,yerr = sig_t,fmt='.',color='g',label='mean')
    b2x.grid(True)
    b2x.set_xlabel(r"$\rm t_{\rm mes}\ (ns)$")
    b2x.set_ylabel(r"$\rm t_{\rm model} -t_{\rm mes}\ (ns)$")
    b2x.legend()
    fb.tight_layout()
    fb.savefig(plot_folder+"delay_coinc"+str(int(recoinc[i]))+".pdf", format='pdf') #; plt.show()
#plt.show()

## Timing resolution plots
uid = np.sort(np.unique(antid))
plt.figure()
for j in range(0,7):
    plt.hist(ds_daq[:,j],bins=np.linspace(-25,25,25),histtype='step', stacked=True, fill=True,label=int(uid[j]))
    print('DU',int(uid[j]),', mean=',np.mean(ds_daq[:,j]),'std dev=',np.std(ds_daq[:,j]))
plt.xlabel('$t_{mes}-t_{mean}$ (ns)')
plt.legend()
plt.savefig(plot_folder+"sig_timing2.pdf", format='pdf') #; plt.show()

uid = np.sort(np.unique(antid))
plt.figure()
for j in range(1,7):
    plt.subplot(3,2,j)
    plt.gca().set_title("DU"+str(int(uid[j])),fontsize='xx-small')
    sel = t_datastack[:,j]>0
    h = plt.hist(t_datastack[sel,j],histtype='step',fill=False)
    s = "mean = {0:0.2f} ns".format(np.mean(t_datastack[sel,j]))
    plt.text(np.mean(t_datastack[sel,j]),0.8*np.max(h[0]),s,fontsize='x-small')
    s = "std dev = {0:0.2f} ns".format(np.std(t_datastack[sel,j]))
    plt.text(np.mean(t_datastack[sel,j]),0.6*np.max(h[0]),s,fontsize='x-small')
    if j>4:
        plt.xlabel("$t_{mes}$ (ns)",fontsize='xx-small')
plt.tight_layout()
plt.savefig(plot_folder+"sig_timing.pdf", format='pdf')
#; plt.show()


# Reconstructed position plots
chi2ndf_delays = np.array(chi2ndf_delays)
selchi2 = np.where(chi2ndf_delays<100)
#sel6 =  np.where(nant[0:5]==6)
#sel7 =  np.where(nant[0:5]==7)
#sel = np.intersect1d(selchi2, sel6)
sel = selchi2

fa1, a1x = plt.subplots()
#fa1.suptitle(r"$\rm GP13$")
a1x.set_xlabel(r"$\rm Northing\ (m)$")
a1x.set_ylabel(r"$\rm Westing\ (m)$")
a1x.scatter(xant, yant, label='DU')
a1x.scatter(xrec[sel], yrec[sel], label='Reconstructed source')
#a1x.scatter(xdaq, ydaq, label='mean')
#a1x.scatter(xdaqm, ydaqm, label='DAQ')
a1x.set_ylim([-1000,  600])
a1x.axis('equal')
a1x.legend()
fa1.savefig(plot_folder+"source_pos_array_xy.pdf", format='pdf') #; plt.show()

fa2, a2x = plt.subplots()
#fa2.suptitle(r"$\rm GP13$")
a2x.set_xlabel(r"$\rm Northing\ (m)$")
a2x.set_ylabel("Altitude asl (m)")
a2x.scatter(xant, zant, label='DU')
a2x.scatter(xrec[sel], zrec[sel], marker='o', label='Reconstructed source')
#a2x.scatter(xdaq, zdaq, label='mean')
#a2x.scatter(xdaqm, zdaqm, label='DAQ')
a2x.axis('equal')
a2x.legend()
fa2.savefig(plot_folder+"source_pos_array_xz.pdf", format='pdf') #; plt.show()


## All delays
fc, cx = plt.subplots()
#fc.suptitle(r"$\rm GP13$")
cx.set_xlabel(r"$\rm t_{\rm mes}\ (ns)$")
cx.set_ylabel(r"$\rm t_{\rm model}\ (ns)$")
cx.scatter(t_data,t_model,label='rec')
#[cx.scatter(peakt_sel[sel[i]]-min(peakt_sel[sel[i]]), tsphere_rec[sel[i]]-min(tsphere_rec[sel[i]]), label='rec') for i in range(len(sel))]
#[cx.scatter(peakt_sel[i]-min(peakt_sel[i]), tspehere_daq[i]-min(tspehere_daq[i]), marker='x', label='daq') for i in range(len(tsphere_rec))]
#plt.show()
fc.savefig(plot_folder+"delay_all.pdf", format='pdf')

## Histos theta' phi plan
fd, (d1x, d2x) = plt.subplots(1, 2)
#fd.suptitle(r"$\rm GP13$")
d1x.set_xlabel(r"$\rm \theta_{\rm rec}\ (deg)$")
d2x.set_xlabel(r"$\rm \phi_{\rm rec}\ (deg)$")
d1x.hist(theta_rec[nant<7], bins=int(2.*np.sqrt(len(theta_rec))), label='6 DU')
d1x.hist(theta_rec[nant>6], bins=int(2.*np.sqrt(len(theta_rec))), label='7,8 DU')
d1x.legend()
d2x.hist(phi_rec[(phi_rec<100)&(nant<7)], bins=int(2.*np.sqrt(len(phi_rec))), label='6 DU')
d2x.hist(phi_rec[(phi_rec<100)&(nant>6)], bins=int(2.*np.sqrt(len(phi_rec))), label='7,8 DU')
d2x.legend()
fd.savefig(plot_folder+"histo_theta_phi.pdf", format='pdf')

## Histos position
fe, (e1x, e2x, e3x) = plt.subplots(1, 3)
#fe.suptitle(r"$\rm GP13$")
e1x.set_xlabel(r"$\rm SN_{\rm source, rec}\ (m)$")
e2x.set_xlabel(r"$\rm EW_{\rm source, rec}\ (m)$")
e3x.set_xlabel(r"$\rm z_{\rm source, rec}\ (m)$")
e1x.hist(xrec[(nant<17)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(xrec))), label='6 DU', zorder=2)
#e1x.hist(xrec[(nant>6)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(xrec))), label='7,8 DU')
print("Mean reconstructed source position")
print('X:, mean=',np.mean(xrec[sel]),'m, std dev =',np.std(xrec[sel]))
e2x.hist(yrec[(nant<17)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(yrec))), label='6 DU', zorder=2)
#e2x.hist(yrec[(nant>6)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(yrec))), label='7,8 DU')
print('Y:, mean=',np.mean(yrec[sel]),'m, std dev =',np.std(yrec[sel]))
e3x.hist(zrec[(nant<17)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(zrec))), label='6 DU', zorder=2)
e3x.hist(zrec[(nant>6)&(xrec<1.e3)], bins=int(1.5*np.sqrt(len(zrec))), label='7,8 DU')
e3x.legend()
fe.savefig(plot_folder+"histo_x_y_z.pdf", format='pdf')

## Chi2
ff, (f1x, f2x) = plt.subplots(1, 2)
#ff.suptitle(r"$\rm GP13$")
f1x.set_xlabel(r"$\rm \frac{\chi^2_{\rm plane}}{ N_{\rm ant}-2}$")
f2x.set_xlabel(r"$\rm \frac{\chi^2_{\rm sphere}}{ N_{\rm ant}-4}$")

sigma_t = (15*1.e-9*kc)**2
# f1x.hist(chi2_plane[chi2_plane<1e8]/(nant[chi2_plane<1e8]-2)/sigma_t, bins=int(2*np.sqrt(len(chi2_plane))))
# f2x.hist(chi2_sphere[chi2_sphere<1e8]/(nant[chi2_plane<1e8]-4)/sigma_t, bins=int(2*np.sqrt(len(chi2_sphere))))
f1x.hist(chi2_plane[chi2_plane<1e8]/sigma_t, bins=int(2*np.sqrt(len(chi2_plane))))
f2x.hist(chi2_sphere[chi2_sphere<1e8]/sigma_t, bins=int(2*np.sqrt(len(chi2_sphere))))
ff.savefig(plot_folder+"histo_chi2.pdf", format='pdf') #; plt.show()


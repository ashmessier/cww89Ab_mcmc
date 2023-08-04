#from complete_fitfunctions import *
from astropy.stats import sigma_clip
import numpy as np

# PARAMETERS AND PRIORS ------------------------------------------

Nowak = True
Beatty = True
Spitzer = True
combo = True

#lastrun = np.load("final_params_working.npz")
#pars_lastrun = lastrun["pars"][:-3]
pars_lastrun = [ 2.45734633e+06,  7.23671448e-01 , 8.96680539e-02 , 9.08712237e-02,
  9.32840342e-02 , 1.10105843e+00 , 3.01517908e-02 ,-4.69965731e-02,
  1.87089610e-01 ,-3.25863522e-03  ,2.23321109e-03] # best pars from that one run

# From RV fitting
Tc_RV = [2457341.0487, 0.013]
period_RV = [5.29236, 0.00038]
log10per_RV = np.log10(period_RV[0])
sig_log10per_RV = 1 / (np.log(10)) * period_RV[1] / period_RV[0]

# from TB paper
sqrtecosw = [0.4213, 0.0011]
sqrtesinw = [-0.1061, 0.0006]

e = (sqrtesinw[0])**2 + (sqrtecosw[0])**2
sig_e = np.sqrt((2 * sqrtesinw[1] * sqrtesinw[0])**2 + (2 * sqrtecosw[1] * sqrtecosw[0])**2)

w = np.arctan2(sqrtesinw[0], sqrtecosw[0])
# w error calculation
partial_w_partial_x = sqrtecosw[0] / (sqrtesinw[0] ** 2 + sqrtecosw[0] ** 2)
partial_w_partial_y = -sqrtesinw[0] / (sqrtesinw[0] ** 2 + sqrtecosw[0] ** 2)
sig_w = np.sqrt((partial_w_partial_x * sqrtesinw[1]) ** 2 + (partial_w_partial_y * sqrtecosw[1]) ** 2)

esinw_TB = e * np.sin(w)
ecosw_TB = e * np.cos(w)

# esinw_TB error calculation
partial_m_partial_x = np.sin(w)
partial_m_partial_y = e * np.cos(w)
sig_esinw_TB = np.sqrt((partial_m_partial_x * sig_e) ** 2 + (partial_m_partial_y * sig_w) ** 2)

# ecosw_TB error calculation
partial_m_partial_x = np.cos(w)
partial_m_partial_y = -e * np.sin(w)
sig_ecosw_TB = np.sqrt((partial_m_partial_x * sig_e) ** 2 + (partial_m_partial_y * sig_w) ** 2)

# Nowak kepler fit
if Nowak:
    # # priors from Nowak
    priors = np.zeros((8, 2))
    priors[0] = [2457346.32942, 0.00011]  # T0
    priors[1] = [5.29236, 0.00038]
    priors[2] = [0.09321, 0.00046]  # Rp/R*
    priors[3] = [12.62, 0.15]  # a/R*
    priors[4] = [88.53, 0.7]  # i
    priors[5] = [0.1929, 0.0019]  # e
    priors[6] = [345.9, 1.0]  # w
    priors[7] = [0, 0.0001]

    # modified parameters
    ecosw = priors[5][0] * np.cos(priors[6][0] * np.pi / 180)  # e cos w
    esinw = priors[5][0] * np.sin(priors[6][0] * np.pi / 180)  # e sin w
    cosi = np.cos(priors[4][0] * np.pi / 180)
    log10per = np.log10(priors[1][0])
    log10ars = np.log10(priors[3][0])
    # errors
    sig_esinw = np.sqrt((np.sin(priors[6][0] * np.pi / 180)) ** 2 * (priors[5][1] * np.pi / 180) ** 2 + (
                priors[5][0] * np.pi / 180) ** 2 * (np.cos(priors[6][0] * np.pi / 180)) ** 2 * (
                                    priors[6][1] * np.pi / 180) ** 2)
    sig_ecosw = np.sqrt((np.cos(priors[6][0] * np.pi / 180)) ** 2 * (priors[5][1] * np.pi / 180) ** 2 + (
                priors[5][0] * np.pi / 180) ** 2 * (np.sin(priors[6][0] * np.pi / 180)) ** 2 * (
                                    priors[6][1] * np.pi / 180) ** 2)
    # sig_cosi = np.sqrt((np.sin(priors[4][0] * np.pi/180))**2 * (priors[4][1] * np.pi/180)**2)
    sig_cosi = np.sin(cosi) * priors[4][1]
    sig_log10per = 1 / (np.log(10)) * priors[1][1] / priors[1][0]
    sig_log10ars = 1 / (np.log(10)) * priors[3][1] / priors[3][0]

    # import priors from Nowak et al 2018, modified
    # T0, orbital period, planet radius, semimajor axis, orbital inclination, eccentricity, longitude of periastrum, slope
    priors_mod = np.zeros((11, 2))
    priors_mod[0] = [2457346.32942, 0.00011]  # T0
    priors_mod[1] = [log10per, sig_log10per]  # orbital period
    priors_mod[2] = [0.09321, 0.00046]  # Rp/R* ch 1
    priors_mod[3] = [0.09321, 0.00046]  # Rp/R* ch 2
    priors_mod[4] = [0.09321, 0.00046]  # Rp/R* KEPLER
    priors_mod[5] = [log10ars, sig_log10per]  # a/R*
    priors_mod[6] = [cosi, sig_cosi/10]  # i
    priors_mod[7] = [esinw, sig_esinw]  # e
    priors_mod[8] = [ecosw, sig_ecosw]  # w
    priors_mod[9] = [0.0, 0.0001]  # m ch 1
    priors_mod[10] = [0.0, 0.0001]  # m ch 2


# Beatty kepler fit
if Beatty:
    priors_TB = np.zeros((11, 2))
    priors_TB[0] = [2457341.0487, 0.013]  # t0
    priors_TB[1] = [0.7236684570630421, 5.856336103837734e-07]  # period
    priors_TB[2] = [0.0978060591950399, 0.00016162981830498968]  # kepler RpRs
    priors_TB[3] = [0.0978060591950399, 0.00016162981830498968]  # kepler RpRs
    priors_TB[4] = [0.0978060591950399, 0.00016162981830498968]  # rprs kepler
    priors_TB[5] = [1.01309399020987, 0.0020814793363661543]  # ars
    priors_TB[6] = [0.06260982006391358, 0.0007577994519925302]  # cosi
    priors_TB[7] = [-0.04879303048476328, 0.0011160372572116273]  # sin
    priors_TB[8] = [0.18647582765885531, 0.001071563655824781]  # cos
    priors_TB[9] = [0.01, 0.01]  # slope 1
    priors_TB[10] = [0.01, 0.01]  # slope 2

# bestfit spitzer parameters from dual fitting
if Spitzer:
    priors_spitz = np.zeros((11, 2))
    priors_spitz[0] = [2.45734633e+06, 1.12186652e-04]  # t0
    priors_spitz[1] = [7.23672336e-01, 1.31643107e-07]  # period
    priors_spitz[2] = [8.95726966e-02, 1.98664231e-03]  # Rprs 1
    priors_spitz[3] = [9.09581488e-02, 1.43929431e-03]  # RpRs 2
    priors_spitz[4] = [9.09581488e-02, 1.43929431e-03]  # RpRs kepler (2)
    priors_spitz[5] = [1.10105914e+00, 2.08192079e-06]  # ars
    priors_spitz[6] = [2.81958975e-02, 1.24345437e-03]  # cosi
    priors_spitz[7] = [-4.69970010e-02, 5.74542193e-05]  # sin
    priors_spitz[8] = [1.87084060e-01, 3.57706784e-05]  # cos
    priors_spitz[9] = [-3.15100977e-03, 5.08412992e-03]  # slope 1
    priors_spitz[10] = [2.26508542e-03, 2.37258645e-03]  # slope 2

    pars_spitz = priors_spitz[:, 0]

if combo:# MCMC WORKS WITH COMBO PRIORS as PARS AND USING PRIORS_SPITZ AS T0 AND PERIOD
    combo_priors = np.zeros((11, 2))
    combo_priors[0] = Tc_RV # t0 # used to be RV T0 # WORKS WITH PRIORS_SPITZ
    combo_priors[1] = [log10per_RV, sig_log10per_RV]   # WORKS WITH PRIORS_SPITZ
    combo_priors[2] = priors_spitz[2]  # RpRs 1
    combo_priors[3] = priors_spitz[3]  # RpRs 2
    combo_priors[4] = [0.0938397497949011, priors_TB[4][1]] # used to both be priors_TB
    combo_priors[5] = priors_mod[5]  # Ars
    combo_priors[6] = priors_spitz[6]  # Cosi
    combo_priors[7] = [esinw_TB, sig_esinw_TB]
    combo_priors[8] = [ecosw_TB, sig_ecosw_TB]
    combo_priors[9] = priors_spitz[9]  # m1
    combo_priors[10] = priors_spitz[10]  # m2

    combo_pars = np.zeros((11, 1))
    combo_pars[0] = priors_spitz[0][0] # t0 # used to be RV T0 # WORKS WITH PRIORS_SPITZ
    combo_pars[1] = priors_spitz[1][0]  # WORKS WITH PRIORS_SPITZ
    combo_pars[2] = priors_spitz[2][0]  # RpRs 1
    combo_pars[3] = priors_spitz[3][0]  # RpRs 2
    combo_pars[4] = [0.0938397497949011] # used to both be priors_TB
    combo_pars[5] = priors_mod[5][0]  # Ars
    combo_pars[6] = priors_spitz[6] [0] # Cosi
    combo_pars[7] = priors_mod[7][0]  # esinw
    combo_pars[8] = priors_mod[8][0]  # ecosw
    combo_pars[9] = priors_spitz[9][0]  # m1
    combo_pars[10] = priors_spitz[10][0]  # m2


# These are the values that wokr individually
sd_combo_priors = np.zeros((10, 2))
sd_combo_priors[0] = Tc_RV # t0
sd_combo_priors[1] = priors_TB[1]  # period
sd_combo_priors[2] = priors_spitz[2]  # RpRs 1
sd_combo_priors[3] = priors_spitz[3]  # RpRs 2
sd_combo_priors[4] = priors_mod[5]  # Ars
sd_combo_priors[5] = priors_spitz[6]  # Cosi
sd_combo_priors[6] = priors_mod[7]  # esinw
sd_combo_priors[7] = priors_mod[8]  # ecosw
sd_combo_priors[8] = priors_spitz[9]  # m1
sd_combo_priors[9] = priors_spitz[10]  # m2

# Kepler
kd_combo_priors = np.zeros((7, 2))
kd_combo_priors[0] = priors_spitz[0]
kd_combo_priors[1] = priors_spitz[1]  # period
kd_combo_priors[2] = [0.0938397497949011, 0.001]
kd_combo_priors[3] = priors_mod[5]  # Ars
kd_combo_priors[4] = priors_spitz[6]  # Cosi
kd_combo_priors[5] = priors_mod[7]  # esinw
kd_combo_priors[6] = priors_mod[8]  # ecosw
#kd_combo_priors[7] = [0.323, 0.01]  # u1
#kd_combo_priors[8] = [0.232, 0.01]  # u2




# CH1
ch1 = {}
idlsav1 = np.load('./Data/ch1_straight_test.npz')
time1 = idlsav1['time'][:1860]
flux1 = idlsav1['flux'][0,:,10][:1860] # aperture size
error1 = idlsav1['error'][0,:,10][:1860]
xpos1 = idlsav1['xpos'][:1860]
ypos1 = idlsav1['ypos'][:1860]
bkg1 = idlsav1['bkg'][:1860]

# CH 2
ch2 = {}
idlsav2 = np.load('./Data/ch2_straight_test.npz')
time2 = idlsav2['time']
flux2 = idlsav2['flux'][0,:,11] # aperture size
error2 = idlsav2['error'][0,:,11]
xpos2 = idlsav2['xpos']
ypos2 = idlsav2['ypos']
bkg2 = idlsav2['bkg']

#KEPLER
kepler = {}
data = np.loadtxt("./Data/cww89a.Kepler.Kepler.txt")
kepler["time"] = data[:, 0]
kepler["flux"] = data[:, 1]
kepler["error"] = data[:, 2]

#CH 1
#removes outliers >3 sigma
masked_arr1 = sigma_clip(flux1, 3, masked=True) # all values greater than 3 sigma
mask_sigmaclip1 = masked_arr1.mask # boolean array to apply to other variables

#CH 2
#removes outliers >3 sigma
masked_arr2 = sigma_clip(flux2, 3, masked=True) # all values greater than 3 sigma
mask_sigmaclip2 = masked_arr2.mask #

mask_sigmaclip = np.ma.mask_or(mask_sigmaclip1, mask_sigmaclip2)

# final values with both masks applied
ch1["flux_raw"] = flux1[~mask_sigmaclip]
ch1["time"] = time1[~mask_sigmaclip]
ch1["error_raw"] = error1[~mask_sigmaclip]
ch1["xpos"] = xpos1[~mask_sigmaclip]
ch1["ypos"] = ypos1[~mask_sigmaclip]

ch2["flux_raw"] = flux2[~mask_sigmaclip]
ch2["time"] = time2[~mask_sigmaclip]
ch2["error_raw"] = error2[~mask_sigmaclip]
ch2["xpos"] = xpos2[~mask_sigmaclip]
ch2["ypos"] = ypos2[~mask_sigmaclip]

#CH 1
i1 = 400 # index 1
i2 = 1610 # index 2
base11 = ch1["flux_raw"][:i1] # defines baseline from index
base21 = ch1["flux_raw"][i2:]
baseline1 = np.concatenate((base11, base21)) # baseline flux
norm_value1 = np.mean(baseline1) # average baseline value

ch1["flux"] = ch1["flux_raw"] / norm_value1 # normalizes flux
ch1["error"] = ch1["error_raw"] / norm_value1 # normalizes error

# CH 2
i1 = 687 # index 1
i2 = 1610 # index 2
base12 = ch2["flux_raw"][:i1] # defines baseline from index
base22 = ch2["flux_raw"][i2:]
baseline2 = np.concatenate((base12, base22)) # baseline flux
norm_value2 = np.mean(baseline2) # average baseline value

ch2["flux"] = ch2["flux_raw"] / norm_value2 # normalizes flux
ch2["error"] = ch2["error_raw"] / norm_value2 # normalizes error

# CH 1
# code from Beatty to initalize map for pixel mapping
stepx = 0.01
stepy = 0.01
BLISSparams36 = {}
BLISSparams36['xmin'] = np.amin(ch1["xpos"]) - 5. * stepx
BLISSparams36['xmax'] = np.amax(ch1["xpos"]) + 5. * stepx
BLISSparams36['ymin'] = np.amin(ch1["ypos"]) - 5. * stepy
BLISSparams36['ymax'] = np.amax(ch1["ypos"]) + 5. * stepy
BLISSparams36['knotsx'] = np.arange(BLISSparams36['xmin'], BLISSparams36['xmax'], stepx)
BLISSparams36['knotsy'] = np.arange(BLISSparams36['ymin'], BLISSparams36['ymax'], stepy)
BLISSparams36['nx'] = BLISSparams36['knotsx'].shape[0]
BLISSparams36['ny'] = BLISSparams36['knotsy'].shape[0]
BLISSparams36['knotsgrid'] = np.empty((BLISSparams36['nx'],BLISSparams36['ny']), dtype=object)
BLISSparams36['grid_x_y'] = np.mgrid[BLISSparams36['xmin']:BLISSparams36['xmax']:stepx, BLISSparams36['ymin']:BLISSparams36['ymax']:stepy]
for i in range(ch1["xpos"].shape[0]):
    nearx = np.searchsorted(BLISSparams36['knotsx'], ch1["xpos"][i])
    neary = np.searchsorted(BLISSparams36['knotsy'], ch1["ypos"][i])
    x1 = BLISSparams36['knotsx'][nearx - 1]
    x2 = BLISSparams36['knotsx'][nearx]
    y1 = BLISSparams36['knotsy'][neary - 1]
    y2 = BLISSparams36['knotsy'][neary]
    if (x2 - ch1["xpos"][i]) <= (ch1["xpos"][i] - x1): nearestx = nearx
    else: nearestx = nearx - 1
    if (y2 - ch1["ypos"][i]) <= (ch1["ypos"][i] - y1): nearesty = neary
    else: nearesty = neary -1
    BLISSparams36['knotsgrid'][int(nearestx),int(nearesty)] = np.append(BLISSparams36['knotsgrid'][int(nearestx),int(nearesty)], i)

#CH 2
# code from Beatty to initalize map for pixel mapping
stepx = 0.01
stepy = 0.01
BLISSparams45 = {}
BLISSparams45['xmin'] = np.amin(ch2["xpos"]) - 5. * stepx
BLISSparams45['xmax'] = np.amax(ch2["xpos"]) + 5. * stepx
BLISSparams45['ymin'] = np.amin(ch2["ypos"]) - 5. * stepy
BLISSparams45['ymax'] = np.amax(ch2["ypos"]) + 5. * stepy
BLISSparams45['knotsx'] = np.arange(BLISSparams45['xmin'], BLISSparams45['xmax'], stepx)
BLISSparams45['knotsy'] = np.arange(BLISSparams45['ymin'], BLISSparams45['ymax'], stepy)
BLISSparams45['nx'] = BLISSparams45['knotsx'].shape[0]
BLISSparams45['ny'] = BLISSparams45['knotsy'].shape[0]
BLISSparams45['knotsgrid'] = np.empty((BLISSparams45['nx'],BLISSparams45['ny']), dtype=object)
BLISSparams45['grid_x_y'] = np.mgrid[BLISSparams45['xmin']:BLISSparams45['xmax']:stepx, BLISSparams45['ymin']:BLISSparams45['ymax']:stepy]
for i in range(ch2["xpos"].shape[0]):
    nearx = np.searchsorted(BLISSparams45['knotsx'], ch2["xpos"][i])
    neary = np.searchsorted(BLISSparams45['knotsy'], ch2["ypos"][i])
    x1 = BLISSparams45['knotsx'][nearx - 1]
    x2 = BLISSparams45['knotsx'][nearx]
    y1 = BLISSparams45['knotsy'][neary - 1]
    y2 = BLISSparams45['knotsy'][neary]
    if (x2 - ch2["xpos"][i]) <= (ch2["xpos"][i] - x1): nearestx = nearx
    else: nearestx = nearx - 1
    if (y2 - ch2["ypos"][i]) <= (ch2["ypos"][i] - y1): nearesty = neary
    else: nearesty = neary -1
    BLISSparams45['knotsgrid'][int(nearestx),int(nearesty)] = np.append(BLISSparams45['knotsgrid'][int(nearestx),int(nearesty)], i)

ch1["BLISSparams"] = BLISSparams36
ch2["BLISSparams"] = BLISSparams45
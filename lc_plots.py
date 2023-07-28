import matplotlib.pyplot as plt
from complete_fitfunctions import *
import numpy as np
import corner
from scipy.stats import binned_statistic
import sys
from matplotlib import rc # for global times new roman font
from pathlib import Path
rc("font",**{"family":"serif", "serif":["Times"]})
rc("text", usetex=True)

# PLOT OPTIONS ----------------------------------------------------------
print_pars = True
plot_corner = False
plot_spitzer = False
combo_spitzer = False
plot_kepler= False
plot_residuals = False
plot_bliss = False
plot_depths = True
plot_all_lc2 = True
plot_sequence = False

savefigs = True

fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

lc_col = "crimson"
point_col = "mediumaquamarine"
bliss_cmap = "RdYlBu"
lam_colors = ["darkorange", "forestgreen", "midnightblue"]

# LOADS DATA ---------------------------------------------
flat_sample = np.load("flat_sample_1.npy")

final_params = np.load("final_params_working.npz")

just_params = final_params["pars"]
just_errs = final_params["errors"]

d_ch1 = np.load("ch1_data.npz")
d_ch2 = np.load("ch2_data.npz")
d_kepler = np.load("kepler_data.npz")

# DICTIONARY VALUES --------------------------------------------------
keys = ["time", "flux", "err", "mcmc_lc", "residuals", "x", "y", "blissflux"]
ch1 = {}
ch2={}
kepler={}

raw_data = np.load("additional_plotstuff.npz")
ch1["raw_flux"] = raw_data["normflux1"]
ch2["raw_flux"] = raw_data["normflux2"]

for key in keys:
    ch1[key] = d_ch1[key]
    ch2[key] = d_ch2[key]
for key in keys[:5]:
    kepler[key] = d_kepler[key]

ch1["BJD_decimal"] = ch1["time"] - np.floor(ch1["time"][0])
ch2["BJD_decimal"] = ch2["time"] - np.floor(ch2["time"][0])
kepler["BJD_decimal"] = kepler["time"] - np.floor(kepler["time"][0])


print(np.std(kepler["flux"][:100]) * 1e6)
sum = 0
for point in kepler["flux"]:
    if point < 0.992:
        sum += 1

print(186/np.sqrt(sum))


# BINNING DATA ----------------------------------------------------------
bin_data(ch1, just_params, ch=1, dict_append=True)
bin_data(ch2, just_params, ch=2, dict_append=True)

# PHASEFOLDING
PhaseFold(kepler, just_params, ch="kepler", SuperSample=True, residuals=True)
PhaseFold(ch1, just_params, binned=True, ch=1)
PhaseFold(ch2, just_params, binned=True, ch=2)
ch1_nonbin_ph = PhaseFold(ch1, just_params, binned=False, new_dict=True, ch=1)
ch2_nonbin_ph = PhaseFold(ch2, just_params, binned=False, new_dict=True, ch=2)

if print_pars:
    labels = ["T0", "log_period","RpRs_ch1", "RpRs_ch2", "RpRsK", "log_ars", "cosi",
              "esinw", "ecosw","slope_ch1", "slope_ch1", "depth_ch1","depth_ch2", "depthk"]

    for i, label in enumerate(labels):
        print(labels[i], ":", just_params[i], "+/-", just_errs[i])

# plots cornerplot from MCMC - 1000 burnin, 2000 nprods
if plot_corner:
    labels = ["T0", "Log10(P)", "Rp/R*1", "Rp/R*2", "RpRsK", "Log10(a/R*)", "cosi", "esinw", "ecosw", "slope1", "slope2"]
    fig = corner.corner(flat_sample, labels=labels, show_titles=True)
    plt.tight_layout()
    if savefigs:
        fig_path = fig_dir / "cornerplot.png"
        plt.savefig(fig_path)
    else:
        plt.show()

# plot spitzer CH 1, CH 2 lightcurves
if plot_spitzer:
    plt.figure(figsize=(7, 5), dpi=750)
    plt.scatter(ch1["BJD_decimal"], ch1["flux"], s=1, color="gray", alpha=0.3)
    plt.plot(ch1["binned_BJD"], ch1["binned_lc"], color=lc_col, zorder = 3)
    plt.errorbar(ch1["binned_BJD"], ch1["binned_flux"], yerr=ch1["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch1["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity")
    plt.ylim(1-0.017, 1+0.013)
    plt.title(r"3.6 $\mu$m Transit")
    if savefigs:
        fig_path = fig_dir / "spitzer_ch1_lc.png"
        plt.savefig(fig_path)
    else:
        plt.show()

if combo_spitzer:
    # plot spitzer CH 2
    plt.figure(figsize=(7*2, 6), dpi=750)
    plt.subplot(131)
    plt.scatter(ch2["BJD_decimal"], ch2["flux"], s=1, color="gray", alpha=0.3)
    plt.plot(ch2["binned_BJD"], ch2["binned_lc"], color=lc_col, zorder = 3)
    plt.errorbar(ch2["binned_BJD"], ch2["binned_flux"], yerr=ch2["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch2["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity")
    plt.ylim(1-0.017, 1+0.013)
    plt.title(r"4.5 $\mu$m Transit" )

    plt.subplot(132)
    plt.scatter(ch1["BJD_decimal"], ch1["flux"], s=1, color="gray", alpha=0.3)
    plt.plot(ch1["binned_BJD"], ch1["binned_lc"], color=lc_col, zorder = 3)
    plt.errorbar(ch1["binned_BJD"], ch1["binned_flux"], yerr=ch1["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch1["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity")
    plt.ylim(1-0.017, 1+0.013)
    plt.title(r"3.6 $\mu$m Transit")

    plt.subplot(133)
    plt.plot(kepler["phase_time"], kepler["lc_php"], color=lc_col)
    plt.scatter(kepler["phasefold_results"], kepler["flux"], s=9,linewidths = 0.5, edgecolors="black", color=point_col)
    plt.title("Kepler Phasefold Transit")
    plt.xlabel("Phase")
    plt.ylabel("Normalized Intensity")
    plt.xlim(-0.1, 0.1)

    if savefigs:
        fig_path = fig_dir / "all_lcs_blue.png"
        plt.savefig(fig_path)
    else:
        plt.show()

    # plot spitzer CH 2
    plt.figure(figsize=(7, 5), dpi=750)
    plt.scatter(ch2["BJD_decimal"], ch2["flux"], s=1, color="gray", alpha=0.3)
    plt.plot(ch2["binned_BJD"], ch2["binned_lc"], color=lc_col, zorder = 3)
    plt.errorbar(ch2["binned_BJD"], ch2["binned_flux"], yerr=ch2["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch2["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity")
    plt.ylim(1-0.017, 1+0.013)
    plt.title(r"4.5 $\mu$m Transit" )

# Plot phasefolded Kepler LC
if plot_kepler:
    plt.figure(figsize=(7, 5), dpi=750)
    plt.plot(kepler["phase_time"], kepler["lc_php"], color=lc_col)
    plt.scatter(kepler["phasefold_results"], kepler["flux"], s=9,linewidths = 0.5, edgecolors="black", color=point_col)
    plt.title("Kepler Phasefold Transit")
    plt.xlabel("Phase")
    plt.ylabel("Normalized Intensity")
    plt.xlim(-0.1, 0.1)
    if savefigs:
        fig_path = fig_dir / "kepler_ph_lc.png"
        plt.savefig(fig_path)
    else:
        plt.show()

# plot CH 1, CH 2 sensitivity map
if plot_bliss:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=750)
    p1 = axes[0].scatter(ch1["x"], ch1["y"], c=ch1["blissflux"],
                cmap = bliss_cmap, marker="s", s=12)
    axes[0].text(18.3, 18.55, r"3.6 $\mu$m", size=20)
    axes[0].set_xlabel("X-pixel")
    axes[0].set_ylabel("Y-pixel")
    fig.colorbar(p1, ax = axes[0])

    p1 = axes[1].scatter(ch2["x"], ch2["y"], c=ch2["blissflux"],
                cmap = bliss_cmap, marker="s", s=12)
    #fig.colorbar(label="Sensitivity")
    axes[1].text(19.12, 19.16, r"4.5 $\mu$m", size=20)
    axes[1].set_xlabel("X-pixel")
    axes[1].set_ylabel("Y-pixel")

    fig.colorbar(p1, ax = axes[1])
    plt.tight_layout()
    if savefigs:
        fig_path = fig_dir / "BLISS_ch1ch2.png"
        plt.savefig(fig_path)
    else:
        plt.show()

# Plots spitzer residuals
if plot_residuals:
    plt.figure(figsize=(7*3, 5), dpi=750)
    plt.subplot(131)
    plt.scatter(ch1["BJD_decimal"], ch1["residuals"], s=1, color="gray", alpha=0.3)
    plt.axhline(0, color=lc_col, zorder = 3)
    plt.errorbar(ch1["binned_BJD"], ch1["binned_residuals"], yerr=ch1["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch1["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity - Model")
    plt.xlim(-0.0016 ,0.0016)
    plt.title(r"3.6 $\mu$m Residuals" )

    # plot spitzer CH 2
    #plt.figure(figsize=(7, 5), dpi=750)
    plt.subplot(132)
    plt.scatter(ch2["BJD_decimal"], ch2["residuals"], s=1, color="gray", alpha=0.3)
    plt.axhline(0, color=lc_col, zorder = 3)
    plt.errorbar(ch2["binned_BJD"], ch2["binned_residuals"], yerr=ch2["binned_err"], fmt="o",
                 ecolor=point_col, elinewidth = 1, markersize=3, markerfacecolor=point_col, markeredgecolor="black")
    bjd_value = round(np.floor(ch2["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity - Model")
    plt.xlim(-0.0016 ,0.0016)
    plt.title(r"4.5 $\mu$m Residuals")
    # if savefigs:
    #     fig_path = fig_dir / "spitzer_ch2_residuals.png"
    #     plt.savefig(fig_path)
    # else:
    #     plt.show()

    # plot Kepler

    #plt.figure(figsize=(7, 5), dpi=750)
    plt.subplot(133)
    plt.scatter(kepler["phasefold_results"], kepler["flux"] - lc(just_params, kepler, ch="kepler", SuperSample=True), s=9, color=point_col)
    plt.axhline(0, color=lc_col, zorder=3)
    bjd_value = round(np.floor(kepler["time"][0]))
    plt.xlabel(r"${BJD}_{TBD} -" + f"{bjd_value}$")
    plt.ylabel("Normalized Intensity - Model")
    plt.title(r"Kepler Residuals")

    plt.tight_layout()

    if savefigs:
        fig_path = fig_dir / "all_residuals.png"
        plt.savefig(fig_path)
    else:
        plt.show()

# Plots depths as a function of wavelength
if plot_depths:
    plt.figure(figsize=(5, 5), dpi=750)
    lam = 10000 # conversion from angstroms to microns
    ch1_lam = 35074.83 / lam
    ch2_lam = 44365.56/ lam
    kepler_lam = 5978.14/ lam

    filter_lams = [ch1_lam, ch2_lam, kepler_lam]
    depths = [just_params[11], just_params[12], just_params[13]]
    depth_errs = [just_errs[11], just_errs[12], just_errs[13]]

    labels = ["Spitzer CH 1", "Spitzer CH 2", "Kepler"]

    for i in range (len(depths)):
        plt.errorbar(filter_lams[i], depths[i], yerr=depth_errs[i], fmt="^", color=lam_colors[i])
        plt.axhline(depths[i], color=lam_colors[i], linestyle="--", label=labels[i], alpha=0.5)

    plt.title("Transit depth per wavelength of observation ")
    plt.xlabel(r"Filter $\lambda_{eff}$ ($\mu m$)")
    plt.ylabel("Transit Depth $(R_{P}/R_{*})^2$  (ppm)")
   # plt.ylim(11000, 7400)
    plt.grid(alpha=0.3)
    plt.legend()

    if savefigs:
        fig_path = fig_dir / "transit_depths.png"
        plt.savefig(fig_path)
    else:
        plt.show()

# Plots trend detrending sequence
if plot_sequence:
    ramp1 = (1 + just_params[8] * (ch1["time"] - np.median(ch1["time"])))
    ramp2 = (1 + just_params[9] * (ch2["time"] - np.median(ch2["time"])))

    fig, axes = plt.subplots(3, 5, figsize=(14, 6), dpi=750)
    axes[0][0].scatter(ch1["time"], ch1["raw_flux"], s=1, color=point_col)
    axes[1][0].scatter(ch2["time"], ch2["raw_flux"], s=1, color=point_col)
    axes[0][0].set_ylabel("CH 1",size=18)
    axes[1][0].set_ylabel("CH 2",size=18)
    axes[0][0].set_title("Normalized flux \n flux/baseline flux", size=18)

    axes[0][1].plot(ch1["time"], ch1["mcmc_lc"] * ch1["blissflux"], color=lc_col, linewidth=1, alpha=0.7)
    axes[1][1].plot(ch2["time"], ch2["mcmc_lc"] * ch2["blissflux"], color=lc_col, linewidth=1, alpha=0.7)
    axes[0][1].scatter(ch1["time"], ch1["raw_flux"], s=1, color=point_col)
    axes[1][1].scatter(ch2["time"], ch2["raw_flux"], s=1, color=point_col)
    axes[0][1].set_title("Initial fitting \nlc+BLISS+ramp",size=18)

    axes[0][2].scatter(ch1["time"], ch1["raw_flux"]/ch1["blissflux"], s=1, color=point_col)
    axes[1][2].scatter(ch2["time"], ch2["raw_flux"]/ch2["blissflux"], s=1, color=point_col)
    axes[0][2].set_title("Detrended flux \nflux-BLISS-ramp",size=18)

    axes[0][3].scatter(ch1["time"], ch1["flux"], s=1, color=point_col)
    axes[1][3].scatter(ch2["time"], ch2["flux"], s=1, color=point_col)
    axes[0][3].plot(ch1["time"], ch1["mcmc_lc"], color=lc_col, linewidth=1)
    axes[1][3].plot(ch2["time"], ch2["mcmc_lc"], color=lc_col, linewidth=1)
    axes[0][3].set_title("Model fit \nflux-BLISS-ramp + lc",size=18)

    axes[0][4].scatter(ch1["time"], ch1["flux"]-ch1["mcmc_lc"], s=1, color=point_col)
    axes[1][4].scatter(ch2["time"], ch2["flux"]-ch2["mcmc_lc"], s=1, color=point_col)
    axes[0][4].set_title("Residuals \nflux-BLISS-ramp-lc",size=18)

    axes[2][0].scatter(kepler["time"], kepler["flux"], s=1, color=point_col)
    axes[2][0].set_ylabel("Kepler" ,size=18)
    axes[2][0].set_title("Raw Kepler data",size=18)
    axes[2][1].scatter(kepler["time"], kepler["flux"], s=1, color=point_col)
    axes[2][1].plot(kepler["time"], kepler["mcmc_lc"], color=lc_col)
    axes[2][1].set_title("MCMC 1 fit Kepler Data \n for period, T0",size=18)
    axes[2][2].scatter(kepler["phasefold_results"], kepler["flux"], s=1, color=point_col)
    axes[2][2].set_title("Phasefolded Kepler Data \n for MCMC 2 fitting",size=18)
    axes[2][2].set_xlim(-0.1, 0.1)
    axes[2][3].scatter(kepler["phasefold_results"], kepler["flux"], s=1, color=point_col)
    axes[2][3].plot(kepler["phase_time"], kepler["lc_php"], color=lc_col)
    axes[2][3].set_title("MCMC 2 fit \n on phasefold",size=18)
    axes[2][3].set_xlim(-0.1, 0.1)
    axes[2][4].scatter(kepler["time"], kepler["residuals"], s=1, color=point_col)
    axes[2][4].set_title("Residuals \n Kepler Data - Kepler lc",size=18)

    plt.setp(axes, xticks=[], yticks=[])

    plt.tight_layout()
    if savefigs:
        fig_path = fig_dir / "sequence_lcs.png"
        plt.savefig(fig_path)
    else:
        plt.show()

if plot_all_lc2:
    plt.figure(figsize=(7, 5), dpi=750)
    plt.plot(ch1_nonbin_ph["phase_time"], ch1_nonbin_ph["lc_php"], color=lam_colors[0])
    plt.scatter(ch1["phasefold_results"], ch1["binned_flux"], s=9, color=lam_colors[0], alpha=0.5, label="Spitzer Ch 1")

    plt.plot(ch2_nonbin_ph["phase_time"], ch2_nonbin_ph["lc_php"], color=lam_colors[1])
    plt.scatter(ch2["phasefold_results"], ch2["binned_flux"], s=9, color=lam_colors[1], alpha=0.5, label="Spitzer Ch 2")

    plt.plot(kepler["phase_time"], kepler["lc_php"], color=lam_colors[2])
    plt.scatter(kepler["phasefold_results"], kepler["flux"], s=9, color=lam_colors[2], alpha=0.5, label="Kepler")

    plt.xlabel("Phase")
    plt.legend()
    plt.title("Phasefolded Lightcurves Overplotted")
    plt.ylabel("Normalized Intensity")
    plt.xlim(-0.025, 0.025)
    if savefigs:
        fig_path = fig_dir / "stacked_lcs.png"
        plt.savefig(fig_path)
    else:
        plt.show()

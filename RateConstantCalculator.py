# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:27:15 2026

@author: Usama
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scipy.io as spio
import SCT

# ============================================================
# User settings
# ============================================================
Trange = [200]

calculate_ZCT = True
calculate_SCT = True

# If True, use true E0/VAG/SAG from alldf in the estimated run
# If False, estimated run computes its own E0/VAG/SAG
use_trueVAGandE0 = True

base_dir = "../"
systems = os.listdir(base_dir)
systems = [folder for folder in systems if ("Rad" in folder or "Oxo" in folder)]
folders = [folder for folder in systems if os.path.isdir(os.path.join(base_dir, folder))]
#folders=['Rad']
#folders=['Rad', 'Rad_C3', 'Rad_C5', 'Rad_CH3', 'Rad_NO2']   


#folders=['Rad', 'Rad_C2', 'Rad_C5', 'Rad_CH3', 'Rad_NO2'] 

lams=[0.01, 0.99]  # or set manually, e.g. ["Rad", "Rad_C2", "Oxo"]


pd.options.mode.chained_assignment = None
tag='final'

# ============================================================
# Helpers
# ============================================================
REQUIRED_ALLDF_COLS = [
    "s",
    "Energy",
    "Gradient_notnormalized",
    "Hessian",
    "Eigenvalues",
    "Eigenvectors",
]

REQUIRED_RECONDF_COLS = [
    "Hessian",
    "Eigenvalues",
    "Eigenvectors",
]


def validate_columns(df, required_cols, name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def build_true_inputs(alldf):
    """
    True calculation:
      - path quantities from alldf
      - vibrational quantities from alldf
    """
    return {
        "svals": alldf["s"],
        "Vmep": alldf["Energy"],
        "grads": alldf["Gradient_notnormalized"],
        "Hessians": alldf["Hessian"],
        "evals": alldf["Eigenvalues"],
        "evecs": alldf["Eigenvectors"],
    }


def build_estimated_inputs(alldf, recondf):
    """
    Estimated calculation:
      - path quantities from alldf
      - reconstructed vibrational quantities from recondf
    """
    return {
        "svals": alldf["s"],
        "Vmep": alldf["Energy"],
        "grads": alldf["Gradient_notnormalized"],
        "Hessians": recondf["Hessian"],
        "evals": recondf["Eigenvalues"],
        "evecs": recondf["Eigenvectors"],
    }


def compute_tunneling(T, inputs, use_zct=False, E0_arg="calc", VAG_arg="calc", SAG_arg="calc"):
    """
    Wrapper around SCT.sct for cleaner calls.
    """
    return SCT.sct(
        T,
        inputs["Vmep"],
        inputs["svals"],
        E0_arg,
        VAG_arg,
        SAG_arg,
        inputs["grads"],
        inputs["Hessians"],
        inputs["evals"],
        inputs["evecs"],
        if_ZCT=use_zct,
    )


# ============================================================
# Main loop
# ============================================================

# for lam in lams: # For hyperparameter optimization
#     tag='_lam'+str(lam*100)
#     print(lam)


sys_list=[]

zct_true_list=[]
zct_est_list=[]

sct_true_list=[]
sct_est_list=[]


# for lam in lams:
#     tag=f'_lam{lam*100}'
#     print(f'Value of lambda is {lam}')
for folder in folders:
    system = folder
    sys_list.append(system)
    print(f"\n{'='*70}")
    print(f"System: {system}")
    print(f"{'='*70}")

    # File paths
    alldf_path = f"../{folder}/MatPKL/{system}.pkl"
    recondf_path = f"../{system}/{system}_{tag}.pkl"

    # Optional .mat files if you still want to inspect them later
    true_mat_path = f"../{folder}/ResultMATs/results_{system}.mat"
    est_mat_path = f"../{folder}/result_{system}_Hesscg_{tag}.mat"

    # Load dataframes
    alldf = pd.read_pickle(alldf_path)
    recondf = pd.read_pickle(recondf_path)

    validate_columns(alldf, REQUIRED_ALLDF_COLS, "alldf")
    validate_columns(recondf, REQUIRED_RECONDF_COLS, "recondf")

    # Ensure row-wise alignment
    if len(alldf) != len(recondf):
        raise ValueError(
            f"Length mismatch for {system}: len(alldf)={len(alldf)} != len(recondf)={len(recondf)}"
        )

    # Align reconstructed dataframe index to alldf index for consistency
    recondf = recondf.copy()
    recondf.index = alldf.index

    # Build input bundles
    true_inputs = build_true_inputs(alldf)
    est_inputs = build_estimated_inputs(alldf, recondf)

    # Optional loading of .mat files, only if present
    Xtrue_mat = None
    Xest_mat = None

    if os.path.exists(true_mat_path):
        results = spio.loadmat(true_mat_path)
        if "Xtrue" in results:
            Xtrue_mat = np.array(results["Xtrue"])

    if os.path.exists(est_mat_path):
        est_results = spio.loadmat(est_mat_path)
        if "Xvmc2" in est_results:
            Xest_mat = np.array(est_results["Xvmc2"])

    # Containers
    ZCT_true_vals = []
    ZCT_est_vals = []
    SCT_true_vals = []
    SCT_est_vals = []

    Ks_true_all = []
    Ks_est_all = []
    BmFs_true_all = []
    BmFs_est_all = []

    # These may be reused in estimated run if use_trueVAGandE0=True
    saved_true_E0 = None
    saved_true_VAG = None
    saved_true_SAG = None

    # ------------------------------------------------------------
    # TRUE calculations
    # ------------------------------------------------------------
    print("\n--- TRUE calculations (all vibrational data from alldf) ---")

    for T in Trange:
        if calculate_ZCT:
            (
                ZCT_true_val,
                E0_true_zct,
                VAG_true_zct,
                SAG_true_zct,
                V_aG_true_zct,
                BmFs_true_zct,
                Ks_true_zct,
            ) = compute_tunneling(
                T,
                true_inputs,
                use_zct=True,
                E0_arg="calc",
                VAG_arg="calc",
                SAG_arg="calc",
            )

            print(f"True ZCT at {T} K: {ZCT_true_val}")
            ZCT_true_vals.append(ZCT_true_val)

        if calculate_SCT:
            (
                SCT_true_val,
                E0_true_sct,
                VAG_true_sct,
                SAG_true_sct,
                V_aG_true_sct,
                BmFs_true_sct,
                Ks_true_sct,
            ) = compute_tunneling(
                T,
                true_inputs,
                use_zct=False,
                E0_arg="calc",
                VAG_arg="calc",
                SAG_arg="calc",
            )

            print(f"True SCT at {T} K: {SCT_true_val}")
            SCT_true_vals.append(SCT_true_val)
            BmFs_true_all.append(BmFs_true_sct)
            Ks_true_all.append(Ks_true_sct)

            # Save barrier quantities from true run if requested
            if use_trueVAGandE0:
                saved_true_E0 = E0_true_sct
                saved_true_VAG = VAG_true_sct
                saved_true_SAG = SAG_true_sct

    # ------------------------------------------------------------
    # ESTIMATED calculations
    # ------------------------------------------------------------
    print("\n--- ESTIMATED calculations (path from alldf, vib. data from recondf) ---")

    for T in Trange:
        if use_trueVAGandE0 and saved_true_E0 is not None:
            E0_arg = saved_true_E0
            VAG_arg = saved_true_VAG
            SAG_arg = saved_true_SAG
        else:
            E0_arg = "calc"
            VAG_arg = "calc"
            SAG_arg = "calc"

        if calculate_ZCT:
            (
                ZCT_est_val,
                E0_est_zct,
                VAG_est_zct,
                SAG_est_zct,
                V_aG_est_zct,
                BmFs_est_zct,
                Ks_est_zct,
            ) = compute_tunneling(
                T,
                est_inputs,
                use_zct=True,
                E0_arg=E0_arg,
                VAG_arg=VAG_arg,
                SAG_arg=SAG_arg,
            )

            print(f"Estimated ZCT at {T} K: {ZCT_est_val}")
            ZCT_est_vals.append(ZCT_est_val)

        if calculate_SCT:
            (
                SCT_est_val,
                E0_est_sct,
                VAG_est_sct,
                SAG_est_sct,
                V_aG_est_sct,
                BmFs_est_sct,
                Ks_est_sct,
            ) = compute_tunneling(
                T,
                est_inputs,
                use_zct=False,
                E0_arg=E0_arg,
                VAG_arg=VAG_arg,
                SAG_arg=SAG_arg,
            )

            print(f"Estimated SCT at {T} K: {SCT_est_val}")
            SCT_est_vals.append(SCT_est_val)
            BmFs_est_all.append(BmFs_est_sct)
            Ks_est_all.append(Ks_est_sct)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n--- Summary ---")
    if calculate_ZCT:
        for T, z_true, z_est in zip(Trange, ZCT_true_vals, ZCT_est_vals):
            print(f"T = {T} K | True ZCT = {z_true} | Estimated ZCT = {z_est}")

    if calculate_SCT:
        for T, s_true, s_est in zip(Trange, SCT_true_vals, SCT_est_vals):
            print(f"T = {T} K | True SCT = {s_true} | Estimated SCT = {s_est}")
            
    zct_true_list.append(z_true)
    zct_est_list.append(z_est)
    
    sct_true_list.append(s_true)
    sct_est_list.append(s_est)



sys_array=np.array(sys_list)

z_tr_array=np.array(zct_true_list)
s_tr_array=np.array(sct_true_list)

z_array=np.array(zct_est_list)
s_array=np.array(sct_est_list)

df=pd.DataFrame({'system':sys_array, 'ZCT_True':z_tr_array,
                 'ZCT_estimated': z_array, 'SCT_True': s_tr_array,
                 'SCT_estimated:': s_array})


df.to_csv('tunneling_H.csv')

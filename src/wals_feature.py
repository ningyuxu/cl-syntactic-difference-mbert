import os
import json
from tqdm import tqdm
import re
import pandas as pd
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import subspace_angles

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, RepeatedKFold, \
    permutation_test_score
from sklearn.inspection import permutation_importance

from datasets.constant import ISO638_TO_LANG
from datasets import constant

LANGS_ISO = ['ar', 'bg', 'de', 'el', 'en', 'es', 'et', 'fa', 'fi',
             'fr', 'he', 'hi', 'it', 'ja', 'ko', 'lv', 'nl', 'pl',
             'pt', 'ro', 'ru', 'tr', 'vi', 'zh']

LANG_TO_WALSNAME = {
    "Arabic": ["ams", "aeg", "ako", "amr", "anl", "apa", "arg", "arj", "arn", "ars", "asy", "atu", "abb", ],
    "Bulgarian": ["bul", ],
    "Czech": ["cze", ],
    "Catalan": ["ctl", ],
    "Estonian": ["est", ],
    "German": ["gbl", "ger", "gha", "gma", "gth", "gti"],
    "Greek": ["grk", "gcy"],
    "English": ["eng", ],
    "Spanish": ["spa", "spc", ],
    "Persian": ["prs", ],
    "Finnish": ["fin", ],
    "French": ["fre", ],
    "Hebrew": ["heb", "hba"],
    "Hindi": ["hin", ],
    "Hungarian": ["hun", ],
    "Italian": ["ita", ],
    "Japanese": ["jpn", ],
    "Korean": ["kor", ],
    "Latvian": ["lat", ],
    "Dutch": ["dut", ],  # dbr, dli, duz
    "Polish": ["pol", ],
    "Portuguese": ["por", ],
    "Romanian": ["rom", "mol"],
    "Russian": ["rus", ],
    "Tamil": ["tml", "tsp", ],
    "Turkish": ["tur"],
    "Urdu": ["urd"],
    "Vietnamese": ["vie"],
    "Chinese": ["mnd", "kug"],
}


def read_wals_data(
        value_path="../data/values.csv",
        code_path="../data/codes.csv",
        parameters_path="../data/parameters.csv",
        langs_iso=None,
        lang_to_wals=None,
        kind="simple",
):
    if lang_to_wals is None:
        lang_to_wals = LANG_TO_WALSNAME
    if langs_iso is None:
        langs_iso = LANGS_ISO
    data = pd.read_csv(value_path)
    feature_idx = [str(i) for i in range(20, 129)]
    feature_idx.extend([str(i) for i in range(143, 145)])
    pattern_idx = "|".join(feature_idx)
    pattern_str = f"({pattern_idx})" + "[A-Z]"
    pattern = re.compile(pattern_str)

    feature_ls = [i for i in list(set(data.Parameter_ID)) if re.match(pattern, i)]
    df = data[data.Parameter_ID.isin(feature_ls)]

    code_ls = list(set(df.Code_ID))
    pattern = re.compile(pattern_idx)
    code_ls = sorted(
        code_ls,
        key=lambda x: (int(re.match(pattern, x).group(0)),
                       re.search("[A-Z]", x).group(0),
                       int(x.split("-")[1]))
    )
    lang_ls = sorted([lang for lang in list(set(df.Language_ID))
                      if isinstance(lang, str)])
    parameter_ls = sorted(
        list(set(df.Parameter_ID)),
        key=lambda x: (int(re.match(pattern, x).group(0)),
                       re.search("[A-Z]", x).group(0))
    )

    df_langfeat = pd.DataFrame(index=lang_ls, columns=parameter_ls)

    for lang in tqdm(lang_ls):
        for parameter in df.loc[df.Language_ID == lang].Parameter_ID:
            df_langfeat.loc[lang][parameter] = int(df[(df.Language_ID == lang) & (df.Parameter_ID == parameter)].Value)

    langs = [ISO638_TO_LANG[lang] for lang in langs_iso]
    df_langs = pd.DataFrame(columns=list(df_langfeat.columns), index=langs)

    for lang in langs:
        for feat in list(df_langs.columns):
            for langid in LANG_TO_WALSNAME[lang]:
                keep_values = []
                if (langid in df_langfeat.index) and (not pd.isna(df_langfeat.loc[langid][feat])):
                    keep_values.append(df_langfeat.loc[langid][feat])
                    df_langs.loc[lang][feat] = keep_values[0]

    to_drop = []
    for column in list(df_langs.columns):
        if column.endswith("A") and df_langs[column].isnull().values.any():
            df_langs.drop(columns=column, inplace=True)
            to_drop.append(column.replace("A", ""))
        elif column.startswith(to_drop[-1]) and not column.endswith("A"):
            df_langs.drop(columns=column, inplace=True)
        else:
            continue

    df_codes = pd.read_csv(code_path)
    df_parameters = pd.read_csv(parameters_path)
    features = []
    for feat in list(df_langs.columns):
        features.extend(df_codes[df_codes.Parameter_ID == feat].ID.values)
    features = [feat.replace("-", "_") for feat in features]

    df_final = pd.DataFrame(columns=features, index=langs)
    for lang in langs:
        for feat in list(df_langs.columns):
            df_final.loc[lang]["_".join((feat, str(df_langs.loc[lang][feat])))] = 1

    code_dict = {}
    for feat in list(df_final.columns):
        feat_name = df_parameters[df_parameters.ID == feat.split("_")[0]].Name.values[0] \
                    + " _ " + df_codes[df_codes.ID == "-".join(feat.split("_"))].Name.values[0]
        code_dict[feat] = feat_name

    # Process Features

    # Feature 81
    mapping = {}
    word_order_keys = [k for k in code_dict.keys() if k.startswith("81A")]
    du_word_order_keys = [k for k in code_dict.keys() if k.startswith("81B")]
    for key in du_word_order_keys:
        mapping[key] = []
        dom_word_orders = code_dict[key].split("_")[-1].strip().split(" or ")
        for wo in word_order_keys:
            if code_dict[wo].split("_")[-1].strip() in dom_word_orders:
                mapping[key].append(wo)

    for lang in langs:
        for param in mapping.keys():
            if df_final.loc[lang][param] == 1:
                for p in mapping[param]:
                    df_final.loc[lang][p] = 1

    df_final.drop(columns=list(mapping.keys()) + ["81A_7"], inplace=True)

    # Feature 90
    mapping = {
        "90B_1": ["90A_2"],
        "90B_2": ["90A_1", "90A_2"],
        "90B_3": ["90A_2", "90A_3"],
        "90B_4": ["90A_2", "90A_4"],
        "90B_5": ["90A_2", "90A_6"],
        "90C_1": ["90A_1"],
        "90C_2": ["90A_1", "90A_2"],
        "90C_3": ["90A_1", "90A_3"],
        "90C_4": ["90A_1", "90A_4"],
        "90D_1": ["90A_3"],
        "90D_2": ["90A_2", "90A_3"],
        "90D_3": ["90A_1", "90A_3"],
        "90D_4": ["90A_3", "90A_4"],
        "90D_5": ["90A_3", "90A_6"],
        "90D_6": ["90A_3.5"],
        "90D_7": ["90A_3.25"],
        "90E_1": ["90A_4"],
        "90E_2": ["90A_2", "90A_4"],
        "90E_3": ["90A_1", "90A_4"],
        "90E_4": ["90A_3", "90A_4"],
        "90E_5": ["90A_4", "90A_5"],
        "90E_6": ["90A_4.5"],
        "90E_7": ["90A_4.25"],
        "90F_1": ["90A_5"],
        "90F_2": ["90A_4", "90A_5"],
        "90G_1": ["90A_6"],
        "90G_2": ["90A_2", "90A_6"],
        "90G_3": ["90A_3", "90A_6"],
        "90G_4": ["90A_6.5"],
    }

    for lang in langs:
        for param in mapping.keys():
            if df_final.loc[lang][param] == 1:
                for p in mapping[param]:
                    if "." in p:
                        df_final.loc[lang][p.split(".")[0]] = float("0." + p.split(".")[1])
                    else:
                        df_final.loc[lang][p] = 1.0

    df_final.drop(columns=list(mapping.keys()) + ["90A_7"], inplace=True)

    # Feature 143
    df_final.drop(columns=[
        # "143A_16", "143A_17",
        "143E_4", "143F_4", "143G_4"
    ], inplace=True)
    mapping = {
        "143A_6": ["143A_1", "143A_2"],
        "143A_7": ["143A_1", "143A_3"],
        "143A_8": ["143A_1", "143A_4"],
        "143A_9": ["143A_2", "143A_3"],
        "143A_10": ["143A_2", "143A_4"],
        "143A_11": ["143A_3", "143A_4"],
        "143A_12": ["143A_3", "143G_2"],
        "143B_10": ["143B_1", "143B_2"],
        "143B_11": ["143B_1", "143B_3"],
        "143B_12": ["143B_1", "143A_5", "143A_2"],
        "143B_13": ["143B_1", "143B_7"],
        "143B_14": ["143B_1", "143B_9"],
        "143B_15": ["143B_2", "143B_4"],
        "143B_5": ["143A_5", "143A_2"],
        "143B_6": ["143A_5", "143A_3"],
        "143B_16": ["143A_16"],
        "143C_23": ["143A_17"],
        "143C_1": ["143A_1", "143B_1"],
        "143C_2": ["143A_2", "143B_1"],
        "143C_3": ["143A_1", "143B_2"],
        "143C_4": ["143A_4", "143B_2"],
        "143C_5": ["143A_3", "143B_3"],
        "143C_6": ["143A_2", "143B_3"],
        "143C_7": ["143A_3", "143B_4"],
        "143C_8": ["143A_4", "143B_4"],
        "143C_9": ["143A_2", "143B_9"],
        "143C_10": ["143A_4", "143B_9.5"],
        "143C_11": ["143A_1", "143B_7"],
        "143C_12": ["143A_1", "143B_8"],
        "143C_13": ["143A_1", "143G_3.5"],
        "143C_14": ["143A_1", "143B_4"],
        "143C_15": ["143A_2", "143B_4"],
        "143C_16": ["143A_3", "143B_1"],
        "143C_17": ["143A_1", "143A_2", "143A_5"],
        "143C_18": ["143A_1", "143A_4", "143B_2"],
        "143C_19": ["143A_1", "143A_2", "143B_1"],
        "143C_20": ["143A_1", "143A_4", "143B_4"],
        "143C_21": ["143A_3", "143A_2", "143B_4"],
        "143C_22": ["143A_1", "143A_3", "143B_8"],
        "143D_1": ["143B_4", "143D_1"],
        "143D_2": ["143B_9.5", "143G_2", "143D_2"],
        "143D_3": ["143B_1", "143D_3"],
        "143D_4": ["143B_2", "143B_1", "143A_5"],
        "143D_5": ["143B_1", "143B_3", "143D_5"],
        "143D_6": ["143B_2", "143B_1"],
        "143E_1": ["143A_1"],
        "143E_2": ["143A_3"],
        "143E_3": ["143A_1", "143A_3"],
        "143F_1": ["143A_2"],
        "143F_2": ["143A_4"],
        "143F_3": ["143A_2", "143A_4"],
        "143G_1": ["143A_5"],
    }

    for lang in langs:
        for param in mapping.keys():
            if df_final.loc[lang][param] == 1:
                for p in mapping[param]:
                    if "." in p:
                        df_final.loc[lang][p.split(".")[0]] = float("0." + p.split(".")[1])
                    else:
                        df_final.loc[lang][p] = 1.0
    to_drop = [k for k in list(mapping.keys()) if k not in [item for sublist in mapping.values() for item in sublist]]
    df_final.drop(columns=to_drop, inplace=True)
    df_final.drop(columns=["143A_16", "143A_17", ], inplace=True)

    # Feature 144
    mapping = {
        "144C_1": ["144A_1", "143A_1"],
        "144C_2": ["144A_6", "143A_1"],
        "144C_3": ["144A_7", "143A_1"],
        "144C_4": ["144A_8", "143A_2"],
        "144C_5": ["144A_9", "143A_1"],
        "144C_6": ["144A_8", "143A_4"],
        "144C_7": ["144A_7", "143A_3"],
        "144C_8": ["144A_1", "144A_13", "143A_3", "143A_1"],
        "144C_9": ["144N_2", "143B_7"],
        "144C_10": ["144N_1", "143B_2", "143B_4"],
        "144C_11": ["144N_1", "143B_2", "144F_2", "143B_4"],
        "144C_12": ["144F_4", "143B_1"],
        "144C_13": ["144A_9", "144U_3", "143A_3", "143B_3"],
        "144C_14": ["144A_4", "143A_2"],
        "144C_15": ["144A_2", "143A_1"],
        "144C_16": ["144A_6", "143A_1"],
        "144C_17": ["144A_8", "143A_2"],
        "144D_1": ["144A_1", "143A_1"],
        "144D_2": ["144A_2", "143A_1"],
        "144D_3": ["144A_3", "143A_2"],
        "144D_4": ["144A_4", "143A_2"],
        "144D_5": ["144A_2", "143A_3"],
        "144D_6": ["144A_3", "143A_4"],
        "144D_7": ["81A_2", "143A_5"],
        "144D_8": ["144A_1", "143A_1"],
        "144D_9": ["144A_4", "143A_2"],
        "144D_10": ["144A_2", "143A_1"],
        "144D_11": ["143A_1"],
        "144D_12": ["143A_2"],
        "144D_14": ["144A_18"],
        "144D_15": ["144A_19"],
        "144E_1": ["144A_2", "144A_3", "143A_1", "143A_2"],
        "144E_2": ["144A_1", "144A_2", "143A_1"],
        "144E_3": ["144A_1", "144A_3", "143A_1", "143A_2"],
        "144E_4": ["144A_2", "144A_4", "143A_1", "143A_2"],
        "144E_5": ["144A_3", "144A_4", "143A_2"],
        "144E_6": ["144A_16", "81A_2"],
        "144E_7": ["144A_2", "144A_3", "143A_4", "143A_1"],
        "144E_8": ["144A_2", "144A_3", "143A_3", "143A_2"],
        "144E_9": ["144A_1", "144A_2", "144A_3", "143A_4", "143A_1"],
        "144E_10": ["144A_2", "144A_9", "143A_1"],
        "144E_11": ["144A_4", "144A_11", "143A_2"],
        "144E_12": ["144A_1", "144A_12", "143A_1"],
        "144E_13": ["144A_2", "144A_12", "143A_1"],
        "144E_14": ["144A_2", "144A_12", "143A_3"],
        "144E_15": ["144A_4", "144A_7", "143A_1", "143A_2"],
        "144E_16": ["144A_3", "144A_8", "143A_2"],
        "144E_17": ["144A_4", "144A_6", "144A_8", "143A_1", "143A_2"],
        "144E_18": ["144A_1", "144A_3", "144A_5", "144A_6", "143A_1", "143A_2"],
        "144E_19": ["144A_2", "144A_7", "144A_4", "144A_8", "143A_1", "143A_2"],
        "144E_20": ["144A_2", "144A_7", "143A_3"],
        "144E_21": ["144A_3", "144A_8", "143A_4"],
        "144E_22": ["144A_2", "144A_7", "143A_3", "143A_1"],
        "144E_23": ["144A_1", "144A_13", "143A_3", "143A_1"],
        "144E_24": ["143A_1", "143A_2"],
        "144E_25": ["143A_1", "143A_4"],
        "144E_26": ["143A_1"],
        "144E_27": ["143A_1"],
        "144E_28": ["143A_1", "143A_2"],
        "144F_1": ["144F_1", "143B_1"],
        "144F_2": ["144F_2", "143B_1"],
        "144F_3": ["144F_3", "143B_9"],
        "144F_4": ["144F_4", "143B_1"],
        "144F_5": ["144F_1", "143B_3"],
        "144F_6": ["144F_2", "143B_2"],
        "144F_7": ["144F_2", "143B_3"],
        "144F_8": ["143B_2", "144F_8"],
        "144F_9": ["144F_2", "143B_4"],
        "144F_10": ["144A_3", "143A_5"],
        "144F_11": ["144A_2", "143A_5", "143A_3"],
        "144F_12": ["144F_2", "144F_12", "143B_1", "143B_7"],
        "144F_13": ["144F_1", "144F_2", "143B_1"],
        "144F_14": ["144F_1", "143B_3", "143B_1"],
        "144F_15": ["144F_1", "144F_4", "143B_1", "144F_15"],
        "144F_16": ["144F_2", "143A_5", "143B_1", "143B_2"],
        "144F_17": ["144F_2", "144F_17", "143B_4"],
        "144F_18": ["144F_1", "144F_18", "143B_1", "143B_3"],
        "144F_19": ["144F_1", "144F_19", "143B_1"],
        "144F_20": ["144A_4", "143A_5", "144F_19", "143B_1"],
        "144F_21": ["144F_21", "144F_1", "144F_3", "143B_1", "143B_9"],
        "144F_22": ["143B_1"],
        "144F_23": ["144A_2", "143A_1", "143A_3"],
        "144G_1": ["144F_1", "144A_2", "143A_1", "143B_1"],
        "144G_2": ["144F_1", "144A_4", "143A_2", "143B_1"],
        "144G_3": ["144F_2", "144A_3", "143A_2", "143B_1"],
        "144G_4": ["144F_2", "144A_2", "143A_1", "143B_1"],
        "144G_5": ["144F_4", "144A_4", "143A_2", "143B_1"],
        "144G_6": ["144F_3", "144A_3", "143A_2", "143B_9"],
        "144G_7": ["143B_2", "144F_8", "144A_1", "143A_1"],
        "144G_8": ["144F_2", "143B_3", "144A_3", "143A_2"],
        "144G_9": ["144F_1", "143B_3", "144A_2", "143A_3"],
        "144G_10": ["144A_2", "144A_3", "143A_5", "143A_1", "143A_2"],
        "144G_11": ["144F_2", "143B_2", "144A_2", "144A_3", "143A_1", "143A_4"],
        "144G_12": ["144F_2", "143B_4", "144A_2", "143A_3"],
        "144G_13": ["144F_4", "144F_1", "144A_4", "143B_1", "143A_2"],
        "144G_14": ["144A_2", "143A_3", "144F_8", "143B_1"],
        "144G_15": ["144F_2", "143B_2", "143B_4"],
        "144G_16": ["144A_3", "143B_2", "143A_1"],
        "144G_17": ["143A_4", "143B_2"],
        "144G_18": ["143A_1", "143A_2", "143B_1"],
        "144L_1": ["144A_5", "143A_1"],
        "144L_2": ["144A_6", "143A_1"],
        "144L_3": ["144A_7", "143A_1"],
        "144L_4": ["144A_8", "143A_2"],
        "144L_5": ["144A_7", "143A_3"],
        "144L_6": ["144A_8", "143A_4"],
        "144L_7": ["144A_6", "143A_1"],
        "144L_8": ["144A_7", "143A_1"],
        "144L_9": ["144A_8", "143A_2"],
        "144L_10": ["144A_8", "143A_4"],
        "144L_11": ["144A_7", "143A_3"],
        "144L_12": ["144A_6", "143A_1"],
        "144L_13": ["144A_8", "143A_2"],
        "144L_14": ["143A_1", "143A_1"],
        "144L_15": ["144A_16"],
        "144L_16": ["144A_18"],
        "144L_17": ["144A_19"],
        "144L_18": ["143A_1"],
        "144L_19": ["143A_2"],
        "144L_20": ["143A_3"],
        "144L_21": ["143A_4"],
        "144L_22": ["144B_3", "143A_1"],
        "144L_23": ["144B_1"],
        "144M_1": ["144A_7", "144A_8", "143A_1", "143A_2"],
        "144M_2": ["144A_6", "144A_7", "143A_1"],
        "144M_3": ["144A_5", "144A_6", "144A_7", "143A_1"],
        "144M_4": ["144A_5", "144A_16", "143A_1"],
        "144M_5": ["144A_8", "143A_4", "143A_2"],
        "144M_6": ["144A_7", "144A_8", "143A_3", "143A_4"],
        "144M_7": ["144A_7", "143A_3", "143G_2"],
        "144M_8": ["144A_6", "144A_7", "144A_8", "143A_1", "143A_4"],
        "144M_9": ["144A_7", "144A_4", "143A_1", "143A_2"],
        "144M_10": ["144A_8", "144A_3", "143A_2"],
        "144M_11": ["144A_6", "144A_8", "144A_4", "143A_1", "143A_2"],
        "144M_12": ["144A_5", "144A_6", "144A_1", "144A_3", "143A_1", "143A_2"],
        "144M_13": ["144A_7", "144A_8", "144A_2", "144A_4", "143A_1", "143A_2"],
        "144M_14": ["144A_7", "144A_2", "143A_3"],
        "144M_15": ["144A_8", "144A_3", "143A_4"],
        "144M_16": ["144A_7", "144A_2", "143A_1", "143A_3"],
        "144M_17": ["144A_5", "144Y_8", "143A_1"],
        "144M_18": ["144A_8", "144A_14", "143A_2"],
        "144M_19": ["144A_8", "144A_14", "143A_4"],
        "144M_20": ["143A_1", "143A_2"],
        "144M_21": ["143A_1", "143A_3"],
        "144M_22": ["143A_1", "143A_4"],
        "144M_23": ["143A_1", "143A_2"],
        "144M_24": ["143A_1", "143A_2"],
        "144M_25": ["143A_1", "143A_4"],
        "144M_26": ["143A_3", "143A_4"],
        "144M_27": ["143A_2", "143A_4"],
        "144N_1": ["144N_1", "143B_1"],
        "144N_2": ["144N_2", "143B_7"],
        "144N_3": ["144N_3", "143B_2"],
        "144N_4": ["144N_1", "143B_2"],
        "144N_5": ["144N_1", "143B_3"],
        "144N_6": ["144N_1", "143B_4"],
        "144N_7": ["144A_7", "143A_3", "143A_5"],
        "144N_8": ["144N_1", "143B_1", "143B_2"],
        "144N_9": ["144N_1", "143B_2", "143B_4"],
        "144N_10": ["144N_10", "143B_2", "144N_1"],
        "144N_11": ["144N_11", "143B_9", "143A_4", "143G_2", "143A_3"],
        "144N_12": ["144N_10", "144F_1", "143B_1"],
        "144N_13": ["144N_10", "144A_4", "143A_5", "143B_1", "143A_2"],
        "144N_14": ["143B_1"],
        "144N_15": ["143B_2"],
        "144N_16": ["143B_8"],
        "144N_17": ["143B_2"],
        "144N_18": ["143B_4"],
        "144T_1": ["144A_9", "143A_1"],
        "144T_2": ["144A_10", "143A_2"],
        "144T_3": ["144A_11", "143A_2"],
        "144T_4": ["144A_12", "143A_1"],
        "144T_5": ["144A_9", "143A_1", "144A_12"],
        "144T_6": ["143A_1"],
        "144T_7": ["144A_9", "143A_1"],
        "144T_8": ["144A_9", "143A_3"],
        "144T_9": ["144A_12", "143A_3"],
        "144T_10": ["144T_10", "143A_4"],
        "144T_11": ["143A_4"],
        "144T_12": ["144A_9", "143A_1", "143A_3"],
        "144T_13": ["144A_12", "143A_1", "143A_3"],
        "144T_14": ["144A_9", "143A_1", "144A_2"],
        "144T_15": ["144A_11", "143A_2", "144A_4"],
        "144T_16": ["144A_12", "143A_1", "144A_1"],
        "144T_17": ["144A_12", "143A_1", "144A_2"],
        "144T_18": ["144A_12", "143A_3", "144A_2"],
        "144T_19": ["143A_1"],
        "144T_20": ["143A_1"],
        "144T_21": ["144A_17"],
        "144T_22": ["144A_18"],
        "144T_23": ["144A_19"],
        "144U_1": ["144U_1", "143B_1"],
        "144U_2": ["144U_2", "143B_1"],
        "144U_3": ["144U_3", "143B_1"],
        "144U_4": ["144U_4", "143B_7"],
        "144U_5": ["144U_5", "143B_9"],
        "144U_6": ["143B_4"],
        "144U_7": ["144F_1", "144F_3", "144F_21", "143B_1", "143B_9"],
        "144U_8": ["144U_5", "143B_9", "144A_11", "143B_9", "143A_2"],
        "144U_9": ["144A_9", "144U_4", "143B_7", "143A_1"],
        "144U_10": ["144A_9", "143A_1", "144U_1", "143B_4"],
        "144U_11": ["144A_9", "143A_1", "143G_3"],
        "144U_12": ["144A_9", "143A_3", "144U_3", "143B_3"],
        "144U_13": ["143A_1", "143B_8"],
        "144U_14": ["143A_1", "143A_4", "143B_4"],
        "144U_15": ["143B_1"],
        "144U_16": ["143B_1", "143A_1", "143A_2"],
        "144X_1": ["144B_6", "144A_18.0", "144A_19.0"],
        "144X_2": ["144B_6", "144A_19"],
        "144X_4": ["144B_6.0"],
        "144Y_1": ["144A_13", "143A_1"],
        "144Y_2": ["144A_14", "143A_2"],
        "144Y_3": ["144A_15", "143A_2"],
        "144Y_4": ["144A_14", "143A_4"],
        "144Y_5": ["144A_15", "144Y_5", "143A_1", "143A_4", "143B_2"],
        "144Y_6": ["144Y_6", "143B_4"],
        "144Y_7": ["144A_14", "144A_8", "143A_2"],
        "144Y_8": ["144A_5", "143A_1", "144Y_8"],
        "144Y_9": ["144A_14", "144A_8", "143A_4"],
        "144Y_10": ["144A_13", "144A_1", "143A_1", "143A_3"],
        "144Y_11": ["143A_1"],
    }

    for lang in langs:
        for param in mapping.keys():
            if df_final.loc[lang][param] == 1:
                for p in mapping[param]:
                    if "." in p:
                        df_final.loc[lang][p.split(".")[0]] = float("0." + p.split(".")[1])
                    else:
                        df_final.loc[lang][p] = 1.0

    to_drop = [k for k in list(mapping.keys()) if k not in [item for sublist in mapping.values() for item in sublist]]

    drop_feat = [feat for feat in df_final.columns if feat.startswith("144H")
                 or feat.startswith("144I") or feat.startswith("144J")
                 or feat.startswith("144K") or feat.startswith("144P")
                 or feat.startswith("144Q") or feat.startswith("144R")
                 or feat.startswith("144S") or feat.startswith("144V")
                 or feat.startswith("144W") or feat.startswith("144X_3")]

    df_final.drop(columns=to_drop + drop_feat, inplace=True)

    df_final.rename(columns={
        "144Y_8": "NegOVS",
        "144N_10": "SNegOVNeg",
        "144N_11": "SOVNegNeg",
        "144T_10": "VNegOS",
    })

    df_final.drop(columns=["144D_13", ], inplace=True)

    to_drop = []
    for column in list(df_langs.columns):
        if (not column.split("_")[0].endswith("A")) and (
                df_langs[column].isna().sum() == df_langs.shape[0]):
            to_drop.append(column)

    drop_feat = [feat for feat in df_final.columns if feat.startswith(tuple(to_drop))]
    df_final.drop(columns=drop_feat, inplace=True)

    to_drop = []
    for column in list(df_final.columns):
        if df_final[column].isna().sum() == df_final.shape[0]:
            to_drop.append(column)
    df_diff = df_final.drop(columns=to_drop, inplace=False)
    df_diff = df_diff.fillna(0).astype("float32")

    if kind == "simple":
        return df_diff, langs

    else:
        df_diff.drop(columns=[
            "26A_1", "33A_6", "46A_4", "51A_9", "53A_8", "69A_5", "70A_5", "71A_2",
            "72A_4", "82A_3", "83A_3", "85A_4", "86A_3", "87A_2", "88A_6", "92A_6", "95A_5",
            "96A_5", "97A_5", "112A_4", "116A_6", "143B_4",
        ], inplace=True)

        corr = df_diff.corr()
        corr_dict = {}
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            corr_dict[df_diff.columns[i]] = []
            for j in range(i + 1, corr.shape[0]):
                if np.absolute(corr.iloc[i, j]) > 0.99:
                    if columns[j]:
                        columns[j] = False
                        corr_dict[df_diff.columns[i]].append(df_diff.columns[j])
            if not corr_dict[df_diff.columns[i]]:
                del corr_dict[df_diff.columns[i]]
        selected_columns = df_diff.columns[columns]
        df_encode = df_diff[selected_columns]
        return df_encode, langs


def compute_typo_dist_simple(df_diff, langs):
    features = sorted(
        list(set([f.split("_")[0] for f in df_diff.columns])),
        key=lambda x: (int(x.strip("AB")),
                       re.search("[A-Z]", x).group(0))
    )
    typo_matrix = []
    for src in langs:
        row = []
        for tgt in langs:
            distances = []
            for feat in features:
                fs = [f for f in df_diff.columns if f.startswith(feat)]
                src_feat = df_diff[fs].loc[src].to_numpy(dtype="float")
                tgt_feat = df_diff[fs].loc[tgt].to_numpy(dtype="float")
                if np.count_nonzero(src_feat) == 1 and np.count_nonzero(tgt_feat) == 1:
                    dist = np.max(np.absolute(src_feat - tgt_feat))
                else:
                    denom = np.count_nonzero(np.in1d(src_feat.nonzero()[0], tgt_feat.nonzero()[0]))
                    if denom:
                        dist = 1 / (2 * denom) * np.max(np.absolute(src_feat - tgt_feat))
                    else:
                        dist = np.max(np.absolute(src_feat - tgt_feat))
                distances.append(dist)
            row.append(np.array(distances))
        typo_matrix.append(np.array(row))
    return np.array(typo_matrix)



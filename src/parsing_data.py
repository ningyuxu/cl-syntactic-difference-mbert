import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import constant

metric2order = {
    "pos_acc": 11,
    "confusion_matrix": 10,
    "uas": 11,
    "las": 11,
    "uem": 11,
    "lem": 11,
    "las_confmat": 10,
}

axlabel2type = {
    "pos": constant.UD_POS_LABELS,
    "parsing": constant.UD_HEAD_LABELS,
}


def get_lang(acc_label):
    lang_list = acc_label[4:-4].split('-')
    if lang_list[0] == "Chinese":
        if "Simp" in lang_list[1]:
            lang = "-".join((lang_list[0], lang_list[1][3:7]))
        else:
            lang = lang_list[0]
    else:
        lang = lang_list[0]
    return lang


def get_metric(data_path, metric2order=metric2order, metric="pos_acc"):
    records = dict()
    for p in os.listdir(data_path):
        if "version" in p:
            result_file = data_path + '/' + p + "/results.json"
            with open(result_file, mode='r', encoding="utf-8") as f:
                rows = f.readlines()
                lang = get_lang(list(json.loads(rows[0]).keys())[0])
                record = dict()
                for k, v in json.loads(rows[metric2order[metric]]).items():
                    if f"_{metric}" in k and k != f"tst_{metric}":
                        k = get_lang(k)
                        record[k] = v
                records[lang] = record
        else:
            continue
    return records


def plot_accuracy(accs):
    langs = []
    for lang in list(constant.LANGUAGE_TO_ISO639.keys()):
        langs.append(lang.split('-')[0])

    matrix = []
    for source in langs:
        acc = accs.get(source)
        row = []
        for target in langs:
            row.append(acc.get(target))
        matrix.append(np.array(row))
    matrix = np.array(matrix)
    # matrix = matrix - matrix.transpose()

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        matrix,
        xticklabels=langs,
        yticklabels=langs,
        cmap='coolwarm',
        annot=True,
        linewidth=0.5,
    )
    # ax.set_xticks(np.arange(len(langs)))
    # ax.set_yticks(np.arange(len(langs)))
    # ax.set_xticklabels(langs)
    # ax.set_yticklabels(langs)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    ax.set_title("Cross Lingual Zero Shot Accuracy")
    plt.show()
    return


def plot_confmat(src_lang, tgt_lang, confmats, normalize="gold", axlabel="pos"):
    assert axlabel in axlabel2type
    conf_mtr = confmats[src_lang][tgt_lang]
    conf_mtr = np.stack(conf_mtr)
    assert conf_mtr.shape[0] == conf_mtr.shape[1], "shape of confusion matrix not supported"
    if not normalize:
        conf_mtr_norm = conf_mtr
    elif normalize == "pred":
        conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum(axis=0, keepdims=True)
    elif normalize == "all":
        conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum()
    else:
        assert normalize == "gold", "normalize should be in [None, 'all', 'gold', 'pred']"
        conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum(axis=1, keepdims=True)
    conf_mtr_norm = np.nan_to_num(conf_mtr_norm)
    conf_mtr_round = np.round(conf_mtr_norm, 2)

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        conf_mtr_round * 100,
        xticklabels=axlabel2type[axlabel],
        yticklabels=axlabel2type[axlabel],
        cmap="YlGnBu",
        annot=True,
        linewidth=0.5,
        fmt="g",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    ax.set_title(f"Confusion Matrix ({src_lang} to {tgt_lang})")
    plt.show()
    return


def plot_certain_relation(confmats, normalize="gold", relation="nsubj"):
    langs = []
    for lang in list(constant.LANGUAGE_TO_ISO639.keys()):
        langs.append(lang.split('-')[0])
    label2id = {label: idx for idx, label in enumerate(constant.UD_HEAD_LABELS)}
    matrix = []
    for source in langs:
        row = []
        for target in langs:
            conf_mtr = confmats[source][target]
            conf_mtr = np.stack(conf_mtr)
            assert conf_mtr.shape[0] == conf_mtr.shape[1], "shape of confusion matrix not supported"
            if not normalize:
                conf_mtr_norm = conf_mtr
            elif normalize == "pred":
                conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum(axis=0, keepdims=True)
            elif normalize == "all":
                conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum()
            else:
                assert normalize == "gold", "normalize should be in [None, 'all', 'gold', 'pred']"
                conf_mtr_norm = conf_mtr.astype('float') / conf_mtr.sum(axis=1, keepdims=True)
            conf_mtr_norm = np.nan_to_num(conf_mtr_norm)
            conf_mtr_round = np.round(conf_mtr_norm, 2)
            row.append(conf_mtr_round[label2id[relation]][label2id[relation]])
        matrix.append(np.array(row))
    matrix = np.array(matrix)

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        matrix,
        xticklabels=langs,
        yticklabels=langs,
        cmap='cubehelix',  # 'RdYlBu', 'mako', 'magma', 'Spectral', 'PuBuGn_r'
        annot=True,
        linewidth=0.5,
        fmt="g",
        vmin=0.0,
        vmax=1.0,
    )
    # ax.set_xticks(np.arange(len(langs)))
    # ax.set_yticks(np.arange(len(langs)))
    # ax.set_xticklabels(langs)
    # ax.set_yticklabels(langs)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    ax.set_title(f"Cross-Lingual Zero-Shot Precision {relation}")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--figure_type", default="las_confmat", type=str)
    parser.add_argument("--src_lang", default=None, type=str)
    parser.add_argument("--tgt_lang", default=None, type=str)
    parser.add_argument("--relation", default="compound", type=str)
    parser.add_argument("--file_path", default="ud_parsing", type=str)  # ud-parsing-with-postags

    params = parser.parse_args()
    data_path = params.file_path

    records = get_metric(data_path, metric=params.figure_type)

    if params.figure_type == "confusion_matrix":
        assert params.src_lang and params.tgt_lang, "src_lang and tgt_lang are required."
        plot_confmat(params.src_lang, params.tgt_lang, confmats=records, axlabel="pos")
    elif params.figure_type == "las_confmat":
        if params.relation:
            assert params.relation in constant.UD_HEAD_LABELS, "relation type is required."
            plot_certain_relation(confmats=records, relation=params.relation)
        else:
            assert params.src_lang and params.tgt_lang, "src_lang and tgt_lang are required."
            plot_confmat(params.src_lang, params.tgt_lang, confmats=records, axlabel="parsing")
    else:
        plot_accuracy(records)

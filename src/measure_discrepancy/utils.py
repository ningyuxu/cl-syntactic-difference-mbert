import json
import os
import torch


REPORT_PATH = "./experiment/discrepancies/new"


def write_dist(dist_dict, dist_kind, src_lang_iso, out_path=REPORT_PATH, model="fine-tuned", layer=7):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if model == "pretrain":
        file_path = os.path.join(out_path, f"{dist_kind}_pretrain_{src_lang_iso}_{layer}.json")
    elif model == "random":
        file_path = os.path.join(out_path, f"{dist_kind}_random_{src_lang_iso}_{layer}.json")
    elif model == "test":
        file_path = os.path.join(out_path, f"{dist_kind}_pretrain_{layer}_test.json")
    else:
        file_path = os.path.join(out_path, f"{dist_kind}_{src_lang_iso}.json")
    with open(file_path, mode='a', encoding="utf-8") as f:
        logs = dict()
        for k, v in dist_dict.items():
            logs[k] = v
        print(json.dumps(logs), file=f)


def compute_depth(idx, head_ids):
    head = idx
    length = 0
    while True:
        head = head_ids[head]
        if head == -1:
            return 1000
        if head != 0:
            length += 1
        else:
            return length


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

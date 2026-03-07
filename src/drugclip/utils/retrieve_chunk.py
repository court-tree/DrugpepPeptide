import lmdb
import pickle
from tqdm import tqdm
from glob import glob
import os
from itertools import islice
import pandas as pd
import numpy as np
from multiprocessing import Pool
from argparse import ArgumentParser


N_PROCS = 64
INPUT_FILES = "output/merge*.pkl"
OUTPUT_DIR = "retrieval_results"

parser = ArgumentParser()
parser.add_argument("--input_files", type=str, default=INPUT_FILES)
parser.add_argument("--mol_lmdb", type=str, nargs="+", required=True)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--num_threads", type=int, default=N_PROCS)
args = parser.parse_args()

N_PROCS = args.num_threads
INPUT_FILES = args.input_files
OUTPUT_DIR = args.output_dir
lmdb_list = args.mol_lmdb

os.makedirs(OUTPUT_DIR, exist_ok=True)


def batched(iterable, n=1):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def read_one(i):
    data = []
    env = lmdb.open(i, readonly=True, subdir=False)
    with env.begin() as txn:
        for k, v in txn.cursor():
            v = pickle.loads(v)
            data.append((v["smiles"], v["oid"]))
    env.close()
    return np.array(data)

print("loading data into ram...")
with Pool(len(lmdb_list)) as p:
    data_list = np.concatenate(p.map(read_one, lmdb_list))

offset = [0]
for i in lmdb_list:
    env = lmdb.open(i, readonly=True, subdir=False)
    with env.begin() as txn:
        offset.append(txn.stat()["entries"])
offset = np.cumsum(offset)

def process_one(inputs):
    for pn, scr, ind, fi in zip(*inputs):
        ind = ind.numpy() + offset[fi.numpy()]
        hits = np.concatenate([data_list[ind], scr.numpy().reshape(-1, 1)], axis=1)
        df = pd.DataFrame(hits, columns=["smiles", "oid", "score"])
        df = df.sort_values("score", ascending=False)
        df.to_csv(
            f"{OUTPUT_DIR}/{pn}.csv",
            index_label="Name",
        )

print("data loaded, start retrieving...")
for file in sorted(glob(INPUT_FILES)):
    with open(file, "rb") as f:
        pocket_names, scores, m_inds, f_inds = pickle.load(f)
    print(f"batch size: {scores[0].shape[0]}")
    with Pool(N_PROCS) as p:
        for i in tqdm(
            p.imap_unordered(
                process_one,
                zip(
                    batched(pocket_names, scores[0].shape[0]),
                    scores,
                    m_inds,
                    f_inds,
                ),
            ),
            desc=f"iter batch in {file}",
            total=len(scores),
        ):
            pass

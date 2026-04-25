import pandas as pd
import requests
import os
import time

# === 路径设置 ===
biolip_file = r"E:\pep\BioLiP_nr.txt"
save_dir = r"E:\pep\dow\BioLiP_download"

os.makedirs(save_dir, exist_ok=True)

colnames = [
    "pdb_id", "receptor_chain", "resolution", "binding_site_id",
    "ligand_id", "ligand_chain", "ligand_serial",
    "binding_residues", "binding_residues_2",
    "catalytic_residues", "catalytic_residues_2",
    "ec_number", "go_terms",
    "binding_affinity_manual", "binding_affinity_moad",
    "binding_affinity_pdbbind", "binding_affinity_bindingdb",
    "uniprot_id", "pubmed_id", "ligand_resseq", "receptor_seq"
]

df = pd.read_csv(
    biolip_file,
    sep="\t",
    header=None,
    names=colnames,
    low_memory=False
)

# 筛选 peptide
pep = df[df["ligand_id"].str.lower() == "peptide"]

print("peptide rows:", len(pep))

pdb_ids = sorted(pep["pdb_id"].str.lower().unique())
print("unique pdb:", len(pdb_ids))

failed = []

for i, pdb in enumerate(pdb_ids, 1):

    out_path = os.path.join(save_dir, f"{pdb}.pdb")

    if os.path.exists(out_path):
        print(f"[{i}/{len(pdb_ids)}] skip {pdb}")
        continue

    url = f"https://files.rcsb.org/download/{pdb.upper()}.pdb"

    try:
        r = requests.get(url, timeout=30)

        if r.status_code == 200 and len(r.text) > 1000:
            with open(out_path, "w") as f:
                f.write(r.text)

            print(f"[{i}/{len(pdb_ids)}] downloaded {pdb}")

        else:
            print(f"[{i}/{len(pdb_ids)}] failed {pdb}")
            failed.append(pdb)

    except Exception as e:
        print(f"[{i}/{len(pdb_ids)}] error {pdb}")
        failed.append(pdb)

    time.sleep(0.1)

print("done")
print("failed:", len(failed))
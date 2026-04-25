import json
from pathlib import Path
import gemmi

step3_path = Path("/mnt/e/pep/project/runs/smoke_server/step3/step3_candidates.jsonl")
cif_path = Path("/mnt/e/pep/download/1adu.cif")
out_path = Path("/mnt/e/pep/project/trace_1adu_manual/anchor_positions_1adu.tsv")

target = ("1adu", "A", "B")

anchors = {}
with step3_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        key = (row.get("pdb_id"), row.get("receptor_chain_id"), row.get("peptide_source_chain_id"))
        if key != target:
            continue
        rank = int(row["method_a_anchor_rank"])
        if rank not in anchors:
            anchors[rank] = row

st = gemmi.read_structure(str(cif_path))
model = st[0]

AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","MSE","PHE","PRO",
    "SER","THR","TRP","TYR","VAL"
}

def get_chain(chain_name):
    for chain in model:
        if str(chain.name).strip() == chain_name:
            return chain
    raise KeyError(chain_name)

def protein_residues(chain):
    keep = []
    for res in chain:
        if res.name in AA:
            keep.append(res)
    return keep

rec_chain = get_chain("A")
pep_chain = get_chain("B")

rec_res = protein_residues(rec_chain)
pep_res = protein_residues(pep_chain)

with out_path.open("w", encoding="utf-8") as out:
    out.write("anchor_rank\treceptor_chain\treceptor_resseq\treceptor_resname\tpeptide_chain\tpeptide_resseq\tpeptide_resname\tanchor_min_distance\n")
    for rank in sorted(anchors):
        row = anchors[rank]
        rec_idx = int(row["anchor_receptor_res_index"])
        pep_idx = int(row["anchor_peptide_res_index"])
        rr = rec_res[rec_idx]
        pr = pep_res[pep_idx]
        out.write(
            f"{rank}\tA\t{rr.seqid.num}\t{rr.name}\tB\t{pr.seqid.num}\t{pr.name}\t{row['anchor_min_distance']:.3f}\n"
        )

print(f"[DONE] wrote {out_path}")

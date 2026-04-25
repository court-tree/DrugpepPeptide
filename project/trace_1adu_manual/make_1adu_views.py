import json
from pathlib import Path

base = Path("/mnt/e/pep/project/runs/smoke_server")
out = Path("/mnt/e/pep/project/trace_1adu_manual")
out.mkdir(parents=True, exist_ok=True)

TARGET = ("1adu", "A", "B")
STEP3_IDS = {
    "9b068a64-7b47-429c-9867-3101fe33ce41",
    "46eb77f2-c9f8-4973-bfc3-691ff992ccab",
    "67e523bd-969f-4874-b27d-3e8509e961d4",
    "ac9058e5-5cc4-45ae-b822-a8f288a9fcb7",
    "7f998ca0-baad-446c-903a-2de2d777bb42",
    "27ab16f8-f6c8-4619-b1c2-b2e73e3bc5f7",
}
STEP5_IDS = {
    "c1636ab1-6ed9-443e-8546-f7dfcfeaac0e",
    "c35bc048-794f-4c17-b71a-0cfdf4c32cf6",
    "c66f8ab3-b5da-459f-948e-da365d4d7049",
    "a0e6d583-ce9f-4bae-a285-6caa2e57cb41",
    "349a3da2-8577-4987-ad13-a9c07dee618d",
}
FINAL_ID = "c1636ab1-6ed9-443e-8546-f7dfcfeaac0e"

def filt(src, dst, keep_ids=None):
    n = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            key = (row.get("pdb_id"), row.get("receptor_chain_id"), row.get("peptide_source_chain_id"))
            if key != TARGET:
                continue
            if keep_ids is not None and row.get("candidate_id") not in keep_ids:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(dst, n)

filt(base / "step3" / "step3_candidates.jsonl", out / "step3_anchor1_six.jsonl", STEP3_IDS)
filt(base / "step5" / "step5_final.jsonl", out / "step5_kept5.jsonl", STEP5_IDS)
filt(base / "step6_align" / "step6_survived.jsonl", out / "step6_kept.jsonl", {FINAL_ID})
filt(base / "step7" / "step7_main.jsonl", out / "step7_kept.jsonl", {FINAL_ID})
filt(base / "step8" / "lmdb" / "final_metadata.jsonl", out / "step8_kept.jsonl", {FINAL_ID})

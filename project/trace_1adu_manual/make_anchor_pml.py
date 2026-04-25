from pathlib import Path

tsv = Path("/mnt/e/pep/project/trace_1adu_manual/anchor_positions_1adu.tsv")
out = Path("/mnt/e/pep/project/trace_1adu_manual/step3_anchor_positions.pml")
png = Path("/mnt/e/pep/project/trace_1adu_manual/step3_anchor_positions.png")
cif = Path("/mnt/e/pep/download/1adu.cif")

colors = {
    "1": "red",
    "2": "marine",
    "3": "magenta",
    "4": "forest",
    "5": "orange",
}

lines = [
    f'load {cif.as_posix()}, complex',
    'hide everything, all',
    'bg_color white',
    'show cartoon, chain A',
    'color gray80, chain A',
    'show cartoon, chain B',
    'color wheat, chain B',
    'set cartoon_transparency, 0.25, chain A',
    'set cartoon_transparency, 0.45, chain B',
]

with tsv.open("r", encoding="utf-8") as f:
    next(f)
    for line in f:
        rank, rchain, rseq, rname, pchain, pseq, pname, dist = line.strip().split("\t")
        lines.append(f"select rec_a{rank}, chain {rchain} and resi {rseq}")
        lines.append(f"select pep_a{rank}, chain {pchain} and resi {pseq}")
        lines.append(f"show spheres, rec_a{rank} or pep_a{rank}")
        lines.append(f"color {colors[rank]}, rec_a{rank} or pep_a{rank}")
        lines.append(f'label rec_a{rank} and name CA, \"A{rank}\"')
        lines.append(f'label pep_a{rank} and name CA, \"A{rank}\"')

lines += [
    'set sphere_scale, 0.55',
    'set label_size, 18',
    'set label_color, black',
    'orient',
    'zoom visible, 6',
    'ray 2200,1600',
    f'png {png.as_posix()}, dpi=300',
]

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[DONE] wrote {out}")

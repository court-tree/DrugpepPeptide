from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import gemmi


STANDARD_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class BuildConfig:
    input_jsonl: Path
    structure_root: Path
    output_dir: Path
    min_peptide_length: int = 8
    max_peptide_length: int = 20
    contact_cutoff: float = 5.0
    context_cutoff: float = 10.0
    split_mode: str = "peptide_exact_sequence"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    random_seed: int = 20260630
    progress_every: int = 1000
    allowed_sources: Tuple[str, ...] = ("BioLiP_peptide", "Q-BioLiP_PIII", "PepBDB", "Propedia")
    allow_asymmetric_unit_fallback: bool = False
    receptor_family_map: Optional[Path] = None
    receptor_family_identity_threshold: float = 0.40
    receptor_family_min_coverage: float = 0.60
    manual_audit_sample_count: int = 24


@dataclass(frozen=True)
class ResidueItem:
    chain_id: str
    residue: gemmi.Residue
    seq_index: int


@dataclass(frozen=True)
class AtomItem:
    chain_id: str
    residue_id: str
    residue_index: int
    residue_name: str
    atom_name: str
    element: str
    coord: Tuple[float, float, float]


class Audit:
    def __init__(self) -> None:
        self.reject_counts: Counter[str] = Counter()
        self.qc_flag_counts: Counter[str] = Counter()
        self.input_candidate_count = 0
        self.deduplicated_candidate_count = 0
        self.dropped_duplicates = 0

    def reject(self, reason: str) -> None:
        self.reject_counts[reason] += 1
        self.qc_flag_counts[reason] += 1


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def stable_hash(*parts: Any, length: int = 16) -> str:
    text = "|".join(json.dumps(part, sort_keys=True, ensure_ascii=True) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def record_value(record: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def resolve_path(value: Any, root: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def resolve_structure_path(record: Dict[str, Any], root: Path) -> Path:
    for key in ("complex_structure_file", "structure_path", "source_file", "pdb_file", "cif_file"):
        path = resolve_path(record.get(key), root)
        if path is not None:
            return path
    pdb_id = str(record.get("pdb_id", "")).strip()
    if pdb_id:
        for suffix in (".cif", ".mmcif", ".pdb"):
            for name in (pdb_id, pdb_id.lower(), pdb_id.upper()):
                candidate = root / f"{name}{suffix}"
                if candidate.exists():
                    return candidate
    raise ValueError("missing_structure_path")


def load_model(path: Path) -> gemmi.Model:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if len(structure) == 0:
        raise ValueError("empty_structure")
    return structure[0]


def load_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if len(structure) == 0:
        raise ValueError("empty_structure")
    return structure


def source_provided_complex(record: Dict[str, Any]) -> bool:
    assembly_id = str(record_value(record, "biological_assembly_id", "assembly_id", default="")).strip()
    confidence = str(record_value(record, "assembly_confidence", default="")).strip()
    return assembly_id in {
        "biolip_source_pdb",
        "pepbdb_curated_complex",
        "propedia_curated_complex",
    } or confidence in {
        "biolip_binding_site_record",
        "pepbdb_curated_complex",
        "propedia_curated_complex",
    }


def load_complex_model_with_assembly(
    structure_path: Path,
    record: Dict[str, Any],
    config: BuildConfig,
) -> Tuple[gemmi.Model, Dict[str, Any]]:
    structure = load_structure(structure_path)
    assembly_id = str(record_value(record, "biological_assembly_id", "assembly_id", default="")).strip()
    status: Dict[str, Any] = {
        "assembly_id_requested": assembly_id,
        "assembly_policy": "unknown",
        "assembly_status": "unknown",
        "assembly_available_names": [assembly.name for assembly in structure.assemblies],
    }
    if source_provided_complex(record):
        status["assembly_policy"] = "source_provided_complex"
        status["assembly_status"] = "accepted_source_provided"
        return structure[0], status
    if assembly_id and structure.assemblies:
        names = {assembly.name for assembly in structure.assemblies}
        if assembly_id not in names:
            if config.allow_asymmetric_unit_fallback:
                status["assembly_policy"] = "asymmetric_unit_fallback"
                status["assembly_status"] = "requested_assembly_not_found"
                return structure[0], status
            raise ValueError(f"requested_assembly_not_found:{assembly_id}")
        structure.transform_to_assembly(assembly_id, gemmi.HowToNameCopiedChain.AddNumber)
        if len(structure) == 0:
            raise ValueError("empty_assembly_model")
        status["assembly_policy"] = "gemmi_transform_to_assembly"
        status["assembly_status"] = "reconstructed"
        return structure[0], status
    if config.allow_asymmetric_unit_fallback:
        status["assembly_policy"] = "asymmetric_unit_fallback"
        status["assembly_status"] = "no_assembly_metadata"
        return structure[0], status
    raise ValueError("no_biological_assembly_metadata")


def find_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    target = str(chain_id).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    raise KeyError("missing_chain")


def residue_num(residue: gemmi.Residue) -> int:
    if residue.seqid.num is None:
        raise ValueError("missing_residue_number")
    return int(residue.seqid.num)


def residue_icode(residue: gemmi.Residue) -> str:
    return str(residue.seqid.icode).strip()


def residue_id(chain_id: str, residue: gemmi.Residue) -> str:
    icode = residue_icode(residue)
    number = residue_num(residue)
    if icode:
        return f"{chain_id}:{number}{icode}:{residue.name.strip().upper()}"
    return f"{chain_id}:{number}:{residue.name.strip().upper()}"


def is_standard_residue(residue: gemmi.Residue) -> bool:
    return residue.name.strip().upper() in STANDARD_AA


def chain_residues(chain: gemmi.Chain) -> List[ResidueItem]:
    out: List[ResidueItem] = []
    for residue in chain:
        if is_standard_residue(residue):
            out.append(ResidueItem(str(chain.name), residue, len(out)))
    return out


def select_segment(items: Sequence[ResidueItem], start: Any, end: Any) -> List[ResidueItem]:
    if start in (None, "") and end in (None, ""):
        return list(items)
    if start in (None, "") or end in (None, ""):
        raise ValueError("incomplete_peptide_residue_range")
    start_i, end_i = sorted((int(start), int(end)))
    selected = [item for item in items if start_i <= residue_num(item.residue) <= end_i]
    if not selected:
        raise ValueError("empty_peptide_segment")
    return selected


def residue_sequence(items: Sequence[ResidueItem]) -> str:
    return "".join(STANDARD_AA.get(item.residue.name.strip().upper(), "X") for item in items)


def has_backbone(residue: gemmi.Residue) -> bool:
    present = {atom.name.strip().upper() for atom in residue}
    return BACKBONE_ATOMS.issubset(present)


def validate_peptide_continuity(items: Sequence[ResidueItem], start: Any, end: Any) -> Tuple[bool, str]:
    if not items:
        return False, "empty_peptide_segment"
    if any(residue_icode(item.residue) for item in items):
        return False, "unsupported_insertion_code"
    numbers = [residue_num(item.residue) for item in items]
    if any(right != left + 1 for left, right in zip(numbers, numbers[1:])):
        return False, "non_contiguous_peptide_segment"
    if start not in (None, "") and end not in (None, ""):
        start_i, end_i = sorted((int(start), int(end)))
        if numbers[0] != start_i or numbers[-1] != end_i:
            return False, "peptide_range_boundary_mismatch"
        if len(numbers) != end_i - start_i + 1:
            return False, "residue_range_length_mismatch"
    return True, ""


def validate_peptide(items: Sequence[ResidueItem], min_len: int, max_len: int) -> Tuple[bool, str]:
    n = len(items)
    if n < min_len or n > max_len:
        return False, "peptide_length_out_of_range"
    seq = residue_sequence(items)
    if len(seq) != n or any(ch not in "ACDEFGHIKLMNPQRSTVWY" for ch in seq):
        return False, "noncanonical_sequence"
    if any(not has_backbone(item.residue) for item in items):
        return False, "missing_backbone_atoms"
    if not heavy_atom_items(items):
        return False, "missing_heavy_atoms"
    return True, ""


def assign_length_group(length: int) -> str:
    if 8 <= length <= 10:
        return "short_8_10"
    if 11 <= length <= 15:
        return "medium_short_11_15"
    if 16 <= length <= 20:
        return "medium_long_16_20"
    raise ValueError(f"unsupported_length:{length}")


def contact_thresholds(length_group: str) -> Dict[str, int]:
    if length_group == "short_8_10":
        return {"contact_count": 4, "interface_residue_count": 3}
    if length_group == "medium_short_11_15":
        return {"contact_count": 5, "interface_residue_count": 4}
    if length_group == "medium_long_16_20":
        return {"contact_count": 8, "interface_residue_count": 5}
    raise ValueError(f"unknown_length_group:{length_group}")


def heavy_atom_items(items: Sequence[ResidueItem], backbone_only: bool = False) -> List[AtomItem]:
    atoms: List[AtomItem] = []
    for item in items:
        rid = residue_id(item.chain_id, item.residue)
        for atom in item.residue:
            atom_name = atom.name.strip()
            if atom.element.name == "H":
                continue
            if backbone_only and atom_name.upper() not in BACKBONE_ATOMS:
                continue
            atoms.append(
                AtomItem(
                    chain_id=item.chain_id,
                    residue_id=rid,
                    residue_index=item.seq_index,
                    residue_name=item.residue.name.strip().upper(),
                    atom_name=atom_name,
                    element=atom.element.name.strip().upper(),
                    coord=(float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)),
                )
            )
    return atoms


def atom_records(atoms: Sequence[AtomItem]) -> List[Dict[str, Any]]:
    return [
        {
            "atom_index": index,
            "atom_name": atom.atom_name,
            "element": atom.element,
            "residue_id": atom.residue_id,
            "residue_index": atom.residue_index,
            "residue_name": atom.residue_name,
            "x": atom.coord[0],
            "y": atom.coord[1],
            "z": atom.coord[2],
        }
        for index, atom in enumerate(atoms)
    ]


def residue_records(items: Sequence[ResidueItem]) -> List[Dict[str, Any]]:
    return [
        {
            "chain_id": item.chain_id,
            "residue_id": residue_id(item.chain_id, item.residue),
            "residue_index": item.seq_index,
            "residue_name": item.residue.name.strip().upper(),
            "one_letter": STANDARD_AA[item.residue.name.strip().upper()],
            "seq_number": residue_num(item.residue),
            "insertion_code": residue_icode(item.residue),
            "has_backbone": has_backbone(item.residue),
        }
        for item in items
    ]


def squared_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def contact_statistics(
    receptor_atoms: Sequence[AtomItem],
    peptide_atoms: Sequence[AtomItem],
    cutoff: float,
) -> Dict[str, Any]:
    if not receptor_atoms or not peptide_atoms:
        return {
            "min_heavy_atom_distance": None,
            "contact_count": 0,
            "interface_residue_count": 0,
            "peptide_contact_positions": [],
            "peptide_contact_residue_ids": [],
            "receptor_contact_residue_ids": [],
        }
    cutoff_sq = cutoff * cutoff
    min_sq: Optional[float] = None
    contact_count = 0
    peptide_residue_ids = set()
    peptide_positions = set()
    receptor_residue_ids = set()
    for receptor_atom in receptor_atoms:
        for peptide_atom in peptide_atoms:
            dist_sq = squared_distance(receptor_atom.coord, peptide_atom.coord)
            if min_sq is None or dist_sq < min_sq:
                min_sq = dist_sq
            if dist_sq <= cutoff_sq:
                contact_count += 1
                peptide_residue_ids.add(peptide_atom.residue_id)
                peptide_positions.add(peptide_atom.residue_index)
                receptor_residue_ids.add(receptor_atom.residue_id)
    return {
        "min_heavy_atom_distance": (min_sq ** 0.5) if min_sq is not None else None,
        "contact_count": int(contact_count),
        "interface_residue_count": int(len(receptor_residue_ids)),
        "peptide_contact_positions": sorted(peptide_positions),
        "peptide_contact_residue_ids": sorted(peptide_residue_ids),
        "receptor_contact_residue_ids": sorted(receptor_residue_ids),
    }


def atom_contact_pairs(
    receptor_atoms: Sequence[AtomItem],
    peptide_atoms: Sequence[AtomItem],
    cutoff: float,
) -> List[Dict[str, Any]]:
    cutoff_sq = cutoff * cutoff
    pairs: List[Dict[str, Any]] = []
    for receptor_index, receptor_atom in enumerate(receptor_atoms):
        for peptide_index, peptide_atom in enumerate(peptide_atoms):
            dist_sq = squared_distance(receptor_atom.coord, peptide_atom.coord)
            if dist_sq <= cutoff_sq:
                pairs.append(
                    {
                        "receptor_atom_index": receptor_index,
                        "peptide_atom_index": peptide_index,
                        "receptor_residue_id": receptor_atom.residue_id,
                        "peptide_residue_id": peptide_atom.residue_id,
                        "distance": dist_sq ** 0.5,
                    }
                )
    return pairs


def extract_residues_within(
    receptor_items: Sequence[ResidueItem],
    peptide_atoms: Sequence[AtomItem],
    cutoff: float,
) -> List[ResidueItem]:
    receptor_atoms = heavy_atom_items(receptor_items)
    cutoff_sq = cutoff * cutoff
    selected_ids = set()
    for receptor_atom in receptor_atoms:
        for peptide_atom in peptide_atoms:
            if squared_distance(receptor_atom.coord, peptide_atom.coord) <= cutoff_sq:
                selected_ids.add(receptor_atom.residue_id)
                break
    return [item for item in receptor_items if residue_id(item.chain_id, item.residue) in selected_ids]


def deduplicate_records(records: Sequence[Dict[str, Any]], audit: Audit) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    source_dbs: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    source_ids: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for record in records:
        key = (
            str(record_value(record, "pdb_id", default="")),
            str(record_value(record, "biological_assembly_id", "assembly_id", default="")),
            str(record_value(record, "receptor_chain_id", "receptor_chain", default="")),
            str(record_value(record, "peptide_chain_id", "peptide_chain", default="")),
            str(record_value(record, "peptide_residue_start", "peptide_start", default="")),
            str(record_value(record, "peptide_residue_end", "peptide_end", default="")),
        )
        source_db = str(record_value(record, "source_database", "source_db", default="unknown"))
        source_id = str(record_value(record, "source_entry_id", default=""))
        source_dbs[key].append(source_db)
        if source_id:
            source_ids[key].append(source_id)
        if key not in by_key:
            by_key[key] = dict(record)
        else:
            audit.dropped_duplicates += 1
    merged = []
    for key, record in by_key.items():
        record["source_dbs"] = sorted(set(source_dbs[key]))
        record["source_entry_ids"] = sorted(set(source_ids[key]))
        merged.append(record)
    return merged


def make_anchor_id(record: Dict[str, Any], peptide_sequence: str, interface_residue_ids: Sequence[str]) -> str:
    return "anc_" + stable_hash(
        record_value(record, "pdb_id", default=""),
        record_value(record, "biological_assembly_id", "assembly_id", default=""),
        record_value(record, "receptor_chain_id", "receptor_chain", default=""),
        record_value(record, "peptide_chain_id", "peptide_chain", default=""),
        record_value(record, "peptide_residue_start", "peptide_start", default=""),
        record_value(record, "peptide_residue_end", "peptide_end", default=""),
        peptide_sequence,
        list(interface_residue_ids),
    )


def build_anchor(record: Dict[str, Any], config: BuildConfig, audit: Audit) -> Optional[Dict[str, Any]]:
    source = str(record_value(record, "source_database", "source_db", default="unknown"))
    if source not in set(config.allowed_sources):
        audit.reject(f"source_not_allowed:{source}")
        return None
    try:
        assembly_info: Dict[str, Any]
        receptor_structure_path = resolve_path(record_value(record, "receptor_structure_file", "receptor_structure_path"), config.structure_root)
        peptide_structure_path = resolve_path(record_value(record, "peptide_structure_file", "ligand_structure_file", "peptide_structure_path"), config.structure_root)
        if receptor_structure_path is not None and peptide_structure_path is not None:
            if not receptor_structure_path.exists() or not peptide_structure_path.exists():
                audit.reject("missing_structure")
                return None
            receptor_model = load_model(receptor_structure_path)
            peptide_model = load_model(peptide_structure_path)
            structure_path = receptor_structure_path
            peptide_source_structure_path = peptide_structure_path
            assembly_info = {
                "assembly_id_requested": str(record_value(record, "biological_assembly_id", "assembly_id", default="")).strip(),
                "assembly_policy": "source_provided_separate_receptor_peptide",
                "assembly_status": "accepted_source_provided",
                "assembly_available_names": [],
            }
        else:
            structure_path = resolve_structure_path(record, config.structure_root)
            if not structure_path.exists():
                audit.reject("missing_structure")
                return None
            model, assembly_info = load_complex_model_with_assembly(structure_path, record, config)
            receptor_model = model
            peptide_model = model
            peptide_source_structure_path = structure_path
        receptor_chain_id = str(record_value(record, "receptor_chain_id", "receptor_chain", default="")).strip()
        peptide_chain_id = str(record_value(record, "peptide_chain_id", "peptide_chain", default="")).strip()
        receptor_chain = find_chain(receptor_model, receptor_chain_id)
        peptide_chain = find_chain(peptide_model, peptide_chain_id)
        receptor_items = chain_residues(receptor_chain)
        peptide_items_all = chain_residues(peptide_chain)
        peptide_start = record_value(record, "peptide_residue_start", "peptide_start")
        peptide_end = record_value(record, "peptide_residue_end", "peptide_end")
        peptide_items = select_segment(peptide_items_all, peptide_start, peptide_end)
    except Exception as exc:
        audit.reject(str(exc) or exc.__class__.__name__)
        return None

    ok, reason = validate_peptide_continuity(peptide_items, peptide_start, peptide_end)
    if not ok:
        audit.reject(reason)
        return None
    ok, reason = validate_peptide(peptide_items, config.min_peptide_length, config.max_peptide_length)
    if not ok:
        audit.reject(reason)
        return None

    peptide_sequence = residue_sequence(peptide_items)
    receptor_sequence = residue_sequence(receptor_items)
    length_group = assign_length_group(len(peptide_sequence))
    receptor_atoms = heavy_atom_items(receptor_items)
    peptide_atoms = heavy_atom_items(peptide_items)
    stats = contact_statistics(receptor_atoms, peptide_atoms, config.contact_cutoff)
    min_dist = stats["min_heavy_atom_distance"]
    if min_dist is None or min_dist > config.contact_cutoff:
        audit.reject("no_heavy_atom_contact")
        return None
    thresholds = contact_thresholds(length_group)
    if stats["contact_count"] < thresholds["contact_count"]:
        audit.reject("low_contact_count")
        return None
    if stats["interface_residue_count"] < thresholds["interface_residue_count"]:
        audit.reject("low_interface_residue_count")
        return None

    interface_items = extract_residues_within(receptor_items, peptide_atoms, config.contact_cutoff)
    context_items = extract_residues_within(receptor_items, peptide_atoms, config.context_cutoff)
    if not interface_items:
        audit.reject("empty_interface")
        return None
    if not context_items:
        audit.reject("empty_receptor_context")
        return None
    if any(not has_backbone(item.residue) for item in context_items):
        audit.reject("receptor_patch_quality")
        return None

    backbone_atoms = heavy_atom_items(peptide_items, backbone_only=True)
    if len(backbone_atoms) < len(peptide_items) * len(BACKBONE_ATOMS):
        audit.reject("true_bound_conformer_quality")
        return None
    heavy_atoms = peptide_atoms
    if not heavy_atoms:
        audit.reject("true_bound_conformer_quality")
        return None

    interface_residue_ids = [residue_id(item.chain_id, item.residue) for item in interface_items]
    context_residue_ids = [residue_id(item.chain_id, item.residue) for item in context_items]
    peptide_residue_ids = [residue_id(item.chain_id, item.residue) for item in peptide_items]
    receptor_context_atoms = heavy_atom_items(context_items)
    receptor_interface_atoms = heavy_atom_items(interface_items)
    receptor_context_backbone_atoms = heavy_atom_items(context_items, backbone_only=True)
    if not receptor_context_atoms or len(receptor_context_backbone_atoms) < len(context_items) * len(BACKBONE_ATOMS):
        audit.reject("receptor_patch_quality")
        return None
    receptor_peptide_contact_pairs_5A = atom_contact_pairs(receptor_context_atoms, peptide_atoms, config.contact_cutoff)
    if not receptor_peptide_contact_pairs_5A:
        audit.reject("receptor_patch_quality")
        return None
    pdb_id = str(record_value(record, "pdb_id", default="")).strip()
    assembly_id = str(record_value(record, "biological_assembly_id", "assembly_id", default="")).strip()
    anchor_id = make_anchor_id(record, peptide_sequence, interface_residue_ids)
    interface_id = "ifc_" + stable_hash(anchor_id, interface_residue_ids)
    conformer_id = "tbc_" + stable_hash(anchor_id, peptide_sequence, peptide_chain_id, peptide_start, peptide_end)
    sample_id = "smp_" + stable_hash(anchor_id, conformer_id)
    peptide_sequence_key = stable_hash("peptide_sequence", peptide_sequence)
    receptor_sequence_key = stable_hash("receptor_sequence", receptor_sequence)
    pdb_key = stable_hash("pdb", pdb_id)

    return {
        "sample_id": sample_id,
        "anchor_id": anchor_id,
        "source_dbs": record.get("source_dbs") or [record_value(record, "source_database", "source_db", default="unknown")],
        "source_entry_ids": record.get("source_entry_ids") or [record_value(record, "source_entry_id", default="")],
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "structure_path": str(structure_path),
        "receptor_structure_path": str(structure_path),
        "peptide_structure_path": str(peptide_source_structure_path),
        "assembly_policy": assembly_info["assembly_policy"],
        "assembly_status": assembly_info["assembly_status"],
        "assembly_available_names": assembly_info["assembly_available_names"],
        "receptor_chain_id": receptor_chain_id,
        "peptide_chain_id": peptide_chain_id,
        "peptide_residue_start": peptide_start,
        "peptide_residue_end": peptide_end,
        "peptide_sequence": peptide_sequence,
        "peptide_length": len(peptide_sequence),
        "peptide_residue_ids": peptide_residue_ids,
        "length_group": length_group,
        "receptor_sequence": receptor_sequence,
        "interface_id": interface_id,
        "true_bound_conformer_id": conformer_id,
        "interface_residues_5A": interface_residue_ids,
        "context_residues_10A": context_residue_ids,
        "receptor_context_residues": residue_records(context_items),
        "receptor_interface_residues": residue_records(interface_items),
        "receptor_context_atoms": atom_records(receptor_context_atoms),
        "receptor_interface_atoms": atom_records(receptor_interface_atoms),
        "receptor_context_backbone_atoms": atom_records(receptor_context_backbone_atoms),
        "receptor_peptide_contact_pairs_5A": receptor_peptide_contact_pairs_5A,
        "receptor_patch_sequence": "".join(STANDARD_AA[item.residue.name.strip().upper()] for item in context_items),
        "receptor_patch_seq_indices": [int(item.seq_index) for item in context_items],
        "contact_count": stats["contact_count"],
        "interface_residue_count": stats["interface_residue_count"],
        "min_heavy_atom_distance": stats["min_heavy_atom_distance"],
        "peptide_contact_positions": stats["peptide_contact_positions"],
        "peptide_contact_residue_ids": stats["peptide_contact_residue_ids"],
        "receptor_contact_residue_ids": stats["receptor_contact_residue_ids"],
        "backbone_atoms": atom_records(backbone_atoms),
        "heavy_atoms": atom_records(heavy_atoms),
        "peptide_sequence_key": peptide_sequence_key,
        "receptor_sequence_key": receptor_sequence_key,
        "receptor_family_key": receptor_sequence_key,
        "receptor_family_method": "unassigned_exact_sequence_placeholder",
        "pdb_key": pdb_key,
        "qc_status": "pass",
        "qc_flags": [],
    }


def split_key(anchor: Dict[str, Any], mode: str) -> str:
    if mode == "pair_level":
        return anchor["anchor_id"]
    if mode == "peptide_exact_sequence":
        return anchor["peptide_sequence_key"]
    if mode == "receptor_family":
        return anchor["receptor_family_key"]
    if mode == "strict":
        return "|".join([anchor["peptide_sequence_key"], anchor["receptor_family_key"], anchor["pdb_key"]])
    raise ValueError(f"unsupported_split_mode:{mode}")


def sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())


def sequence_coverage(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def load_receptor_family_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict):
            raise ValueError("receptor_family_map JSON must be an object")
        return {str(k): str(v) for k, v in obj.items()}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                key = row.get("receptor_sequence_key") or row.get("sequence_key") or row.get("receptor_key")
                family = row.get("receptor_family_key") or row.get("family_key") or row.get("cluster_id")
            else:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"{path}:{line_no}: expected at least two columns")
                key, family = parts[0], parts[1]
            if key and family:
                mapping[str(key)] = str(family)
    return mapping


def assign_receptor_family_keys(anchors: List[Dict[str, Any]], config: BuildConfig) -> Dict[str, Any]:
    if not anchors:
        return {"receptor_family_method": "none", "receptor_family_count": 0}

    if config.receptor_family_map is not None:
        mapping = load_receptor_family_map(config.receptor_family_map)
        missing = 0
        for anchor in anchors:
            exact_key = anchor["receptor_sequence_key"]
            family = mapping.get(exact_key)
            if family is None:
                missing += 1
                family = "rfam_exact_" + exact_key
            anchor["receptor_family_key"] = family
            anchor["receptor_family_method"] = "external_map"
        return {
            "receptor_family_method": "external_map",
            "receptor_family_count": len({a["receptor_family_key"] for a in anchors}),
            "receptor_family_map": str(config.receptor_family_map),
            "receptor_family_map_missing_exact_keys": missing,
        }

    seq_by_key: Dict[str, str] = {}
    for anchor in anchors:
        seq_by_key.setdefault(anchor["receptor_sequence_key"], anchor["receptor_sequence"])

    representatives: List[Tuple[str, str, str]] = []
    family_by_exact: Dict[str, str] = {}
    for exact_key, sequence in sorted(seq_by_key.items(), key=lambda item: stable_hash("rfam_order", item[0])):
        assigned_family: Optional[str] = None
        for family_key, _rep_key, rep_sequence in representatives:
            if sequence_coverage(sequence, rep_sequence) < config.receptor_family_min_coverage:
                continue
            if sequence_similarity(sequence, rep_sequence) >= config.receptor_family_identity_threshold:
                assigned_family = family_key
                break
        if assigned_family is None:
            assigned_family = "rfam_" + stable_hash(
                "receptor_family",
                exact_key,
                config.receptor_family_identity_threshold,
                config.receptor_family_min_coverage,
            )
            representatives.append((assigned_family, exact_key, sequence))
        family_by_exact[exact_key] = assigned_family

    for anchor in anchors:
        anchor["receptor_family_key"] = family_by_exact[anchor["receptor_sequence_key"]]
        anchor["receptor_family_method"] = "sequence_similarity_greedy"

    return {
        "receptor_family_method": "sequence_similarity_greedy",
        "receptor_family_count": len(set(family_by_exact.values())),
        "unique_receptor_sequence_count": len(seq_by_key),
        "receptor_family_identity_threshold": config.receptor_family_identity_threshold,
        "receptor_family_min_coverage": config.receptor_family_min_coverage,
    }


def assign_splits(anchors: List[Dict[str, Any]], config: BuildConfig) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if config.split_mode == "strict":
        uf = UnionFind(len(anchors))
        first_by_token: Dict[str, int] = {}
        for idx, anchor in enumerate(anchors):
            tokens = [
                f"pep:{anchor['peptide_sequence_key']}",
                f"rfam:{anchor['receptor_family_key']}",
                f"pdb:{anchor['pdb_key']}",
            ]
            for token in tokens:
                first = first_by_token.get(token)
                if first is None:
                    first_by_token[token] = idx
                else:
                    uf.union(first, idx)
        index_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for idx, anchor in enumerate(anchors):
            index_groups[uf.find(idx)].append(anchor)
        groups = {str(root): rows for root, rows in index_groups.items()}
    else:
        for anchor in anchors:
            groups[split_key(anchor, config.split_mode)].append(anchor)

    total = len(anchors)
    train_target = int(round(total * config.train_fraction))
    val_target = int(round(total * config.val_fraction))
    test_target = max(0, total - train_target - val_target)
    targets = {"train": train_target, "val": val_target, "test": test_target}

    if config.split_mode == "strict" and total > 0:
        group_entries = list(groups.items())
        source_totals: Counter[str] = Counter()
        for anchor in anchors:
            source_totals.update(str(source) for source in anchor.get("source_dbs", []))
        sources = sorted(source_totals)
        group_sources: Dict[str, Counter[str]] = {}
        for group_key, rows in group_entries:
            counts: Counter[str] = Counter()
            for row in rows:
                counts.update(str(source) for source in row.get("source_dbs", []))
            group_sources[group_key] = counts

        assigned_groups: set[str] = set()
        split_by_group: Dict[str, str] = {}
        max_eval_target = max(val_target, test_target, 1)
        max_eval_group_size = max(1, int(max_eval_target * 0.25))

        def choose_eval_groups(split_name: str) -> None:
            target_n = targets[split_name]
            if target_n <= 0:
                return
            target_sources = {
                source: source_totals[source] * target_n / max(total, 1)
                for source in sources
            }
            count = 0
            source_counts: Counter[str] = Counter()
            while count < target_n * 0.98:
                available = [
                    item for item in group_entries
                    if item[0] not in assigned_groups
                ]
                if not available:
                    break
                small_available = [
                    item for item in available
                    if len(item[1]) <= max_eval_group_size
                ]
                candidates = small_available or available

                def score(item: Tuple[str, List[Dict[str, Any]]]) -> Tuple[float, str]:
                    group_key, rows = item
                    group_size = len(rows)
                    next_count = count + group_size
                    next_sources = source_counts + group_sources[group_key]
                    size_error = ((next_count - target_n) / max(target_n, 1)) ** 2
                    if next_count > target_n:
                        size_error *= 5.0
                    balance_error = sum(
                        ((next_sources[source] - target_sources[source]) / max(target_sources[source], 1.0)) ** 2
                        for source in sources
                    )
                    deficit_gain = sum(
                        max(0.0, target_sources[source] - source_counts[source])
                        - max(0.0, target_sources[source] - next_sources[source])
                        for source in sources
                    ) / max(group_size, 1)
                    new_source_count = sum(
                        1 for source in sources
                        if source_counts[source] == 0 and group_sources[group_key][source] > 0
                    )
                    return (
                        balance_error + size_error - 3.0 * deficit_gain - 0.1 * new_source_count + group_size * 1.0e-5,
                        stable_hash(config.random_seed, split_name, group_key),
                    )

                group_key, rows = min(candidates, key=score)
                assigned_groups.add(group_key)
                split_by_group[group_key] = split_name
                count += len(rows)
                source_counts.update(group_sources[group_key])

        choose_eval_groups("test")
        choose_eval_groups("val")
        for group_key, _rows in group_entries:
            if group_key not in split_by_group:
                split_by_group[group_key] = "train"

        for group_key, rows in group_entries:
            split = split_by_group[group_key]
            split_group = "grp_" + stable_hash(config.split_mode, group_key)
            for row in rows:
                row["split_group"] = split_group
                row["split"] = split
        return

    counts = {"train": 0, "val": 0, "test": 0}
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            stable_hash(config.random_seed, item[0], [row["anchor_id"] for row in item[1]]),
        ),
    )
    positive_target_splits = [name for name in SPLIT_ORDER if targets[name] > 0]
    for group_index, (group_key, rows) in enumerate(sorted_groups):
        group_size = len(rows)
        empty_required = [
            name
            for name in positive_target_splits
            if counts[name] == 0 and name != "train"
        ]
        remaining_after_this = len(sorted_groups) - group_index - 1
        if counts["train"] > 0 and empty_required and remaining_after_this >= len(empty_required) - 1:
            split = empty_required[0]
        else:
            split = min(
                SPLIT_ORDER,
                key=lambda name: (
                    ((counts[name] + group_size - targets[name]) / max(targets[name], 1)) ** 2,
                    counts[name] / max(targets[name], 1),
                    stable_hash(config.random_seed, group_key, name),
                ),
            )
        split_group = "grp_" + stable_hash(config.split_mode, group_key)
        for row in rows:
            row["split_group"] = split_group
            row["split"] = split
        counts[split] += len(rows)

def leakage_checks(anchors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def cross_split_count(key: str) -> int:
        splits_by_key: Dict[str, set] = defaultdict(set)
        for anchor in anchors:
            splits_by_key[str(anchor[key])].add(anchor["split"])
        return sum(1 for splits in splits_by_key.values() if len(splits) > 1)

    return {
        "same_peptide_sequence_cross_split_count": cross_split_count("peptide_sequence_key"),
        "same_pdb_cross_split_count": cross_split_count("pdb_key"),
        "same_receptor_family_cross_split_count": cross_split_count("receptor_family_key"),
    }


def pml_quote(path: str) -> str:
    return str(path).replace("\\", "/").replace('"', '\\"')


def pml_resi_selector(residue_ids: Sequence[str]) -> str:
    values: List[str] = []
    for rid in residue_ids:
        parts = str(rid).split(":")
        if len(parts) >= 2 and parts[1]:
            values.append(parts[1])
    if not values:
        return "none"
    return "+".join(sorted(set(values), key=lambda value: (len(value), value)))


def pml_script_for_anchor(anchor: Dict[str, Any]) -> str:
    receptor_path = pml_quote(str(anchor["receptor_structure_path"]))
    peptide_path = pml_quote(str(anchor["peptide_structure_path"]))
    receptor_chain = str(anchor["receptor_chain_id"])
    peptide_chain = str(anchor["peptide_chain_id"])
    interface_resi = pml_resi_selector(anchor["interface_residues_5A"])
    context_resi = pml_resi_selector(anchor["context_residues_10A"])
    peptide_resi = pml_resi_selector(anchor["peptide_residue_ids"])
    same_file = Path(anchor["receptor_structure_path"]) == Path(anchor["peptide_structure_path"])

    lines = [
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, off",
        "set cartoon_transparency, 0.25",
        "set surface_quality, 1",
        "set dash_gap, 0.28",
        "set dash_radius, 0.045",
        "set dash_round_ends, on",
    ]
    if same_file:
        lines.append(f'load "{receptor_path}", complex')
        receptor_obj = "complex"
        peptide_obj = "complex"
    else:
        lines.append(f'load "{receptor_path}", receptor_source')
        lines.append(f'load "{peptide_path}", peptide_source')
        receptor_obj = "receptor_source"
        peptide_obj = "peptide_source"
    lines.extend(
        [
            "hide everything",
            f"select receptor_chain, {receptor_obj} and chain {receptor_chain}",
            f"select peptide_chain, {peptide_obj} and chain {peptide_chain} and resi {peptide_resi}",
            f"select context10, receptor_chain and resi {context_resi}",
            f"select interface5, receptor_chain and resi {interface_resi}",
            "show cartoon, receptor_chain",
            "color gray70, receptor_chain",
            "show sticks, peptide_chain",
            "color magenta, peptide_chain",
            "set stick_radius, 0.22, peptide_chain",
            "show sticks, interface5",
            "color cyan, interface5",
            "set stick_radius, 0.16, interface5",
            "show surface, context10",
            "set transparency, 0.82, context10",
            "distance contacts_core_3p8A, interface5, peptide_chain, 3.8",
            "hide labels, contacts_core_3p8A",
            "color orange, contacts_core_3p8A",
            "set dash_radius, 0.055, contacts_core_3p8A",
            "distance contacts5A_all_hidden, interface5, peptide_chain, 5.0",
            "hide labels, contacts5A_all_hidden",
            "color yellow, contacts5A_all_hidden",
            "disable contacts5A_all_hidden",
            "zoom peptide_chain or interface5, 10",
            "orient peptide_chain or interface5",
        ]
    )
    return "\n".join(lines) + "\n"


def select_manual_audit_anchors(anchors: Sequence[Dict[str, Any]], sample_count: int) -> List[Dict[str, Any]]:
    if sample_count <= 0 or not anchors:
        return []
    strata: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        source = str((anchor.get("source_dbs") or ["unknown"])[0])
        strata[(source, str(anchor["length_group"]), str(anchor["split"]))].append(anchor)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    ordered_strata = sorted(strata.items(), key=lambda item: stable_hash("manual_audit_stratum", item[0]))
    while ordered_strata and len(selected) < sample_count:
        progressed = False
        for key, rows in ordered_strata:
            ordered_rows = sorted(rows, key=lambda row: stable_hash("manual_audit_anchor", row["anchor_id"]))
            for row in ordered_rows:
                if row["anchor_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["anchor_id"])
                    progressed = True
                    break
            if len(selected) >= sample_count:
                break
        if not progressed:
            break
    return selected


def export_manual_structure_audit(anchors: Sequence[Dict[str, Any]], config: BuildConfig) -> Dict[str, Any]:
    audit_dir = config.output_dir / "manual_structure_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    samples = select_manual_audit_anchors(anchors, config.manual_audit_sample_count)
    sample_rows: List[Dict[str, Any]] = []
    for index, anchor in enumerate(samples, start=1):
        case_id = f"case_{index:03d}_{anchor['anchor_id']}"
        pml_path = audit_dir / f"{case_id}.pml"
        pml_path.write_text(pml_script_for_anchor(anchor), encoding="utf-8")
        sample_rows.append(
            {
                "case_id": case_id,
                "anchor_id": anchor["anchor_id"],
                "sample_id": anchor["sample_id"],
                "source_dbs": anchor["source_dbs"],
                "pdb_id": anchor["pdb_id"],
                "assembly_id": anchor["assembly_id"],
                "assembly_policy": anchor["assembly_policy"],
                "split": anchor["split"],
                "length_group": anchor["length_group"],
                "peptide_sequence": anchor["peptide_sequence"],
                "peptide_length": anchor["peptide_length"],
                "receptor_chain_id": anchor["receptor_chain_id"],
                "peptide_chain_id": anchor["peptide_chain_id"],
                "interface_residue_count": anchor["interface_residue_count"],
                "context_residue_count": len(anchor["context_residues_10A"]),
                "contact_count": anchor["contact_count"],
                "min_heavy_atom_distance": anchor["min_heavy_atom_distance"],
                "receptor_context_atom_count": len(anchor["receptor_context_atoms"]),
                "peptide_atom_count": len(anchor["heavy_atoms"]),
                "receptor_structure_path": anchor["receptor_structure_path"],
                "peptide_structure_path": anchor["peptide_structure_path"],
                "pymol_script": str(pml_path),
                "human_decision": "",
                "human_notes": "",
            }
        )

    write_jsonl(audit_dir / "manual_structure_audit_samples.jsonl", sample_rows)
    write_csv(audit_dir / "manual_structure_audit_samples.csv", sample_rows)
    summary = {
        "sample_count_requested": config.manual_audit_sample_count,
        "sample_count_written": len(sample_rows),
        "sampling_method": "deterministic_balanced_by_source_length_group_split",
        "checklist": [
            "peptide chain is the intended 8-20 aa bound peptide",
            "5A interface residues visually contact the peptide",
            "10A context patch covers the local binding pocket without obvious unrelated bulk",
            "biological assembly/source-provided complex looks plausible",
            "no obvious chain swap, missing peptide atoms, or receptor-only artifact",
        ],
        "outputs": {
            "samples_jsonl": str(audit_dir / "manual_structure_audit_samples.jsonl"),
            "samples_csv": str(audit_dir / "manual_structure_audit_samples.csv"),
            "pymol_scripts_dir": str(audit_dir),
        },
    }
    readme = "\n".join(
        [
            "# Manual Structure Audit",
            "",
            "This folder contains deterministic human-review samples for Phase-3 V1.",
            "",
            "Open any `.pml` file in PyMOL. The script colors receptor chain gray, peptide magenta, 5A interface residues cyan, and 10A context surface transparent.",
            "",
            "For readability, only close core contacts <= 3.8A are shown by default as orange distance dashes. The full <= 5.0A contact object is generated as `contacts5A_all_hidden` but disabled; enable it manually only when you need to inspect all atom-pair contacts.",
            "",
            "Recommended decision values for `human_decision`:",
            "",
            "```text",
            "pass",
            "fail_wrong_chain",
            "fail_no_real_contact",
            "fail_bad_assembly",
            "fail_missing_atoms",
            "uncertain",
            "```",
            "",
            "Record decisions in `manual_structure_audit_samples.csv` or a copied review sheet.",
            "",
        ]
    )
    (audit_dir / "README.md").write_text(readme, encoding="utf-8")
    write_json(audit_dir / "manual_structure_audit_summary.json", summary)
    return summary


def export_outputs(
    anchors: List[Dict[str, Any]],
    config: BuildConfig,
    audit: Audit,
    receptor_family_summary: Dict[str, Any],
) -> Dict[str, Any]:
    out = config.output_dir
    coords_dir = out / "coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    anchor_rows: List[Dict[str, Any]] = []
    interface_rows: List[Dict[str, Any]] = []
    conformer_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    track_a_rows: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    track_b_rows: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}

    for anchor in anchors:
        backbone_path = coords_dir / f"{anchor['true_bound_conformer_id']}.backbone.json"
        heavy_path = coords_dir / f"{anchor['true_bound_conformer_id']}.heavy.json"
        receptor_patch_path = coords_dir / f"{anchor['interface_id']}.receptor_patch.json"
        write_json(backbone_path, {"atoms": anchor["backbone_atoms"]})
        write_json(heavy_path, {"atoms": anchor["heavy_atoms"]})
        write_json(
            receptor_patch_path,
            {
                "interface_id": anchor["interface_id"],
                "anchor_id": anchor["anchor_id"],
                "pdb_id": anchor["pdb_id"],
                "assembly_id": anchor["assembly_id"],
                "receptor_chain_id": anchor["receptor_chain_id"],
                "contact_cutoff": config.contact_cutoff,
                "context_cutoff": config.context_cutoff,
                "interface_residues_5A": anchor["interface_residues_5A"],
                "context_residues_10A": anchor["context_residues_10A"],
                "context_residue_count": len(anchor["context_residues_10A"]),
                "interface_residue_count": len(anchor["interface_residues_5A"]),
                "receptor_context_residues": anchor["receptor_context_residues"],
                "receptor_interface_residues": anchor["receptor_interface_residues"],
                "receptor_context_atoms": anchor["receptor_context_atoms"],
                "receptor_interface_atoms": anchor["receptor_interface_atoms"],
                "receptor_context_backbone_atoms": anchor["receptor_context_backbone_atoms"],
                "receptor_peptide_contact_pairs_5A": anchor["receptor_peptide_contact_pairs_5A"],
            },
        )

        anchor_row = {
            key: anchor[key]
            for key in (
                "anchor_id",
                "source_dbs",
                "source_entry_ids",
                "pdb_id",
                "assembly_id",
                "receptor_chain_id",
                "peptide_chain_id",
                "peptide_residue_start",
                "peptide_residue_end",
                "peptide_sequence",
                "peptide_length",
                "length_group",
                "interface_id",
                "true_bound_conformer_id",
                "split_group",
                "split",
                "contact_count",
                "min_heavy_atom_distance",
                "interface_residue_count",
                "peptide_contact_positions",
                "peptide_sequence_key",
                "receptor_sequence",
                "receptor_sequence_key",
                "receptor_family_key",
                "receptor_family_method",
                "pdb_key",
                "qc_status",
                "qc_flags",
                "assembly_policy",
                "assembly_status",
            )
        }
        anchor_rows.append(anchor_row)

        interface_rows.append(
            {
                "interface_id": anchor["interface_id"],
                "anchor_id": anchor["anchor_id"],
                "pdb_id": anchor["pdb_id"],
                "assembly_id": anchor["assembly_id"],
                "receptor_chain_id": anchor["receptor_chain_id"],
                "interface_residues_5A": anchor["interface_residues_5A"],
                "context_residues_10A": anchor["context_residues_10A"],
                "context_residue_count": len(anchor["context_residues_10A"]),
                "interface_atom_count": len(anchor["receptor_interface_atoms"]),
                "context_atom_count": len(anchor["receptor_context_atoms"]),
                "contact_pair_count_5A": len(anchor["receptor_peptide_contact_pairs_5A"]),
                "receptor_patch_coords_path": str(receptor_patch_path),
                "coords_path": str(receptor_patch_path),
            }
        )

        conformer_rows.append(
            {
                "conformer_id": anchor["true_bound_conformer_id"],
                "anchor_id": anchor["anchor_id"],
                "peptide_sequence": anchor["peptide_sequence"],
                "peptide_length": anchor["peptide_length"],
                "length_group": anchor["length_group"],
                "pdb_id": anchor["pdb_id"],
                "chain_id": anchor["peptide_chain_id"],
                "start_residue": anchor["peptide_residue_start"],
                "end_residue": anchor["peptide_residue_end"],
                "source_anchor_id": anchor["anchor_id"],
                "source_split": anchor["split"],
                "is_true_bound": True,
                "backbone_coords_path": str(backbone_path),
                "heavy_atom_coords_path": str(heavy_path),
                "missing_ratio": 0.0,
                "qc_status": "pass",
                "qc_flags": [],
            }
        )

        edge = {
            "anchor_id": anchor["anchor_id"],
            "interface_id": anchor["interface_id"],
            "conformer_id": anchor["true_bound_conformer_id"],
            "edge_type": "positive_strong_bound",
            "edge_weight": 1.0,
            "loss_scope": ["sequence_branch", "conformer_branch", "fusion_branch"],
            "split": anchor["split"],
            "reason": "true receptor-peptide bound conformer",
        }
        edge_rows.append(edge)

        track_a = {
            "sample_id": anchor["sample_id"],
            "anchor_id": anchor["anchor_id"],
            "split": anchor["split"],
            "pdb_id": anchor["pdb_id"],
            "source_file": anchor["structure_path"],
            "receptor_chain_id": anchor["receptor_chain_id"],
            "peptide_source_chain_id": anchor["peptide_chain_id"],
            "receptor_sequence": anchor["receptor_sequence"],
            "receptor_patch_residue_ids": anchor["context_residues_10A"],
            "receptor_patch_seq_indices": anchor["receptor_patch_seq_indices"],
            "receptor_patch_sequence": anchor["receptor_patch_sequence"],
            "peptide_sequence": anchor["peptide_sequence"],
            "peptide_length": anchor["peptide_length"],
            "length_group": anchor["length_group"],
            "avg_contact_count": anchor["contact_count"] / max(anchor["peptide_length"], 1),
            "contact_coverage": len(anchor["peptide_contact_positions"]) / max(anchor["peptide_length"], 1),
            "receptor_key": "|".join([anchor["pdb_id"], anchor["receptor_chain_id"], str(anchor["context_residues_10A"])]),
            "peptide_key": anchor["peptide_sequence"],
            "peptide_sequence_id": anchor["peptide_sequence_key"],
            "receptor_family_30_id": anchor["receptor_family_key"],
            "receptor_interface_key": anchor["interface_id"],
            "edge_type": "positive_strong_bound",
        }
        track_b = {
            "sample_id": anchor["sample_id"],
            "anchor_id": anchor["anchor_id"],
            "split": anchor["split"],
            "pdb_id": anchor["pdb_id"],
            "source_file": anchor["structure_path"],
            "receptor_chain_id": anchor["receptor_chain_id"],
            "peptide_source_chain_id": anchor["peptide_chain_id"],
            "interface_id": anchor["interface_id"],
            "true_bound_conformer_id": anchor["true_bound_conformer_id"],
            "conformer_id": anchor["true_bound_conformer_id"],
            "receptor_coords_path": str(receptor_patch_path),
            "receptor_patch_coords_path": str(receptor_patch_path),
            "interface_residues_5A": anchor["interface_residues_5A"],
            "context_residues_10A": anchor["context_residues_10A"],
            "patch_residue_ids": anchor["context_residues_10A"],
            "peptide_residue_ids": anchor["peptide_residue_ids"],
            "patch_atoms": anchor["receptor_context_atoms"],
            "receptor_atoms": anchor["receptor_context_atoms"],
            "peptide_atoms": anchor["heavy_atoms"],
            "patch_cutoff": config.context_cutoff,
            "receptor_context_residue_count": len(anchor["context_residues_10A"]),
            "receptor_context_atom_count": len(anchor["receptor_context_atoms"]),
            "receptor_interface_atom_count": len(anchor["receptor_interface_atoms"]),
            "receptor_peptide_contact_pair_count_5A": len(anchor["receptor_peptide_contact_pairs_5A"]),
            "peptide_sequence": anchor["peptide_sequence"],
            "backbone_coords_path": str(backbone_path),
            "heavy_atom_coords_path": str(heavy_path),
            "receptor_key": "|".join([anchor["pdb_id"], anchor["receptor_chain_id"], str(anchor["context_residues_10A"])]),
            "peptide_key": "|".join([anchor["pdb_id"], anchor["peptide_chain_id"], str(anchor["peptide_residue_ids"])]),
            "peptide_sequence_id": anchor["peptide_sequence_key"],
            "receptor_family_30_id": anchor["receptor_family_key"],
            "receptor_interface_key": anchor["interface_id"],
            "edge_type": "positive_strong_bound",
        }
        track_a_rows[anchor["split"]].append(track_a)
        track_b_rows[anchor["split"]].append(track_b)

    write_csv(out / "receptor_peptide_anchor.csv", anchor_rows)
    write_jsonl(out / "receptor_peptide_anchor.jsonl", anchor_rows)
    write_jsonl(out / "receptor_interface.jsonl", interface_rows)
    write_jsonl(out / "peptide_true_bound_conformer.jsonl", conformer_rows)
    write_jsonl(out / "positive_strong_bound_edges.jsonl", edge_rows)
    for split in SPLIT_ORDER:
        write_jsonl(out / f"track_a_{split}.jsonl", track_a_rows[split])
        write_jsonl(out / f"track_b_{split}.jsonl", track_b_rows[split])
    manual_audit_summary = export_manual_structure_audit(anchors, config)

    split_counts = Counter(anchor["split"] for anchor in anchors)
    length_group_counts = Counter(anchor["length_group"] for anchor in anchors)
    assembly_policy_counts = Counter(anchor["assembly_policy"] for anchor in anchors)
    assembly_status_counts = Counter(anchor["assembly_status"] for anchor in anchors)
    checks = leakage_checks(anchors)
    output_paths = [
        out / "receptor_peptide_anchor.csv",
        out / "receptor_peptide_anchor.jsonl",
        out / "receptor_interface.jsonl",
        out / "peptide_true_bound_conformer.jsonl",
        out / "positive_strong_bound_edges.jsonl",
        out / "manual_structure_audit" / "manual_structure_audit_samples.jsonl",
        out / "manual_structure_audit" / "manual_structure_audit_samples.csv",
        out / "manual_structure_audit" / "manual_structure_audit_summary.json",
    ]
    output_hashes = {str(path): sha256_file(path) for path in output_paths if path.exists()}
    audit_obj = {
        "input_candidate_count": audit.input_candidate_count,
        "deduplicated_candidate_count": audit.deduplicated_candidate_count,
        "dropped_duplicates": audit.dropped_duplicates,
        "reject_counts": dict(sorted(audit.reject_counts.items())),
        "qc_flag_counts": dict(sorted(audit.qc_flag_counts.items())),
        "final_anchor_count": len(anchors),
        "split_counts": dict(split_counts),
        "length_group_counts": dict(length_group_counts),
        "assembly_policy_counts": dict(assembly_policy_counts),
        "assembly_status_counts": dict(assembly_status_counts),
        "unique_peptide_count": len({a["peptide_sequence_key"] for a in anchors}),
        "unique_receptor_count": len({a["receptor_family_key"] for a in anchors}),
        "unique_receptor_sequence_count": len({a["receptor_sequence_key"] for a in anchors}),
        "unique_pdb_count": len({a["pdb_key"] for a in anchors}),
        "leakage_checks": checks,
        "receptor_family_summary": receptor_family_summary,
        "parameter_snapshot": {
            "min_peptide_length": config.min_peptide_length,
            "max_peptide_length": config.max_peptide_length,
            "contact_cutoff": config.contact_cutoff,
            "context_cutoff": config.context_cutoff,
            "length_group_contact_thresholds": {
                group: contact_thresholds(group)
                for group in ("short_8_10", "medium_short_11_15", "medium_long_16_20")
            },
            "split_mode": config.split_mode,
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
            "random_seed": config.random_seed,
            "allowed_sources": list(config.allowed_sources),
            "allow_asymmetric_unit_fallback": config.allow_asymmetric_unit_fallback,
            "receptor_family_map": str(config.receptor_family_map) if config.receptor_family_map else None,
            "receptor_family_identity_threshold": config.receptor_family_identity_threshold,
            "receptor_family_min_coverage": config.receptor_family_min_coverage,
            "manual_audit_sample_count": config.manual_audit_sample_count,
            "source_policy": {
                "tier_1": ["BioLiP_peptide", "Q-BioLiP_PIII"],
                "tier_2": ["PepBDB", "Propedia"],
                "not_label_source": ["raw_PDB"],
                "coordinate_reservoir": ["PDB/mmCIF referenced by curated sources"],
            },
            "receptor_3d_contract": {
                "final_training_format": "phase2.pepclip.data.PepCLIP3DDataset_jsonl",
                "track_b_inline_training_fields": [
                    "patch_atoms",
                    "receptor_atoms",
                    "peptide_atoms",
                    "patch_residue_ids",
                    "peptide_residue_ids",
                    "receptor_key",
                    "peptide_key",
                ],
                "track_b_receptor_coords_path": "coords/<interface_id>.receptor_patch.json",
                "required_arrays": [
                    "receptor_context_atoms",
                    "receptor_context_backbone_atoms",
                    "receptor_context_residues",
                    "receptor_interface_atoms",
                    "receptor_peptide_contact_pairs_5A",
                ],
                "patch_context_cutoff": config.context_cutoff,
                "interface_cutoff": config.contact_cutoff,
                "reject_if_context_backbone_incomplete": True,
            },
            "track_a_training_contract": {
                "final_training_format": "phase2.pepclip.data.PepCLIPDataset_jsonl",
                "receptor_field": "receptor_patch_sequence",
                "peptide_field": "peptide_sequence",
                "required_fields": [
                    "receptor_sequence",
                    "receptor_patch_residue_ids",
                    "receptor_patch_seq_indices",
                    "receptor_patch_sequence",
                    "peptide_sequence",
                    "peptide_length",
                ],
            },
        },
        "input_file_hashes": {
            str(config.input_jsonl): sha256_file(config.input_jsonl)
        },
        "output_file_hashes": output_hashes,
        "manual_structure_audit": manual_audit_summary,
    }
    summary = {
        "version": "phase3_v1",
        "created_at_unix": int(time.time()),
        "input_jsonl": str(config.input_jsonl),
        "output_dir": str(config.output_dir),
        "final_anchor_count": len(anchors),
        "split_counts": dict(split_counts),
        "length_group_counts": dict(length_group_counts),
        "assembly_policy_counts": dict(assembly_policy_counts),
        "assembly_status_counts": dict(assembly_status_counts),
        "receptor_family_summary": receptor_family_summary,
        "manual_structure_audit": manual_audit_summary,
        "reject_counts": dict(sorted(audit.reject_counts.items())),
        "leakage_checks": checks,
    }
    write_json(out / "dataset_audit.json", audit_obj)
    write_json(out / "phase3_v1_summary.json", summary)
    return summary


def build_dataset(config: BuildConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    records = list(read_jsonl(config.input_jsonl))
    audit.input_candidate_count = len(records)
    deduped = deduplicate_records(records, audit)
    audit.deduplicated_candidate_count = len(deduped)
    anchors: List[Dict[str, Any]] = []
    for idx, record in enumerate(deduped, start=1):
        anchor = build_anchor(record, config, audit)
        if anchor is not None:
            anchors.append(anchor)
        if config.progress_every and idx % config.progress_every == 0:
            print(f"processed {idx}/{len(deduped)} records; kept={len(anchors)}")
    receptor_family_summary = assign_receptor_family_keys(anchors, config)
    assign_splits(anchors, config)
    return export_outputs(anchors, config, audit, receptor_family_summary)


def parse_args(argv: Optional[Sequence[str]] = None) -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build PepCLIP Phase-3 V1 fine-tuning dataset.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--structure_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_peptide_length", type=int, default=8)
    parser.add_argument("--max_peptide_length", type=int, default=20)
    parser.add_argument("--contact_cutoff", type=float, default=5.0)
    parser.add_argument("--context_cutoff", type=float, default=10.0)
    parser.add_argument("--split_mode", default="peptide_exact_sequence", choices=["pair_level", "peptide_exact_sequence", "receptor_family", "strict"])
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=20260630)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--receptor_family_map", default=None)
    parser.add_argument("--receptor_family_identity_threshold", type=float, default=0.40)
    parser.add_argument("--receptor_family_min_coverage", type=float, default=0.60)
    parser.add_argument("--manual_audit_sample_count", type=int, default=24)
    parser.add_argument(
        "--allowed_source",
        action="append",
        default=None,
        help="Allowed source_database value. Defaults to curated V1 sources.",
    )
    parser.add_argument(
        "--allow_asymmetric_unit_fallback",
        action="store_true",
        help="Allow first-model/asymmetric-unit fallback when no biological assembly metadata is available.",
    )
    args = parser.parse_args(argv)
    return BuildConfig(
        input_jsonl=Path(args.input_jsonl),
        structure_root=Path(args.structure_root),
        output_dir=Path(args.output_dir),
        min_peptide_length=args.min_peptide_length,
        max_peptide_length=args.max_peptide_length,
        contact_cutoff=args.contact_cutoff,
        context_cutoff=args.context_cutoff,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
        progress_every=args.progress_every,
        allowed_sources=tuple(args.allowed_source) if args.allowed_source else ("BioLiP_peptide", "Q-BioLiP_PIII", "PepBDB", "Propedia"),
        allow_asymmetric_unit_fallback=args.allow_asymmetric_unit_fallback,
        receptor_family_map=Path(args.receptor_family_map) if args.receptor_family_map else None,
        receptor_family_identity_threshold=args.receptor_family_identity_threshold,
        receptor_family_min_coverage=args.receptor_family_min_coverage,
        manual_audit_sample_count=args.manual_audit_sample_count,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    summary = build_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))




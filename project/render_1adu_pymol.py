from __future__ import annotations

from pathlib import Path

from pymol import cmd


OUT_DIR = Path(r"e:\pep\project\ppt_complex_trace_1adu")
STRUCTURE = Path(r"e:\pep\download\1adu.cif")


def setup_scene() -> None:
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("depth_cue", 0)
    cmd.hide("everything", "all")
    cmd.show("cartoon", "all")
    cmd.color("gray80", "chain A")
    cmd.color("wheat", "chain B")


def save_png(name: str, width: int = 1800, height: int = 1100) -> None:
    path = OUT_DIR / name
    cmd.viewport(width, height)
    cmd.ray(width, height)
    cmd.png(path.as_posix(), dpi=200)
    print(f"[PNG] {path.as_posix()}")


def scene_full_complex() -> None:
    cmd.reinitialize()
    cmd.load(STRUCTURE.as_posix(), "complex")
    setup_scene()
    cmd.set("cartoon_transparency", 0.35, "chain A")
    cmd.set("cartoon_transparency", 0.55, "chain B")
    cmd.show("sticks", "chain B and resi 228-236+378-387+420-431")
    cmd.color("marine", "chain B and resi 228-236")
    cmd.color("tv_orange", "chain B and resi 378-387")
    cmd.color("magenta", "chain B and resi 420-431")
    cmd.orient("chain A or chain B")
    cmd.zoom("chain A or chain B", 6)
    save_png("pymol_full_complex.png")


def scene_step3_hotspots() -> None:
    cmd.reinitialize()
    cmd.load(STRUCTURE.as_posix(), "complex")
    setup_scene()
    cmd.show("surface", "chain A within 6 of (chain B and resi 228-236+378-387+420-431)")
    cmd.set("transparency", 0.28, "chain A")
    cmd.show("sticks", "chain B and resi 228-236+378-387+420-431")
    cmd.color("marine", "chain B and resi 228-236")
    cmd.color("tv_orange", "chain B and resi 378-387")
    cmd.color("magenta", "chain B and resi 420-431")
    cmd.label("name CA and chain B and resi 228+378+420", "\"Step3 hotspot\"")
    cmd.orient("chain A within 8 of chain B and resi 228-236+378-387+420-431")
    cmd.zoom("chain A within 8 of chain B and resi 228-236+378-387+420-431", 3)
    save_png("pymol_step3_hotspots.png")


def scene_step5_windows() -> None:
    cmd.reinitialize()
    cmd.load(STRUCTURE.as_posix(), "complex")
    setup_scene()
    cmd.show("surface", "chain A within 6 of (chain B and resi 228-235+229-236+378-387+420-427+422-431)")
    cmd.set("transparency", 0.35, "chain A")
    cmd.show("sticks", "chain B and resi 228-235")
    cmd.show("sticks", "chain B and resi 229-236")
    cmd.show("sticks", "chain B and resi 378-387")
    cmd.show("sticks", "chain B and resi 420-427")
    cmd.show("sticks", "chain B and resi 422-431")
    cmd.color("deepteal", "chain B and resi 228-235")
    cmd.color("cyan", "chain B and resi 229-236")
    cmd.color("tv_orange", "chain B and resi 378-387")
    cmd.color("hotpink", "chain B and resi 420-427")
    cmd.color("purple", "chain B and resi 422-431")
    cmd.orient("chain A within 8 of chain B and resi 228-235+229-236+378-387+420-427+422-431")
    cmd.zoom("chain A within 8 of chain B and resi 228-235+229-236+378-387+420-427+422-431", 3)
    save_png("pymol_step5_windows.png")


def scene_step8_final() -> None:
    cmd.reinitialize()
    cmd.load(STRUCTURE.as_posix(), "complex")
    setup_scene()
    cmd.hide("cartoon", "chain B")
    cmd.show("surface", "chain A within 6 of (chain B and resi 378-387)")
    cmd.set("transparency", 0.22, "chain A")
    cmd.show("sticks", "chain A within 6 of (chain B and resi 378-387)")
    cmd.color("cyan", "chain A within 6 of (chain B and resi 378-387)")
    cmd.show("sticks", "chain B and resi 378-387")
    cmd.color("tv_orange", "chain B and resi 378-387")
    cmd.show("spheres", "chain B and resi 378+387")
    cmd.color("red", "chain B and resi 378+387")
    cmd.orient("chain A within 8 of chain B and resi 378-387")
    cmd.zoom("chain A within 8 of chain B and resi 378-387", 2.5)
    save_png("pymol_step8_final.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[STRUCTURE] {STRUCTURE.as_posix()} exists={STRUCTURE.exists()}")
    scene_full_complex()
    scene_step3_hotspots()
    scene_step5_windows()
    scene_step8_final()
    cmd.quit()


if __name__ == "__main__":
    main()

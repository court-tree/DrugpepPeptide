load /mnt/e/pep/download/1adu.cif, complex
hide everything, all
bg_color white
show cartoon, chain A
color gray80, chain A
show cartoon, chain B
color wheat, chain B
set cartoon_transparency, 0.25, chain A
set cartoon_transparency, 0.45, chain B
select rec_a1, chain A and resi 525
select pep_a1, chain B and resi 384
show spheres, rec_a1 or pep_a1
color red, rec_a1 or pep_a1
label rec_a1 and name CA, "A1"
label pep_a1 and name CA, "A1"
select rec_a2, chain A and resi 521
select pep_a2, chain B and resi 380
show spheres, rec_a2 or pep_a2
color marine, rec_a2 or pep_a2
label rec_a2 and name CA, "A2"
label pep_a2 and name CA, "A2"
select rec_a3, chain A and resi 526
select pep_a3, chain B and resi 484
show spheres, rec_a3 or pep_a3
color magenta, rec_a3 or pep_a3
label rec_a3 and name CA, "A3"
label pep_a3 and name CA, "A3"
select rec_a4, chain A and resi 514
select pep_a4, chain B and resi 233
show spheres, rec_a4 or pep_a4
color forest, rec_a4 or pep_a4
label rec_a4 and name CA, "A4"
label pep_a4 and name CA, "A4"
select rec_a5, chain A and resi 513
select pep_a5, chain B and resi 422
show spheres, rec_a5 or pep_a5
color orange, rec_a5 or pep_a5
label rec_a5 and name CA, "A5"
label pep_a5 and name CA, "A5"
set sphere_scale, 0.55
set label_size, 18
set label_color, black
orient
zoom visible, 6
ray 2200,1600
png /mnt/e/pep/project/trace_1adu_manual/step3_anchor_positions.png, dpi=300

#!/usr/bin/env python3
"""Assemble multiphase.wgsl : fusionne les 5 fichiers .wgsl et remplace chaque tableau séparé par une tranche d'UN SEUL buffer
de stockage `S` (indexé par des constantes O_<nom> générées par l'hôte, voir layout()). Résultat : 1 buffer de stockage + 1 uniforme,
donc aucune limite de nombre de buffers par étage, quel que soit le GPU."""
import re, sys
names = ["mass","alpha","eps","temp","prel","rho_mix","mu_cell","k_cell","solid","U","V","Us","Vs","Fx","Fy","w1","w2","w3","w4",
         "mfx","mfy","mtot_old","fl_phase","fl_mass","fl_en","tst","p_start","bnd_flux","sd","sae","san","cx","cr","cz","cp_","cap_","scal","part"]
src = "".join(open(f, encoding="utf-8").read() + "\n" for f in ["mp_common.wgsl","mp_sweep.wgsl","mp_force.wgsl","mp_phys.wgsl","mp_pressure.wgsl"])
# supprime les déclarations de bindings des tableaux
src = re.sub(r"@group\(\d\) @binding\(\d+\) var<storage, read_write> \w+: array<f32>;[^\n]*\n", "", src)
src = re.sub(r"@group\(\d\) @binding\(\d+\) var<storage, read_write> \w+: array<i32>;[^\n]*\n", "", src)
def repl_array(text, name):
    out = []; i = 0
    pat = re.compile(r"(?<![\w.])" + re.escape(name) + r"\[")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        j = m.end(); depth = 1
        while depth:
            ch = text[j]
            if ch == '[': depth += 1
            elif ch == ']': depth -= 1
            j += 1
        inner = text[m.end():j-1]
        out.append("S[O_%s + (%s)]" % (name, inner))
        i = j
    return "".join(out)
for n in names:
    src = repl_array(src, n)
head = "@group(0) @binding(1) var<storage, read_write> S: array<f32>;\n"
# insère après la déclaration de P
src = src.replace("@group(0) @binding(0) var<uniform> P: Params;\n", "@group(0) @binding(0) var<uniform> P: Params;\n" + head, 1)
open("multiphase.template.wgsl", "w", encoding="utf-8", newline="\n").write(src)
print("ok", len(src))
# Tier 1 tests, run on Modal

All three harnesses run on Modal. The two that are finished are reported here.
Neither of them docks, and the first one provably cannot: the container ships the
computed transitive import closure of `data` + `mogp_hadamard`, and `docking` is
not in it.

| test | harness | docks? | status |
|---|---|---|---|
| endpoint reproducibility | `modal_endpoint_reproducibility.py` | no | done, defect confirmed and fixed |
| receptor audit, rungs 1 and 2 | `modal_receptor_audit.py` | no | done, rung 1 GO, rung 2 blocked |
| rung-1 correlation pilot | `modal_rung1_correlation_pilot.py` | yes, on Modal only | running |

---

## Test 1. The nomination endpoint was not reproducible

### What the quantities are

**Jaccard index** is the size of the intersection of two sets divided by the size
of their union. Dimensionless, 0 to 1. Here the two sets are the top-20 molecules
nominated by two repeats of the *same* campaign cell, so 1.000 means the repeats
agree exactly and 0.818 means a repeat shares 18 of 22 distinct molecules with
another repeat of the same cell.

**Predicted selectivity index** is predicted hDHFR affinity minus predicted
PfDHFR affinity, in kcal/mol. Positive means parasite-selective. Its **spread**
below is max minus min across repeats of one cell, so it is the error bar that
the committed `*_pred_si_mean` column does not carry.

### The defect

`nominate_and_score.py` retrains a Hadamard ICM per (arm, seed) cell. The ICM's
task covariance is a GPyTorch `IndexKernel` whose `covar_factor` is initialised
from `torch.randn`, and nothing in that file called `torch.manual_seed`. Two runs
of the same cell therefore started from different task covariances.

### The measurement

12 cells (6 seeds x 2 arms), 5 repeats each, under both regimes. 120 GP fits.
The reimplementation was checked against the committed run: `n_train` and
`n_both` match `nominated_scored.csv` in all 12 cells, so the copy has not
drifted from the original.

| | unseeded (what was committed) | seeded (the fix) |
|---|---|---|
| mean pairwise Jaccard | 0.973 | **1.000** |
| worst cell Jaccard | 0.818 (`full` seed 1) | 1.000 |
| mean molecules stable in every repeat | 19.3 / 20 | **20 / 20** |
| mean predicted-SI spread | 0.024 kcal/mol | **0.000** |
| cells that returned a different top-20 | **6 of 12** | 0 of 12 |

### Which molecules swapped, and how much it matters

| arm | seed | molecules swapped | true SI of the swapped ones, kcal/mol |
|---|---|---|---|
| asym | 0 | 2 | +2.65, -0.16 |
| asym | 1 | 2 | -0.77 |
| asym | 3 | 2 | **+6.19** |
| asym | 4 | 2 | +1.92 |
| asym | 5 | 2 | +0.69 |
| full | 1 | 5 | -0.71, +0.77, +2.56 |

True SI is taken from the committed `nominated.csv` files, which are the
authoritative record. The docking cache is not: it was re-keyed after the
campaign and has lost 53.5% of the campaign's evaluated molecules.

**The mean endpoint is robust and the max endpoint is not.** The within-cell
spread of 0.024 kcal/mol is 5.6% of the mean `|full - asym|` true-SI gap of
0.423 kcal/mol, so `mean_SI` would survive a rerun. But `best_SI` is a maximum
over 20 molecules, and a maximum is not robust to a 3-molecule swap: the
molecule that `asym` seed 3 dropped on some repeats carries a true SI of +6.19
kcal/mol against a committed `best_SI` of 4.24 for that cell. That one RNG draw
moves a headline number by 46%.

### The fix, applied

`torch.manual_seed(seed)` before the `train_fn` call in
`nominate_and_score.py:nominate`. The same measurement confirms it is sufficient,
not merely plausible: every seeded repeat is bit-identical, so there is no second
randomness source on this path.

### What is NOT shown

Nothing here re-docks. The instability is in *which molecules get nominated*, and
that is settled by comparing SMILES sets. What those molecules score is a
different question, and answering it on Modal would be invalid anyway, because a
score from one machine may not be compared with a committed score from another.

---

## Test 2. Receptor audit: rung 1 is clear, rung 2 is not

### What the quantities are

**Resolution**, angstroms: the finest spacing of diffraction data the model is
fitted to. Lower is sharper. Above about 2.5 A the side-chain rotamers lining a
pocket are not reliably placed, and a docking box is positioned against exactly
those side chains.

**R-free**, dimensionless 0 to 1: the cross-validated disagreement between model
and data. Above about 0.28 at these resolutions means the model is loosely
restrained by the measurements.

**Chain breaks**: stretches of residues the crystallographer could not place. A
break *at* the pocket is disqualifying, because Vina will dock a ligand into the
hole where the missing loop should be.

### Validation first

The audit was run on the two receptors the project already uses, before being
trusted on the new ones. It recovers both correctly, NADPH included:

| pdb | role | res (A) | R-free | NADP(H) | organism matches |
|---|---|---|---|---|---|
| 1J3I | P. falciparum DHFR, in use | 2.33 | 0.236 | NDP | yes |
| 1U72 | human DHFR, in use | 1.90 | 0.213 | NDP | yes |

The proposed box centers it derives sit 3.6 A (1J3I) and 4.6 A (1U72) from the
committed ones, which is expected and not a discrepancy: the committed recipe
uses the centroid of the inhibitor's *folate-mimicking head*, and the audit uses
the centroid of the whole molecule including its flexible tail. Both sit well
inside the same 20 A box.

### Rung 1, Chagas disease: 3INV

| pdb | res (A) | R-free | modelled | % of sequence | breaks | at site | NADP(H) | verdict |
|---|---|---|---|---|---|---|---|---|
| **3INV** | **2.37** | **0.243** | 514 + 513 | 99% | **0** | 0 | **NDP** | **use this one** |
| 3CL9 | 3.30 | 0.249 | 505 | 97% | 1 | 0 | NAP | reject |
| 1E7W (L. major) | 1.75 | 0.244 | 267 + 258 | 92% | 5 | 0 | NDP | fallback only |

3INV is a better structure than the PfDHFR receptor this project already
published on: same resolution class, zero chain breaks against 1J3I's two, and
NADPH present. Its 514-residue chain confirms **both domains of the bifunctional
DHFR-TS are resolved**, which is the free third task and the selectivity handle
outside the active site.

3CL9 is rejected on two grounds, and the second is the one that matters: 3.30 A
is too coarse for pocket side chains, and its cofactor is **NAP, which is NADP+,
the oxidised form**, not NADPH. Docking against the wrong oxidation state of the
cofactor that forms one wall of the folate pocket is the same class of error as
removing it.

1E7W has the best resolution of the three but is *Leishmania major*, has five
chain breaks, models only 92% of the sequence, and its 267-residue chains are the
DHFR domain alone, so it gives up the bifunctional handle. Fallback only.

### Rung 2, Chikungunya: blocked

| pdb | res (A) | R-free | bound ligand | verdict |
|---|---|---|---|---|
| 3TRK | 2.40 | 0.258 | **none** | cannot place a box |
| 4ZTB | 2.59 | 0.242 | **none** | cannot place a box |
| 3HHA (human cathepsin L) | 1.27 | 0.152 | NOW, 28 heavy atoms | excellent counter-target |

**Both candidate CHIKV nsP2 protease structures are apo.** Neither has a bound
inhibitor, so there is no co-crystal to center a docking box on, and this project's
box recipe is defined as the centroid of a co-crystallized inhibitor's head. A box
for nsP2 would have to be placed from the catalytic residues instead. That is a
judgment call rather than a measurement, and an unvalidated box is precisely the
kind of decision that produced the NADPH bug.

The counter-target is not the problem. 3HHA is the best structure in the whole
audit at 1.27 A and R-free 0.152.

**Rung 2 is held pending a liganded nsP2 structure or an explicit, separately
validated box.** Rung 1 does not depend on it.

### One heuristic bug found and fixed in the audit itself

The first run reported 3TRK's box ligand as `MSE` with 8 heavy atoms. MSE is
selenomethionine, a *modified amino acid* that carries `het_flag == "H"` in the
file while being chemically part of the polypeptide. A box centered on it is a
box centered on the backbone. Fixed by asking gemmi's residue table whether a het
code is an amino acid or nucleotide rather than maintaining a hand-list. After the
fix 3TRK reports no ligand, which is the correct answer.

---

## Test 3. Rung-1 correlation pilot: running

Measures the Pearson correlation between T. cruzi DHFR-TS (3INV) affinity and
human DHFR (1U72) affinity over 120 randomly sampled library molecules, 240
docking calls, against F19's floor: the ICM beats independent per-task GPs above
rho about 0.45 at 3.6% co-located labels and about 0.25 at 9.8%. Below the floor
the shared model is worse than two separate ones, so this is a GO / PROTOTYPE /
RECONSIDER gate read off the lower bound of a bootstrap interval, not off the
point estimate.

Everything it reports is a correlation between two columns produced on the same
container, which is internally consistent by construction. It never compares a
Modal score with a committed one. A separate control arm re-docks 24 molecules
that already have committed PfDHFR scores and reports the Modal-minus-laptop
delta as a **measurement of the machine offset**, not as a result. On a 2-molecule
smoke that delta was 0.094 kcal/mol at worst, against the project's recorded
ceiling of 0.612, which is consistent with the Modal image being version-matched
to the laptop on purpose: Vina 1.2.7, Python 3.11, RDKit 2024.03.6.

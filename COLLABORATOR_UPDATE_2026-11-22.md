# Project update, 22 November 2026

Short version: the method now has results on two diseases instead of one, both
wins are complete, and we have screened three further diseases cheaply and
decided to pursue only one of them.

## What we set out to test

Everything we had was on malaria. The obvious objection is that a method tuned
on one target wins on that target, so we pointed the whole pipeline at Chagas
disease without changing any settings, and ran it cold.

## Result 1: the method transfers

Two complete campaigns, ten independent repeats each, against the best
published alternative and against the standard industry approach.

| | malaria | Chagas |
|---|---|---|
| ours | 0.4079 | 0.3791 |
| GP-MOBO, the published competitor | 0.3123 | 0.3048 |
| filter-then-dock, standard practice | 0.1950 | 0.1759 |

The number is hypervolume: a single value between 0 and 1 for how well a set of
candidate molecules covers five competing design goals at once, where higher is
better. The two diseases are deliberately not plotted together, since they are
different proteins scored on different machines.

We win on 10 of 10 repeats against both comparisons, on both diseases, with no
overlap between the repeats. Wilcoxon signed-rank p = 0.002, which is the
smallest value ten paired samples can produce. The margin over the competitor
is about three times the spread among our own repeats.

Practically: at a fixed number of expensive simulations, our search returns a
better set of candidate molecules than the published alternative, and it does so
on a disease it was never tuned for.

## Result 2: a negative finding we are keeping

One component of the method, letting the two protein targets share statistical
information, buys nothing. Shared and independent versions tie at 6 of 10
repeats, p = 0.56. This is now the third dataset to show it, and it is what
theory predicts when every molecule is measured against both targets. The
advantage comes from deciding which molecule to simulate next, not from the
relationship between targets. We would rather state this plainly than quietly
drop the comparison.

## Result 3: two validation checks on Chagas, both passed

**The simulation reproduces a known structure.** We take the drug-like molecule
that was crystallised inside the Chagas enzyme, delete it, and ask the software
to put it back. It lands 1.78 Angstroms from the true position, inside the 2.0
Angstrom convention. For reference the two malaria targets give 0.98 and 0.94,
so Chagas passes with less margin and we say so.

**The search rediscovers real drug chemistry.** The chemical group that defines
known inhibitors of this enzyme family appears in our results at 2.8 times its
background rate in the screening library, p below 0.0001. The same figure on
malaria is also 2.8. It survives a control for molecule size, and the randomly
chosen starting molecules contribute almost none of it, so it comes from the
search rather than from where the search began.

While running this we found and fixed a flaw that also affected the malaria
analysis: the fragment being used as a negative control is in fact present in
four of the six reference drugs, so it was never a valid control. We replaced it
with five fragments that are common in the library, absent from every reference
drug, and chemically unrelated to the target. All five come out flat or depleted
on both diseases while the real signal sits at 2.8. The enrichment is specific,
and both campaigns now have a stronger control set than the original analysis
did.

## Result 4: three more diseases screened and triaged

Before committing a full campaign to a new disease we now run a cheap gate: 300
molecules simulated against both of that disease's proteins, roughly 1 percent
of the cost of a campaign.

| disease | verdict |
|---|---|
| Chikungunya, macrodomain target | **pursue.** Promising molecules appear 5 times more often than in Chagas, which succeeded |
| Dengue | decline. The viral protein and a human blood-clotting protein are too alike, so a molecule blocking the virus likely harms the patient |
| Chikungunya, protease target | decline. Molecules prefer the human protein. Nothing to find |

We regard the two declines as a result in their own right. Screening five
diseases and correctly refusing three before spending on them cost about the
same as a single repeat of one campaign.

## Honest limitations

- All of this is computational. No molecule has been made or tested in a lab.
  The simulation scores are rankings, not predicted potencies, and should never
  be quoted as binding affinities.
- The simulation gives slightly different numbers on different computers, up to
  1.6 kcal/mol in the worst case we measured. Every comparison we report is
  within one machine, and we never place numbers from two machines on one axis.
- The chikungunya protease result may understate that target. Its structure has
  no drug bound to it, so the simulation region had to be chosen rather than
  read off a crystal structure. The decline stands, but the reason is partly
  that we cannot set the problem up properly.

## What happens next

1. One full campaign on the chikungunya macrodomain, same settings as Chagas.
2. One remaining validation check, comparing how easy our proposed molecules
   would be to synthesise against the unfiltered source database rather than
   against the already-filtered library.
3. Figures. The analysis is complete and the underlying data is in hand; the
   figures themselves are drafts and are being redone.

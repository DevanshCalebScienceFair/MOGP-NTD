# The new method, in plain words

No jargon. If a sentence needs a technical term, the term comes after the idea.

---

## The problem we are solving

We want a molecule that jams the malaria parasite's copy of an enzyme called
DHFR, while leaving the human copy of that same enzyme alone. Jam both and you
poison the patient. So we are hunting for a **difference**, not just a strong
binder.

There are about 26,660 candidate molecules. Testing one on the computer takes
around fifteen seconds per target, and testing one in a real lab costs real
money and weeks of time. We cannot test them all. So we test a few, learn from
them, and use what we learned to pick the next few. That loop is the whole
method.

---

## What we already had

A **predictor** that guesses how well an untested molecule will score, and how
confident it is in that guess. It looks at which chemical fragments a molecule
contains and says "molecules with fragments like these have scored around here
before". (Technically: a Gaussian process with a Tanimoto kernel over molecular
fingerprints.)

Then a **chooser** that picks the next batch to test. It does not just pick the
molecule with the best predicted score. It picks the batch that would most
expand the *range* of good options we have — the one that adds the most new
ground, not the one that piles up in a corner we already own. (Technically:
qNEHVI, expected hypervolume improvement.)

---

## The thing we thought was our clever bit

Our predictor handled both targets at once and was allowed to notice that they
are related. The two enzymes are cousins, so a molecule that sticks to one tends
to stick to the other. The idea was: if the model knows the parasite score, that
should sharpen its guess about the human score. Sharper guesses, better picks,
fewer molecules wasted.

That sharing is called **coregionalization**. It was the piece we treated as our
contribution.

## It does not work, and we found out why

We ran it against a plain model that treats the two targets as unrelated. Ten
independent repeats, identical starting molecules, only the model different.

**The clever version led in 197 of 400 checkpoints. That is 49 percent. A coin
flip.** No measurement we took could tell the two apart.

The reason turns out to be simple once you see it, and it is not a bug.

> Sharing only helps when something is missing.

Our pipeline docks **every** molecule against **both** targets. Always. So when
the model sits down to guess a molecule's human score, it already has that
molecule's human score. There is nothing to borrow. The clever machinery has
nothing to be clever about.

There is a name for this and a proof behind it: **autokrigeability** (Bonilla,
Chai & Williams, 2008). When every item is measured on every task, a sharing
model collapses into a non-sharing model. It is not that our targets are
insufficiently related — the measured correlation is 0.788, which is strong. It
is that relatedness is useless when nothing is absent.

**An analogy.** Sharing notes with a classmate helps when you missed a lecture.
If you attended every lecture, their notes add nothing, no matter how good a
student they are. We had perfect attendance.

---

## The new method

Stop attending every lecture.

**Dock every candidate against the parasite target. Dock only some of them
against the human target.**

Now most molecules have one score and not the other, and the model finally has a
real job: use the parasite scores, which we have for everything, to fill in the
human scores we skipped. That is precisely the situation where sharing pays, and
it is the situation autokrigeability says is required.

It is also just a better use of money. Docking is not free. If the model can
infer the human score for most molecules from the parasite score plus a subset
of real human measurements, we spend our budget on more distinct molecules
instead of on a second measurement for every one.

### What had to be built

The old model could not do this. It required a complete grid — every molecule,
every target, no gaps — because of how its mathematics was assembled. Feed it a
missing value and it does not complain — it quietly drops that whole
measurement from the model and reports nothing for it. Which is worse than
failing, because the run looks like it worked. It now refuses instead.

So we wrote a new one (`mogp_hadamard.py`). Instead of a table with a row per
molecule and a column per target, it treats **each individual measurement** as
its own entry: "molecule 47, parasite target, score −9.2". A molecule with one
measurement contributes one entry; a molecule with two contributes two. Gaps are
simply entries that do not exist, so any pattern of missing data is expressible.
It is the same sharing model underneath — the same learned relatedness between
targets — written in a form that permits holes.

(Technically: the Hadamard, or stacked-index, form of the intrinsic
coregionalization model. Same `IndexKernel` task covariance; the observations
are `(molecule, task)` pairs rather than rows of a complete matrix. Eight tests
in `test_mogp_hadamard.py`.)

### Does it work?

Testing it directly: hide a fraction of the human-target scores, then ask each
model to predict held-out human scores.

- With **100% of the labels kept**, the two models are **exactly tied** — the
  difference is 0.000. That is autokrigeability showing up on our own data,
  precisely as the theory says.
- As labels are removed, the sharing model pulls ahead, and the gap widens as
  more go missing.

The direction is right and the tie at 100% is a clean confirmation of the
mechanism. Whether the advantage is large enough to matter is still being
measured with more repeats; the first pass was suggestive, not conclusive.

---

## Two other things we fixed along the way

**A setting nobody set.** The chooser measures "how much new ground would this
batch add" by filling an oddly-shaped region with boxes and adding up their
sizes. Filling it *exactly* needs an absurd number of boxes — 120,829 at one
realistic size. There is a setting that says "do not bother splitting boxes
smaller than this". It was never switched on, so the code did the exact version.
Switching it on: **17 times less compute for the same campaign.** A run that
took a working day now takes half an hour.

That also reversed a claim in our own paper. We had written that our method was
the most accurate but by far the most expensive per unit of compute. It was
never the method — it was the unset setting. Corrected, our method is the
**cheapest** per unit of result, not the most expensive.

**A crash that ate three runs.** A matrix that should have been mathematically
well-behaved came out very slightly wrong in the final decimal places, and the
library refused it rather than shrugging. Three full campaigns died to a rounding
error. Now it nudges the matrix by the smallest amount that fixes it, and only
when something actually breaks, so runs that were fine stay bit-for-bit
identical.

---

## Where this leaves the project

**Still true:** the pipeline beats its comparisons. 0.408 against 0.312 and
0.195, ten repeats, same ordering every time. The selectivity results hold after
filtering out the docking artifacts.

**No longer true:** that coregionalization is why it wins. It is not. A simpler
model does the same job under our design.

**The new claim,** which is sharper and which we can actually defend:

> Coregionalization cannot help when every molecule is measured on every target,
> and we show this both theoretically and across ten repeats. It becomes useful
> exactly when measurement is uneven — which is the normal situation in a real
> lab, where one assay is always cheaper than the other. We provide a model that
> handles that case and show the crossover.

That is a more useful finding than "we used a fancy model and it won", and it
came from a result that went against us.

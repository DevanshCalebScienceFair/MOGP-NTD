# What each new method actually adds, in plain words

No jargon. One entry per thing built on this branch, what the old code did, and
what changed. Figure: `F25_where_we_stand.png`.

---

## The honest tally first

**6 wins, 4 losses, 1 mixed.** And the losses all belong to one category.

Every failed idea was a **tuning knob** on an optimizer that already worked:
move a normalization bound, filter what goes on the front, filter what goes into
training, move the reference point.

Every success was **infrastructure or a new capability**: make it 17x cheaper,
make it handle missing data, stop it crashing, stop it misreporting what it did.

That is not bad luck. A pipeline this well tuned has little left on the knobs
and a lot left in what it can *do*. It is also why the next ideas below are
structural rather than another knob.

---

## The wins

### 1. The speed fix — the biggest single change

**Before:** one campaign took 8 hours and 8 minutes. The paper said our method
was the most accurate but by far the most expensive.

**After:** 29 minutes. Same answers' quality, 17 times less computing.

**In plain terms:** the program was measuring the size of an oddly shaped space
by filling it with boxes, and it was told to do that *perfectly*. Perfect takes
120,829 boxes. There is a setting that says "close enough is fine" — nobody had
switched it on. Switching it on uses 124 boxes.

**Why it matters:** the paper's claim that accuracy was bought with compute was
never true. Corrected, the method is the *cheapest* per unit of result, not the
most expensive. It also made every experiment on this branch affordable — none
of the last three weeks of runs would have fit otherwise.

### 2. A model that can handle missing measurements

**Before:** the model needed a complete table — every molecule tested against
both targets, no blanks. Hand it a blank and it did not complain; it quietly
stopped modelling that target and returned "no answer" for half the objective. A
run would finish and print plausible numbers.

**After:** the model stores a *list of measurements* instead of a table. A
molecule measured once contributes one line. A gap is simply a line that was
never written.

**Why it matters:** real labs never have complete tables. One assay is always
cheaper than the other, old records have holes, screens are partial. This is the
only version that can use that data at all — and on complete data it performs
identically, so nothing was traded away for it.

### 3. Three crashes that no longer happen

**Before:** a rounding error in one matrix killed three entire campaigns
mid-run. Hours of work, gone, with a linear-algebra error message.

**After:** it nudges the matrix by the smallest amount that fixes it, and only
when something actually breaks — so runs that were fine are bit-for-bit
unchanged.

### 4. It stopped lying about what it did

Four separate times, a number the program printed did not mean what it said: a
"seed" setting that was actually a count, a hypervolume compared against a
different ruler, a baseline size that ignored its own filter, a "partly
labelled" count that went negative.

**Why it matters:** every one of those would have been read as a result. One of
them — a six-seed sweep that silently reused the same seed six times — was only
caught because two "different" runs gave byte-identical answers.

### 5. Settings that had to exist

`loop.py` could not set a random seed or cap the candidate pool. Without the
first, a multi-seed experiment is one experiment repeated. Without the second,
every run was six times slower than every other result it was compared to.

---

## The losses, and what they taught

| tried | result | what it told us |
|---|---|---|
| widen the hDHFR axis | worse | the axis really is truncated, but widening it rewards clashing poses |
| keep clashes off the front | nothing | artifacts enter through the **model**, not the front |
| keep clashes out of training | worse | **removing bad data is not the same as teaching the model the data was bad** |
| move the reference point | null | one borderline endpoint, not worth acting on |

The third is the most useful. Deleting those rows leaves the model with *no
information* where the data was — and an optimizer whose entire job is to
investigate the unknown walks straight back in. That failure was written down
and committed *before* the run, and it happened exactly as predicted.

**Conclusion:** docking artifacts cannot be fixed inside the optimizer. A failed
pose is a broken *measurement*, and the place to fix it is the docking step —
before it ever becomes a data point.

---

## What we CAN add — in order

### 1. Treat the three ADMET properties as pass/fail, not as goals

**The single biggest change available, and it is structural.**

Right now the search juggles five goals at once. With five, beating a rival
requires being better on **all five** — which almost never happens, so **62.8%
of everything ends up "best of its kind"** and the label stops meaning anything.
That same explosion is what made the computer work so hard (62,433 boxes).

But three of those five — the safety and absorption numbers — **are already known
exactly**. The model never predicts them. So they do not need to be goals; they
can be a bar to clear.

Measured on this project's own data:

- dropping any one ADMET goal shrinks the front by 17–22 points, so they are
  **half the problem**, not passengers
- a lenient bar still lets **50% of the library through**, so the search keeps
  room to work
- going from 5 goals to 2 takes the front from **62.8% to 0.7%** and the box
  count from **62,433 to 3**

"Best of its kind" would mean something again, and the approximation that
currently distorts which molecule looks best would no longer be needed.

### 2. Remove the 2,000-molecule cap

Each round the search scores only 2,000 of 26,660 candidates — **7.5% of the
library, chosen at random, with the rest ignored.** That cap exists for exactly
one reason: the compute cost we just cut by 17×.

Nobody has re-tested it since. It is the cheapest experiment on this list.

### 3. Penalise bad poses instead of deleting them

The failed experiment above deleted them and left a hole. Instead, keep the row
and relabel it as *bad* — so the model learns that region is worthless rather
than learning nothing about it. Directly motivated by a measured failure.

### 4. More repeats

**Six repeats was the binding limit on every single experiment here** — it caps
the statistics at p = 0.031, so several questions came back "cannot tell" purely
for lack of samples. The compute to fix that now exists.

---

## What has not changed

The pipeline still beats its comparisons: **0.408 against 0.312 and 0.195**, ten
repeats, same ordering every time. The selectivity results still hold after
filtering out the broken poses.

What changed is the *explanation*. It does not win because of the shared-learning
model — a simpler one does the same job under this design, and we can prove why.
That is a sharper claim than the original, and it came from results that went
against us.

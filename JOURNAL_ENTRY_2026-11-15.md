Work Day 17                                                         11/15/2026

Today we answered a question that has been sitting underneath this whole project
without us noticing it, and we answered it on Modal at full scale instead of on
the laptop.

**The question nobody had asked.** Everything we have done with coregionalization
so far has been about WHERE the labels are. F12 showed that when every molecule
is measured on both targets, coregionalization does literally nothing (197 of 400
matched checkpoints, a coin flip). F13 showed that when labels are missing, it
helps, and helps more the more are missing. F17 found the sweet spot for how many
molecules should carry both labels (8 out of 225, 3.6%). F18 showed that WHICH
molecules you pick does not matter, which was a negative result we had
pre-registered and got wrong.

Every single one of those fixes the similarity between the two targets at the one
value our data happens to have, corr(PfDHFR, hDHFR) = 0.711, and varies something
else. So the number that decides whether any of this transfers to a new disease
did not exist: **how similar do two protein targets have to be before sharing
information between them actually helps?**

That matters right now because we want to go after Chagas, chikungunya and
dengue, and every guess at a new target costs a receptor preparation plus a whole
arena of docking. Guessing wrong is expensive.

**What we built.** `analysis_scripts/correlation_floor.py`. It takes the real
PfDHFR docking scores and manufactures a second target whose similarity to them
is whatever number we choose. The trick is that we force the similarity to be
exact rather than approximate: we take the noise, subtract off whatever part of it
points in the PfDHFR direction, and mix in the proportion we want. So when we ask
for 0.3 we get 0.3, not "roughly 0.3". We checked and it lands to sixteen decimal
places.

Then we sweep that similarity from 0.1 to 0.95 and ask, at each value, whether
the coregionalized model predicts the second target better than two separate
models would.

**We pre-registered the prediction before running anything** (commit `4fc1e20`),
same as we did for F17 and F18. Five clauses, two of which could kill the
experiment on their own.

**Modal.** This is the first time we have run real science on Modal rather than
tests. 24 containers, 4 CPUs each, all at once, 960 model fits, about 8 minutes
of wall clock. On the laptop this would have been roughly two hours of saturated
fans.

Before trusting a single number off it we ran a 3-cell smoke test, and we built
the smoke test so that it reran THE EXACT SAME three repeats we had already run on
the laptop for F17 (same random seed stream). Not a similar run, the identical
computation. Result:

    rep   ICM laptop   ICM Modal   independent laptop   independent Modal
      0       1.5114      1.5114               1.3310              1.3310
      1       2.0299      2.0299               2.1305              2.1305
      2       1.4766      1.4766               1.7148              1.7148

Biggest disagreement anywhere: 0.0000004 kcal/mol. For scale, our docking oracle
disagrees with itself by up to 0.612 kcal/mol between machines, which is the
thing that normally makes cross-machine numbers illegal to compare. This is a
million times smaller, so it is just float rounding. **Modal results and laptop
results can go on the same axis, as long as nothing docks.**

**The smoke test also caught a bug, and it was our old friend.** The script wrote
its data file perfectly and silently skipped printing the summary table. No error.
The reason is that Modal's "local entrypoint" runs in whichever Python you typed
`modal run` with, which is the Modal CLI's Python, and that one has no PyTorch.
The import failed, the exception got swallowed, and we got a valid-looking run
that measured nothing. That is the seventh time this exact shape of bug has bitten
us on this project (valid output, no error, nothing actually measured), so it went
straight into CLAUDE.md with the others.

----------------------------------------------------------------------
THE RESULT
----------------------------------------------------------------------

    co-located labels   % of training set   crosses zero at   PROVEN from
    k = 8  (F17's peak)        3.6%            rho = 0.450    rho >= 0.711
    k = 22                     9.8%            rho = 0.259    rho >= 0.500

**The floor moves depending on how much you are willing to spend.** If you can
afford to measure 9.8% of your molecules on both targets, the targets only need to
be 0.5 similar. If you can only afford 3.6%, they need to be 0.711 similar. More
shared measurements buy you the right to use a weaker relationship. That was NOT
in our prediction and it is the most useful thing to come out of the day, because
it turns a yes/no question into a budget you can trade.

The effect is perfectly ordered: Spearman between similarity and benefit is
**+1.000** at both budgets. It runs from -0.0695 kcal/mol at the weakest coupling
up to +0.2287 at the strongest.

All five pre-registered clauses resolved, and the big one passed exactly. Clause 4
said the "real" cell, using the actual human DHFR column instead of a manufactured
one, had to reproduce F17's measured +0.122. It came back **+0.1220, 19 wins out
of 20, p = 0.0002**, against F17's +0.122, 19/20, p = 0.0002. If that had missed,
the whole sweep would have been sitting on a different axis and we could not have
plotted it next to anything.

----------------------------------------------------------------------
TWO THINGS WE GOT WRONG, WRITTEN DOWN ON PURPOSE
----------------------------------------------------------------------

**1. We predicted which way our fake second target would bias the answer, twice,
and we were wrong both times.** In the pre-registration we argued that because the
manufactured target is built with exactly the kind of straight-line relationship
the model assumes, it would flatter the model and the floor would come out too
optimistic. Then after the smoke test we saw a hint pointing the other way and
revised to the opposite story, that the real pair shares chemical structure a
manufactured one cannot, so we would be understating the benefit.

At 20 repeats: real minus manufactured is +0.0247 at one budget and -0.0170 at the
other, p = 0.33 and 0.19. Opposite signs, neither significant, both smaller than
the repeat-to-repeat noise. **There is no measurable bias in either direction.**
Both of our stories were wrong and the honest answer is that the stand-in is
simply faithful.

**2. "The model actively hurts below the floor" does not survive correction.**
The raw p-value at the weakest coupling is 0.0121, which reads as a real negative
effect, and we nearly wrote it up that way. After Holm correction across the ten
cells it is 0.073, and **nothing below 0.5 similarity is significant in either
direction at either budget.** So the claim is "no demonstrated benefit", never
"does harm". Written into CLAUDE.md as a do-not-misstate.

----------------------------------------------------------------------
SOMETHING WE DID NOT PREDICT, AND IT HITS EXACTLY WHERE WE CARE
----------------------------------------------------------------------

A NEGATIVE relationship between the two targets pays off just as much as a
positive one of the same strength (gap +0.020 at both budgets, p = 0.73 and 0.50,
so no real difference). But the model has a much harder time PINNING DOWN a
negative relationship: the spread in what it learns is 2.3x wider.

This matters more than it sounds. Selectivity is the whole point of our second
target: we want molecules that bind the parasite and NOT the human protein, which
means we WANT the two scores to disagree. So the case our project is built around
is exactly the case where the learned link is least stable. Our post-hoc guess
(labelled as a guess) is that the math has to flip a sign internally to express a
negative relationship, and the optimizer does not find that flip as reliably from
where it starts.

----------------------------------------------------------------------
WE HAD TO WALK SOMETHING BACK FROM EARLIER THE SAME DAY
----------------------------------------------------------------------

This morning, while picking which disease to attack next, we found what looked
like a free win. Chagas disease's version of our target enzyme, T. cruzi DHFR-TS,
is BIFUNCTIONAL: it fuses two enzymes into one protein, which humans do not do.
That looked like a free third task to feed the model, at almost no extra receptor
work, and we got excited about it.

F19 says slow down. A third task is only worth having if it is similar enough to
the main one to clear the floor, and the fused half binds a completely different
substrate through completely different chemistry, so there is no reason to expect
a high similarity. At our best co-location rate the bar is 0.711. **The free third
task is now conditional on a cheap test and might be worth nothing.** Recording
the retraction next to the original idea rather than quietly dropping it.

----------------------------------------------------------------------
WHAT A RESEARCHER ACTUALLY DOES WITH THIS
----------------------------------------------------------------------

This is the first thing we have produced that is a usable design rule rather than
a result about our own code.

Before anyone commits to preparing a new receptor and docking a whole library
against it, **dock about 100 molecules against both targets and measure the
correlation.** The uncertainty in a correlation is roughly (1 - r squared) divided
by the square root of (n - 1), so 100 molecules places it to within about 0.05 to
0.09. That is not precise, but it is easily enough to tell "below 0.3" from "above
0.7", which is the only call you need to make.

Applied to the diseases we are considering:

    T. cruzi DHFR vs human DHFR    same enzyme family, similar identity to our
                                   measured 0.711 pair -> likely clears, still test
    DENV1-4 proteases              65-70% identity between serotypes -> highest
                                   expected similarity, best technical fit
    CHIKV nsP2 vs cathepsin L      papain family, moderate -> plausible, test first
    the fused TS half              different substrate -> at risk, see above

----------------------------------------------------------------------
IN LAYMANS TERMS
----------------------------------------------------------------------

Our software learns about two proteins at once and uses what it learns about one
to make better guesses about the other. Until today nobody had asked how alike the
two proteins have to be for that sharing to be worth anything. We measured it.

If you can afford to test 1 molecule in 10 against both proteins, the proteins
need to be about 50% alike in how they score molecules. If you can only afford 1
in 28, they need to be about 70% alike. Below that, sharing does not help, though
it does not hurt either.

The practical payoff is a cheap go/no-go test: 100 molecules run against both
proteins tells you whether the method will work on your disease, before you spend
weeks of compute finding out the hard way.

----------------------------------------------------------------------
WHAT THIS DOES NOT SHOW
----------------------------------------------------------------------

The CEILING. We found the lower edge of the useful range. There is an upper edge
too and we have not measured it: as the two targets get MORE alike the model
shares more effectively, but the selectivity window closes, because two proteins
that score every molecule identically cannot be traded off against each other. So
the usable band has a measured floor and an unmeasured ceiling.

And as with F13, F17 and F18, this is about PREDICTION ACCURACY at a fixed number
of labels. It says nothing about whether a full search campaign ends up better.
F14 is still the only budget-matched campaign comparison we have run, and its
sparse-label arm lost 6 out of 6.

Next: the known-drug enrichment check, which uses only docking we have already
paid for and decides whether our closing figure is the paper's strongest claim or
a stated limitation.

Commits: `4fc1e20` (pre-registration), `fe63d80` (result), both on
`ExtraNovelPipeline`.

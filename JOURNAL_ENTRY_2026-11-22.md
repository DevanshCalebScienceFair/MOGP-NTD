# Journal, 2026-11-22: the method travels

Today the thing I most wanted to know turned out well. The method works on a
second disease.

Everything up to now has been malaria. One parasite, one enzyme, one set of
numbers. A reviewer could reasonably say we tuned the whole pipeline to
PfDHFR and that the win is a property of that one target rather than of the
search. So we pointed the entire apparatus at Chagas disease, swapped the
parasite enzyme to *T. cruzi* DHFR against the same human counter-target, and
ran it cold. Nothing was retuned. Same acquisition, same library, same budget.

It won on every single seed.

| | malaria | Chagas |
|---|---|---|
| ours | 0.4079 | 0.3791 |
| GP-MOBO, the published competitor | 0.3123 | 0.3048 |
| filter-then-dock, what industry does | 0.1950 | 0.1759 |

Ten seeds per arm, both diseases. Ten out of ten wins against both comparisons,
both times, with no overlap between the two sets of seeds. The margin over the
competitor is about three times the spread between our own seeds, so this is
not a lucky draw.

What that means outside the plot: given the same number of expensive docking
calculations, our search returns a meaningfully better spread of candidate
molecules than the best published alternative. A screening group with a fixed
compute budget gets more usable chemistry out of the same spend, and gets it on
a disease the method had never seen.

The honest asterisk, which I like more than I expected to: sharing information
between the two protein targets still buys nothing. Shared and independent task
models tie at 6 of 10 seeds. That is the third dataset to say so, and it is
exactly what the theory predicts when every molecule is measured against both
targets. The win is coming from the acquisition, not from the correlation. We
have been saying that for a while, and now a second disease says it too.

Also pleasing: a real competitor was in the room this time. Not a strawman we
wrote, the actual published GP-MOBO implementation, pinned to its upstream
commit and run on our problem. It beat filter-then-dock handily. It still lost
to us on every seed.

Three bugs in that competitor's path had to be found first, and all three were
the quiet kind that produce plausible numbers while measuring nothing. The
pattern holds: on this project the dangerous failures never crash.

One thing deliberately left out of the figures. The two cheap arms, the ones
that skip most of the second target's measurements, score terribly, and that
score is not real. A molecule missing one objective cannot enter a hypervolume
at all, so those arms were being graded on 24 molecules while the full arm was
graded on 290. That is a rigged comparison and it goes in the drawer until it
can be scored on the nomination endpoint, where both arms rank the same
unmeasured library. Catching that before it reached a figure was the second
best thing that happened today.

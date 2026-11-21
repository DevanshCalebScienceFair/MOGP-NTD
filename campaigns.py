"""Named disease campaigns: which targets a run optimizes, and in what order.

WHY THIS EXISTS. `mogp.TASK_NAMES` was a module-level list hardwired to malaria
(`PfDHFR_Docking` first), and every other module imports it as the single source
of objective order. That was correct while this pipeline had one disease. It has
always been intended for several, so the objective set becomes DATA here and
`mogp` reads it, rather than each new disease being an edit to five files.

WHAT A CAMPAIGN IS. One entry fixes four things that must agree with each other
or the run is silently measuring the wrong thing:

  task_names   objective order; every result CSV column name comes from this
  sources      how each objective is produced: ("dock", <docking.TARGETS key>)
               or ("library", <data.ADMET_COLUMNS name>)
  signs        optimization direction per objective, +1 maximize / -1 minimize,
               order-aligned with task_names
  bounds_file  normalization bounds for the hypervolume, per campaign, because
               two campaigns' hypervolumes are NOT commensurable

SELECTION IS BY ENVIRONMENT VARIABLE, `MOGP_CAMPAIGN`, defaulting to `malaria`.
That default makes every existing script, test and analysis byte-identical to
before this file existed.

A global selected by an environment variable is exactly the shape that produced
four of the defects in ENGINEERING_LOG.md: a value accepted somewhere the user
looks and not arriving where it is consumed. Three things guard it:

  1. `describe()` prints the resolved campaign and its full objective list, and
     every runner calls it at startup, so the run log states what it optimized.
  2. `active_campaign()` raises on an unknown name. A typo is not a silent
     fallback to malaria.
  3. `record()` returns a dict that runners write into run_config.json, so the
     campaign is in the artifact, not only in the environment that produced it.

ADDING A DISEASE. Add an entry here and a receptor to `docking.TARGETS`. Do not
edit `mogp.TASK_NAMES`.
"""

from __future__ import annotations

import os


# Shared tail of every campaign: the three cheap ADMET objectives, precomputed
# for the whole library. Order and names must match `data.ADMET_COLUMNS`.
_ADMET_TASKS = ["hERG_Toxicity_Prob", "Caco2_logPapp", "Half_Life_hours"]
_ADMET_SOURCES = {
    "hERG_Toxicity_Prob": ("library", "hERG_Toxicity_Prob"),
    "Caco2_logPapp":      ("library", "Caco2_logPapp"),
    "Half_Life_hours":    ("library", "Half_Life_hours"),
}
_ADMET_SIGNS = [-1, +1, +1]          # hERG down, Caco2 up, half-life up


def _selectivity_campaign(primary_task, primary_target, counter_task,
                          counter_target, bounds_file, label, aux=()):
    """Build the standard two-docking-task selectivity campaign.

    Every campaign in this project has the same shape: one PARASITE target to
    bind tightly (minimize kcal/mol), one HUMAN counter-target to avoid
    (maximize, i.e. bind weakly), and the three ADMET objectives. `aux` adds
    further docking tasks that the GP models but which are NOT part of the
    hypervolume; see the `emit_objectives` note in the chagas entry.
    """
    tasks = [primary_task, counter_task] + [a[0] for a in aux] + _ADMET_TASKS
    sources = {primary_task: ("dock", primary_target),
               counter_task: ("dock", counter_target)}
    sources.update({a[0]: ("dock", a[1]) for a in aux})
    sources.update(_ADMET_SOURCES)
    signs = [-1, +1] + [-1] * len(aux) + _ADMET_SIGNS
    return dict(label=label, task_names=tasks, sources=sources, signs=signs,
                bounds_file=bounds_file, primary=primary_task,
                counter=counter_task,
                auxiliary=[a[0] for a in aux])


CAMPAIGNS = {
    # The published malaria campaign. THIS IS THE DEFAULT AND MUST NOT CHANGE.
    # Its bounds file and every result under campaign_results/ are read-only.
    "malaria": _selectivity_campaign(
        "PfDHFR_Docking", "PfDHFR", "hDHFR_Docking", "hDHFR",
        "evaluation_bounds.json",
        "Plasmodium falciparum DHFR vs human DHFR (published)"),

    # Rung 1. T. cruzi DHFR-TS (3INV) against the same human counter-target.
    # Gated on a measured task correlation of rho = +0.719, 95% CI
    # [+0.623, +0.799], n = 281, which clears F19's floor at both label budgets.
    #
    # NO TS-DOMAIN TASK. 3INV is bifunctional and the TS domain is resolved, so
    # a third task is structurally available and was planned. It is NOT here,
    # because this project defines a docking box as the centroid of a
    # co-crystallized inhibitor's head and no TS-site ligand inventory for 3INV
    # has been produced. Placing that box from catalytic residues instead is a
    # judgment call, which is how the NADPH bug happened and why rung 2 is held.
    # When a TS-site ligand is confirmed, add it as `aux=(("TcTS_Docking",
    # "TcTS"),)` and exclude it from the hypervolume with loop.py's
    # `emit_objectives`; no other file changes.
    "chagas": _selectivity_campaign(
        "TcDHFR_Docking", "TcDHFR", "hDHFR_Docking", "hDHFR",
        "evaluation_bounds_chagas.json",
        "Trypanosoma cruzi DHFR-TS (3INV) vs human DHFR"),
    # Rung 2. The primary and counter targets are the SAME FOLD with the same
    # natural ligand, so this is the most demanding selectivity pair in the
    # project and the one where the shared model has the most to borrow.
    # Not runnable until a correlation pilot measures rho against F19's floor.
    "chikungunya": _selectivity_campaign(
        "CHIKVnsP3_Docking", "CHIKV_nsP3", "hMacroD2_Docking", "hMacroD2",
        "evaluation_bounds_chikv.json",
        "Chikungunya nsP3 macrodomain (9YHH) vs human MacroD2 (4IQY)"),
    # Rung 2, protease route. Runnable, but its box is centred on the catalytic
    # dyad rather than on a bound ligand, because no liganded nsP2 protease
    # structure exists. Any result from it carries that caveat; see
    # docking.RECEPTORS["CHIKV_nsP2"].
    "chikungunya_protease": _selectivity_campaign(
        "CHIKVnsP2_Docking", "CHIKV_nsP2", "hThrombin_Docking", "hThrombin",
        "evaluation_bounds_chikv_pro.json",
        "Chikungunya nsP2 protease (3TRK, dyad box) vs human thrombin (5DGJ)"),
    # Rung 3. Both boxes come from co-crystallized inhibitors, and both
    # proteases are trypsin-like, so the counter-screen is a real liability
    # rather than a contrived one.
    "dengue": _selectivity_campaign(
        "DENVNS3_Docking", "DENV_NS3pro", "hThrombin_Docking", "hThrombin",
        "evaluation_bounds_dengue.json",
        "Dengue NS2B-NS3 protease (6Y3B) vs human thrombin (5DGJ)"),
}

DEFAULT_CAMPAIGN = "malaria"
ENV_VAR = "MOGP_CAMPAIGN"


def active_name():
    """The campaign name this process is running, from `MOGP_CAMPAIGN`."""
    return os.environ.get(ENV_VAR, DEFAULT_CAMPAIGN)


def active_campaign():
    """The resolved campaign dict. Raises on an unknown name, never defaults."""
    name = active_name()
    if name not in CAMPAIGNS:
        raise ValueError(
            f"{ENV_VAR}={name!r} is not a known campaign. Known: "
            f"{sorted(CAMPAIGNS)}. A typo here would otherwise run the wrong "
            f"disease and report it as the right one.")
    return CAMPAIGNS[name]


def record():
    """Provenance dict for run_config.json. The artifact states its own disease."""
    c = active_campaign()
    return {"campaign": active_name(),
            "campaign_label": c["label"],
            "task_names": list(c["task_names"]),
            "objective_signs": list(c["signs"]),
            "docking_targets": [v[1] for k, v in c["sources"].items()
                                if v[0] == "dock" and k in c["task_names"]],
            "bounds_file": c["bounds_file"]}


def describe():
    """One block naming the campaign and every objective, for the run log."""
    c = active_campaign()
    lines = [f"CAMPAIGN: {active_name()}  ({c['label']})"]
    for j, t in enumerate(c["task_names"]):
        kind, ref = c["sources"][t]
        direction = "maximize" if c["signs"][j] > 0 else "minimize"
        aux = "  [auxiliary, not in hypervolume]" if t in c["auxiliary"] else ""
        lines.append(f"  {j}  {t:22s} {direction:8s} {kind}:{ref}{aux}")
    lines.append(f"  bounds: {c['bounds_file']}")
    return "\n".join(lines)

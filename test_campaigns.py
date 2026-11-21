"""Pins the campaign registry: malaria is unchanged, chagas is actually wired.

The defect this guards against is the project's commonest shape: a value is
accepted where the user looks and never arrives where it is consumed. A campaign
selected by an environment variable is exactly that shape, so these tests assert
the resolved objective set at the CONSUMERS (mogp, evaluation, acquisition),
not at the registry that produces it.

Campaign selection happens at IMPORT time, so each cross-campaign test runs in a
subprocess with its own MOGP_CAMPAIGN. Reloading in-process would leave other
already-imported modules holding the old campaign, which is the subtle version
of the bug these tests exist to catch.
"""
import json
import os
import subprocess
import sys

import pytest

import campaigns

HERE = os.path.dirname(os.path.abspath(__file__))

# Exactly what mogp.TASK_NAMES was before campaigns.py existed. If this list
# ever needs editing, something has changed the published malaria campaign.
MALARIA_TASKS = ["PfDHFR_Docking", "hDHFR_Docking", "hERG_Toxicity_Prob",
                 "Caco2_logPapp", "Half_Life_hours"]
MALARIA_SIGNS = [-1, +1, -1, +1, +1]


def _in_campaign(name, code):
    """Run `code` in a subprocess under MOGP_CAMPAIGN=name, return parsed JSON."""
    env = dict(os.environ, MOGP_CAMPAIGN=name, KMP_DUPLICATE_LIB_OK="TRUE")
    out = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


PROBE = """
import json, mogp, evaluation, acquisition
print(json.dumps({
    "tasks": mogp.TASK_NAMES,
    "dock_idx": mogp.DOCKING_TASK_INDICES,
    "dock_targets": {k: v[1] for k, v in mogp.OBJECTIVE_SOURCES.items()
                     if v[0] == "dock" and k in mogp.TASK_NAMES},
    "aux": mogp.AUXILIARY_TASKS,
    "primary": evaluation.PRIMARY_TASK,
    "counter": evaluation.COUNTER_TASK,
    "bounds": evaluation.BOUNDS_PATH.split("/")[-1],
    "signs": acquisition.DEFAULT_OBJECTIVE_SIGNS,
}))
"""


# ---------------------------------------------------------------- malaria

def test_default_is_malaria_and_nothing_about_it_changed():
    """No env var set must reproduce the pre-campaigns behaviour exactly."""
    env = dict(os.environ); env.pop("MOGP_CAMPAIGN", None)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=HERE, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["tasks"] == MALARIA_TASKS
    assert got["signs"] == MALARIA_SIGNS
    assert got["primary"] == "PfDHFR_Docking"
    assert got["bounds"] == "evaluation_bounds.json"


def test_malaria_named_explicitly_matches_the_default():
    assert _in_campaign("malaria", PROBE)["tasks"] == MALARIA_TASKS


# ----------------------------------------------------------------- chagas

def test_chagas_swaps_the_primary_target_everywhere_at_once():
    """The point of the registry: ONE setting moves every consumer together."""
    g = _in_campaign("chagas", PROBE)
    assert g["tasks"][0] == "TcDHFR_Docking"
    assert g["primary"] == "TcDHFR_Docking"
    assert g["dock_targets"] == {"TcDHFR_Docking": "TcDHFR",
                                 "hDHFR_Docking": "hDHFR"}
    # Unchanged across campaigns: the human counter-target, the ADMET tail,
    # the directions, and which objectives the GP models.
    assert g["counter"] == "hDHFR_Docking"
    assert g["tasks"][2:] == MALARIA_TASKS[2:]
    assert g["signs"] == MALARIA_SIGNS
    assert g["dock_idx"] == [0, 1]


def test_chagas_gets_its_own_bounds_file():
    """Two campaigns' hypervolumes are not commensurable; separate files make
    that structural rather than a rule someone has to remember."""
    g = _in_campaign("chagas", PROBE)
    assert g["bounds"] == "evaluation_bounds_chagas.json"
    assert g["bounds"] != "evaluation_bounds.json"


def test_chagas_has_no_auxiliary_task_yet():
    """The TS domain is deliberately absent: 3INV's TS site has no confirmed
    co-crystallized ligand, and this project's box recipe needs one. If this
    test starts failing, a third task was added and it needs a box provenance
    note in docking.TARGETS."""
    assert _in_campaign("chagas", PROBE)["aux"] == []


# -------------------------------------------------------------- the guards

def test_an_unknown_campaign_raises_rather_than_defaulting():
    """A typo must not silently run malaria and report it as the right disease."""
    env = dict(os.environ, MOGP_CAMPAIGN="chagass", KMP_DUPLICATE_LIB_OK="TRUE")
    out = subprocess.run([sys.executable, "-c", "import mogp"], cwd=HERE,
                         env=env, capture_output=True, text=True)
    assert out.returncode != 0
    assert "not a known campaign" in out.stderr


def test_every_campaign_is_internally_consistent():
    """signs, sources and task_names must agree, or the run optimizes one thing
    and reports another."""
    for name, c in campaigns.CAMPAIGNS.items():
        assert len(c["signs"]) == len(c["task_names"]), name
        assert set(c["task_names"]) <= set(c["sources"]), name
        assert c["primary"] in c["task_names"], name
        assert c["counter"] in c["task_names"], name
        assert set(c["auxiliary"]) <= set(c["task_names"]), name
        # The primary is minimized (bind the pathogen), the counter maximized
        # (avoid the human enzyme). A flipped sign here inverts selectivity.
        assert c["signs"][c["task_names"].index(c["primary"])] == -1, name
        assert c["signs"][c["task_names"].index(c["counter"])] == +1, name


def test_every_docking_objective_names_a_real_receptor():
    """A campaign referencing a target that docking.py does not define would
    fail only at the first dock, hours into a run."""
    import docking
    for name, c in campaigns.CAMPAIGNS.items():
        for task in c["task_names"]:
            kind, ref = c["sources"][task]
            if kind == "dock":
                assert ref in docking.TARGETS, f"{name}: {task} -> {ref}"


def test_record_states_the_disease_for_run_config():
    r = campaigns.record()
    assert r["campaign"] == "malaria"
    assert r["task_names"] == MALARIA_TASKS
    assert "PfDHFR" in r["docking_targets"]

#!/usr/bin/env python3
"""Live progress dashboard for an NRP campaign, as a self-contained HTML file.

Run it once for a snapshot, or with --watch to keep rewriting the same file.
The page reloads itself from disk, so a browser left open on it tracks the
cluster without anything being served over HTTP.

Everything it knows comes from `kubectl`, so it is a view of the CLUSTER and
never of an expectation: a cell this invocation cannot see is reported absent
rather than assumed fine. That is the sweep-file rule applied to monitoring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import re
import subprocess
import sys
import time

NS = "lahs-asi"
# Both rungs at once. A set-based selector so rung 2 appears the moment it is
# launched, rather than the dashboard quietly showing only half the cluster.
SELECTOR = "app in (rung1-campaign,rung2-campaign)"

# Rounds each arm is configured for, from nrp/run_cell.py. Used only to turn a
# round number into a percentage; an arm missing here still shows its round.
ARM_ROUNDS = {
    "full": 50, "indep": 50, "asym": 85, "asym102": 102,
    "gpmobo": 50, "greedy": 58,          # greedy: 290 molecules / batch 5
}

ARM_LABEL = {
    "full": "full (ICM, all labels)",
    "indep": "independent per-task GPs",
    "asym": "asym (sparse hDHFR, 85 rounds)",
    "asym102": "asym102 (budget-matched, 102 rounds)",
    "gpmobo": "GP-MOBO (external competitor)",
    "greedy": "greedy filter-then-dock",
}

ITER_RE = re.compile(r"\[Iteration (\d+)\]")
HV_RE = re.compile(r"hypervolume=([0-9.]+)")


def kubectl_json(kind: str) -> dict:
    out = subprocess.run(
        ["kubectl", "-n", NS, "get", kind, "-l", SELECTOR, "-o", "json"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"kubectl get {kind} failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def pod_tail(pod: str, lines: int = 80) -> str:
    out = subprocess.run(
        ["kubectl", "-n", NS, "logs", pod, f"--tail={lines}"],
        capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def collect() -> dict:
    jobs = kubectl_json("jobs")
    pods = kubectl_json("pods")

    # job name -> newest pod (a preempted cell has several)
    by_job: dict[str, dict] = {}
    for p in pods.get("items", []):
        owner = p["metadata"].get("labels", {}).get("job-name")
        if not owner:
            continue
        prev = by_job.get(owner)
        if prev is None or p["metadata"]["creationTimestamp"] >= prev["metadata"]["creationTimestamp"]:
            by_job[owner] = p

    now = dt.datetime.now(dt.timezone.utc)
    cells = []
    for j in jobs.get("items", []):
        name = j["metadata"]["name"]
        labels = j["metadata"].get("labels", {})
        st = j.get("status", {})
        if st.get("succeeded"):
            state = "Complete"
        elif st.get("failed"):
            state = "Failed"
        elif st.get("active"):
            state = "Running"
        else:
            state = "Pending"

        start = st.get("startTime")
        end = st.get("completionTime")
        t0 = dt.datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
        t1 = dt.datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now
        elapsed = int((t1 - t0).total_seconds()) if t0 else None

        pod = by_job.get(name)
        node = pod["spec"].get("nodeName") if pod else None
        restarts = 0
        if pod:
            restarts = sum(c.get("restartCount", 0)
                           for c in pod.get("status", {}).get("containerStatuses", []) or [])

        arm = labels.get("arm") or name.split("-")[1]
        rnd = hv = None
        phase = ""
        err = ""
        if state == "Running" and pod:
            txt = pod_tail(pod["metadata"]["name"])
            it = ITER_RE.findall(txt)
            if it:
                rnd = int(it[-1])
            hvs = HV_RE.findall(txt)
            if hvs:
                hv = float(hvs[-1])
            if "Traceback" in txt:
                err = "Traceback in log"
            if rnd is None:
                if "extracting" in txt and "[recv]" not in txt:
                    phase = "extracting env"
                elif "[recv]" in txt or "Downloaded" in txt:
                    phase = "preparing receptors"
                elif "Docking molecule" in txt:
                    phase = "initial design"
                else:
                    phase = "starting"

        total = ARM_ROUNDS.get(arm)
        if state == "Complete":
            pct = 100.0
        elif rnd and total:
            pct = min(100.0, 100.0 * rnd / total)
        else:
            pct = 0.0

        cells.append(dict(name=name, arm=arm, seed=name.split("seed")[-1],
                          state=state, elapsed=elapsed, node=node,
                          restarts=restarts, round=rnd, total=total,
                          hv=hv, pct=pct, phase=phase, err=err))

    cells.sort(key=lambda c: (c["arm"], int(c["seed"]) if c["seed"].isdigit() else 0))
    return dict(generated=now.isoformat(timespec="seconds"), cells=cells)


def fmt_dur(s: int | None) -> str:
    if s is None:
        return "-"
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def render(snap: dict) -> str:
    cells = snap["cells"]
    arms: dict[str, list] = {}
    for c in cells:
        arms.setdefault(c["arm"], []).append(c)

    # A successful kubectl that matches nothing renders as 0 complete / 0
    # running / 0%, which reads exactly like a campaign that has not started.
    # Say "no cells" instead of letting zeros speak for themselves.
    empty_banner = ("" if cells else
                    '<div class="empty">kubectl returned NO cells for selector '
                    f'<span class="mono">{html.escape(SELECTOR)}</span> in '
                    f'<span class="mono">{html.escape(NS)}</span>. '
                    'That is an empty listing, not a finished campaign.</div>')

    n_done = sum(c["state"] == "Complete" for c in cells)
    n_run = sum(c["state"] == "Running" for c in cells)
    n_fail = sum(c["state"] == "Failed" for c in cells)
    n_pend = sum(c["state"] == "Pending" for c in cells)
    overall = (sum(c["pct"] for c in cells) / len(cells)) if cells else 0.0

    def chip(state):
        return f'<span class="chip s-{state.lower()}">{state}</span>'

    rows = []
    for arm in sorted(arms):
        group = arms[arm]
        done = sum(c["state"] == "Complete" for c in group)
        apct = sum(c["pct"] for c in group) / len(group)
        rows.append(
            f'<tr class="armhead"><td colspan="7">'
            f'<b>{html.escape(ARM_LABEL.get(arm, arm))}</b>'
            f'<span class="dim"> &nbsp;{done}/{len(group)} complete &nbsp;·&nbsp; {apct:.0f}%</span>'
            f'</td></tr>')
        for c in group:
            prog = (f'{c["round"]}/{c["total"]}' if c["round"] and c["total"]
                    else (html.escape(c["phase"]) or "-"))
            hv = f'{c["hv"]:.4f}' if c["hv"] is not None else "-"
            warn = ""
            if c["err"]:
                warn = f'<span class="chip s-failed">{html.escape(c["err"])}</span>'
            elif c["restarts"]:
                warn = f'<span class="chip s-warn">{c["restarts"]} restart(s)</span>'
            rows.append(
                "<tr>"
                f'<td class="mono">seed {html.escape(c["seed"])}</td>'
                f'<td>{chip(c["state"])} {warn}</td>'
                f'<td class="mono">{prog}</td>'
                f'<td class="bar"><div><i style="width:{c["pct"]:.1f}%"></i></div></td>'
                f'<td class="mono num">{hv}</td>'
                f'<td class="mono">{fmt_dur(c["elapsed"])}</td>'
                f'<td class="mono dim">{html.escape(c["node"] or "-")}</td>'
                "</tr>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Campaign Progress</title>
<style>
:root {{
  --bg:#ffffff; --fg:#14181d; --dim:#6b7280; --line:#e3e6ea; --card:#f7f8fa;
  --ok:#0e8a4f; --run:#1663c7; --fail:#c22a2a; --warn:#9a6200; --bar:#1663c7;
  --barbg:#e3e6ea;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0f1216; --fg:#e8eaed; --dim:#9aa3ad; --line:#262b31; --card:#161a1f;
    --ok:#44c98a; --run:#6aa9ff; --fail:#ff7b7b; --warn:#e0a83a; --bar:#6aa9ff;
    --barbg:#262b31;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0f1216; --fg:#e8eaed; --dim:#9aa3ad; --line:#262b31; --card:#161a1f;
  --ok:#44c98a; --run:#6aa9ff; --fail:#ff7b7b; --warn:#e0a83a; --bar:#6aa9ff;
  --barbg:#262b31;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; }}
h1 {{ font-size:19px; margin:0 0 2px; letter-spacing:-.01em; }}
.sub {{ color:var(--dim); font-size:12.5px; margin-bottom:18px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }}
.dim {{ color:var(--dim); font-weight:400; }}
.num {{ text-align:right; }}
.top {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center;
  margin-bottom:16px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:8px 12px; min-width:84px; }}
.stat b {{ display:block; font-size:19px; font-variant-numeric:tabular-nums; }}
.stat span {{ color:var(--dim); font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; }}
button {{ font:inherit; padding:8px 14px; border-radius:8px; cursor:pointer;
  border:1px solid var(--line); background:var(--card); color:var(--fg); }}
button:hover {{ border-color:var(--run); }}
button[aria-pressed="true"] {{ border-color:var(--run); color:var(--run); }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:7px 8px; border-bottom:1px solid var(--line);
  text-align:left; vertical-align:middle; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--dim); font-weight:600; }}
tr.armhead td {{ background:var(--card); border-top:1px solid var(--line);
  font-size:13px; padding-top:10px; padding-bottom:10px; }}
.chip {{ display:inline-block; padding:1px 7px; border-radius:999px;
  font-size:11px; font-weight:600; border:1px solid currentColor; }}
.s-complete {{ color:var(--ok); }} .s-running {{ color:var(--run); }}
.s-failed {{ color:var(--fail); }} .s-pending,.s-warn {{ color:var(--warn); }}
td.bar {{ width:200px; }}
td.bar>div {{ background:var(--barbg); border-radius:999px; height:7px;
  overflow:hidden; }}
td.bar i {{ display:block; height:100%; background:var(--bar); border-radius:999px;
  transition:width .3s; }}
.foot {{ color:var(--dim); font-size:12px; margin-top:18px; }}
.empty {{ border:1px solid var(--fail); color:var(--fail); border-radius:8px;
  padding:10px 12px; margin:4px 0 14px; font-size:13px; }}
#overall>div {{ background:var(--barbg); border-radius:999px; height:9px;
  overflow:hidden; margin:4px 0 2px; }}
#overall i {{ display:block; height:100%; background:var(--bar); }}
/* The pane is often ~430 px wide. Drop the least load-bearing columns rather
   than letting elapsed and node run off the right edge unseen. */
@media (max-width:860px) {{ th:nth-child(7),td:nth-child(7) {{ display:none; }}
  td.bar {{ width:110px; }} th,td {{ padding:6px 5px; }} }}
@media (max-width:620px) {{ th:nth-child(4),td:nth-child(4) {{ display:none; }} }}
</style></head><body><div class="wrap">

<h1>Chagas rung-1 campaign</h1>
<div class="sub">namespace <span class="mono">lahs-asi</span> &nbsp;·&nbsp;
 generated <span class="mono">{snap["generated"]}</span> UTC</div>

<div class="top">
  <div class="stat"><b>{n_done}</b><span>complete</span></div>
  <div class="stat"><b>{n_run}</b><span>running</span></div>
  <div class="stat"><b>{n_pend}</b><span>pending</span></div>
  <div class="stat"><b>{n_fail}</b><span>failed</span></div>
  <div class="stat"><b>{overall:.0f}%</b><span>overall</span></div>
  <button onclick="reload()">Refresh now</button>
  <button id="auto" aria-pressed="false" onclick="toggle()">Auto-refresh: off</button>
</div>

{empty_banner}
<div id="overall"><div><i style="width:{overall:.1f}%"></i></div></div>

<table>
<thead><tr><th>cell</th><th>state</th><th>round</th><th>progress</th>
<th class="num">last HV</th><th>elapsed</th><th>node</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>

<div class="foot">
Every number here is read from <span class="mono">kubectl</span>, so this is the
cluster's own account and not a prediction. <b>Last HV</b> is the most recent
hypervolume the pod printed, in the shared <span class="mono">[-11, -5]</span>
normalization frame; it is a progress signal, not a result. Rounds per arm:
{html.escape(", ".join(f"{a} {n}" for a, n in sorted(ARM_ROUNDS.items())))}.
The file is rewritten on disk by the watcher, so Refresh picks up new data.
</div>

</div><script>
let t=null;
// Cache-busted so a conditional GET cannot hand back the copy already in the
// tab: the file on disk is rewritten every 30 s and a 304 would freeze the page
// while looking perfectly healthy.
function reload(){{location.replace(location.pathname+'?t='+Date.now());}}
function toggle(){{
  const b=document.getElementById('auto');
  if(t){{clearInterval(t);t=null;b.textContent='Auto-refresh: off';
    b.setAttribute('aria-pressed','false');
    try{{localStorage.removeItem('auto')}}catch(e){{}}}}
  else{{t=setInterval(reload,30000);
    b.textContent='Auto-refresh: 30s';b.setAttribute('aria-pressed','true');
    try{{localStorage.setItem('auto','1')}}catch(e){{}}}}
}}
try{{if(localStorage.getItem('auto'))toggle()}}catch(e){{}}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="nrp/progress.html")
    ap.add_argument("--watch", type=int, default=0,
                    help="rewrite every N seconds (0 = once)")
    ap.add_argument("--max-minutes", type=int, default=720,
                    help="stop watching after this long")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_minutes * 60
    while True:
        try:
            snap = collect()
            out.write_text(render(snap))
            live = sum(c["state"] in ("Running", "Pending") for c in snap["cells"])
            print(f"[{snap['generated']}] wrote {out} ({live} live)", flush=True)
        except Exception as exc:                      # keep the watcher alive
            print(f"[warn] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            live = 1
        if not a.watch or live == 0 or time.time() > deadline:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())

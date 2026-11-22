#!/bin/bash
# Emit one JSON status snapshot of the rung-1 campaign.
export PATH=/opt/homebrew/bin:$PATH
NS=lahs-asi
TOTAL_FULL=50; TOTAL_ASYM=85
python3 - <<'PY'
import subprocess, json, re, datetime, os
NS="lahs-asi"
def kc(*a, t=20):
    # Separate budgets. The pod listing is one slow call and must be allowed to
    # finish, or the snapshot reports zero cells and looks like a dead campaign.
    # The per-pod log reads are many and each must stay short, or 18 of them
    # blow past the caller's timeout and leave a truncated file behind.
    try: return subprocess.run(["kubectl","-n",NS,*a],capture_output=True,text=True,timeout=t).stdout
    except Exception: return ""
pods = kc("get","pods","-l","app=rung1-campaign","--no-headers", t=90)
cells={}
for line in pods.strip().splitlines():
    f=line.split()
    if len(f)<5: continue
    m=re.match(r"rung1-(full|asym|indep)-seed(\d)-",f[0])
    if not m: continue
    key=f"{m.group(1)}_seed{m.group(2)}"
    cells.setdefault(key,{"arm":m.group(1),"seed":int(m.group(2)),
                          "status":"Pending","round":0,"total":85 if m.group(1)=="asym" else 50,
                          "pod":f[0],"restarts":0,"age":f[4]})
    # A finished pod reports "Completed"/"Succeeded", not "Running". Ranking
    # matters: a cell can have an old Error pod and a newer Completed one, and
    # the terminal success is the truth. Without this the first finished cell
    # read as 0 rounds and the campaign total went BACKWARDS.
    rank = {"Completed":3,"Succeeded":3,"Running":2}.get(f[2],1)
    if rank >= {"Complete":3,"Running":2}.get(cells[key].get("status"),0):
        st = "Complete" if f[2] in ("Completed","Succeeded") else f[2]
        cells[key].update(status=st,pod=f[0],restarts=int(f[3]),age=f[4])
        if st=="Complete": cells[key]["round"]=cells[key]["total"]
for k,c in cells.items():
    if c["status"]!="Running": continue
    log=kc("logs",c["pod"],"--tail=900", t=25)
    if not log:
        # A timed-out or empty log read is a FAILED MEASUREMENT, not a reading
        # of zero. Reporting 0 made a cell that was at 84/85 look like it had
        # restarted, and dragged the campaign total backwards. Carry the last
        # known value forward and say the reading is stale.
        c["stale"]=True
        continue
    rounds=re.findall(r"\[Iteration (\d+)\] Training GP",log)
    if rounds: c["round"]=int(rounds[-1])
    if "Docking molecule" in log and not rounds:
        m=re.findall(r"Docking molecule (\d+)/(\d+)",log)
        if m: c["phase"]=f"initial {m[-1][0]}/{m[-1][1]}"
    if re.search(r"pareto_front|\[done\]|rc=0",log): c["status"]="Complete"
# Seed each cell's round from the previous snapshot, so a cell whose log read
# fails keeps its last known progress instead of reporting zero.
PREV=os.environ.get("PREV_SNAPSHOT","")
if PREV and os.path.exists(PREV):
    try:
        prev={(c["arm"],c["seed"]):c for c in json.load(open(PREV)).get("cells",[])}
        for k,c in cells.items():
            o=prev.get((c["arm"],c["seed"]))
            if o and c["round"] < o.get("round",0): c["round"]=o["round"]
    except Exception: pass

snap={"updated":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
      "cells":sorted(cells.values(),key=lambda c:(c["arm"],c["seed"]))}
done=sum(1 for c in snap["cells"] if c["status"]=="Complete")
snap["summary"]={"total":len(snap["cells"]),"running":sum(1 for c in snap["cells"] if c["status"]=="Running"),
                 "complete":done,"failed":sum(1 for c in snap["cells"] if c["status"] in("Error","Failed")),
                 "rounds_done":sum(c["round"] for c in snap["cells"]),
                 "rounds_total":sum(c["total"] for c in snap["cells"])}
if not snap["cells"]:
    import sys; sys.exit("collector: kubectl returned no pods; refusing to emit an empty snapshot")
print(json.dumps(snap,indent=2))
PY

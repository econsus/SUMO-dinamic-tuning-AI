"""
Step B: Warm-up contamination test.
Simulate the env's full lifecycle (warm-up + 4 RL steps) and count vehicles EVERY phase
to see how warm-up vehicles bleed into RL step 1.
"""
import os, sys, datetime, json, tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from src.data_repo import DataRepo
from src.data.traffic_data import import_records, data_records

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

STEP_LENGTH = 0.05
MINUTE_STEPS = int(60.0 / STEP_LENGTH)
WARMUP_MINUTES = 5
WARMUP_STEPS = MINUTE_STEPS * WARMUP_MINUTES
PERIOD_MINUTES = 5
PERIOD_STEPS = MINUTE_STEPS * PERIOD_MINUTES
STEPS_PER_DATA = 4

ROU_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="f_sl"  begin="0.00" from="675098743#0" to="675098743#2" end="3600.00" vehsPerHour="{sl}"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2" end="3600.00" vehsPerHour="{spt}"/>
    <flow id="f_ul"  begin="0.00" color="50,255,0" from="585303113#0" to="585303113#2" end="3600.00" vehsPerHour="{ul}"/>
    <flow id="f_upt" begin="0.00" color="255,53,0" from="585303113#0" to="675098743#2" end="3600.00" vehsPerHour="{upt}"/>
</routes>"""

NORTH_LOOPS = ["total_north_left", "total_north_right"]
SOUTH_LOOPS = ["total_south_left", "total_south_right"]
ALL_LOOPS = NORTH_LOOPS + SOUTH_LOOPS

def force_kill_sumo():
    try:
        traci.connection._connections.pop("default", None)
        traci.connection._connections.pop("", None)
    except Exception:
        pass
    for exe in ("sumo.exe", "sumo-gui.exe", "sumo", "sumo-gui"):
        try:
            os.system(f"taskkill /F /IM {exe} 2>nul")
        except Exception:
            pass

import_records()
rows = []
for row in data_records:
    vals = {}
    for d in row:
        vals.update(d)
    sl = vals.get("SL") or 0
    spt = vals.get("SPT") or 0
    ul = vals.get("UL") or 0
    upt = vals.get("UPT") or 0
    os_val = vals.get("OS") or 0
    ou_val = vals.get("OU") or 0
    if os_val == 0 or ou_val == 0:
        continue
    rows.append((sl, spt, ul, upt, os_val, ou_val))

print(f"Loaded {len(rows)} data point(s)")

all_phases = []

for data_idx, (sl, spt, ul, upt, os_val, ou_val) in enumerate(rows):
    print(f"\n{'='*60}")
    print(f"Data point {data_idx}: SL={sl} SPT={spt} UL={ul} UPT={upt} OS={os_val} OU={ou_val}")

    tmpdir = tempfile.mkdtemp(prefix="stepB_")
    rou_path = os.path.join(tmpdir, "flows.rou.xml")
    with open(rou_path, "w") as f:
        f.write(ROU_XML_TEMPLATE.format(sl=sl, spt=spt, ul=ul, upt=upt))

    repo = DataRepo()
    cmd = [
        "sumo",
        "-c", str(repo.sumo_config_path),
        "--route-files", os.path.abspath(rou_path),
        "--step-length", str(STEP_LENGTH),
        "--no-warnings",
        "--start",
    ]

    try:
        traci.start(cmd)

        # Phase 1: Warm-up — count vehicles too
        warmup_counts = {lid: set() for lid in ALL_LOOPS}
        for _ in range(WARMUP_STEPS):
            traci.simulationStep()
            for lid in ALL_LOOPS:
                warmup_counts[lid].update(traci.inductionloop.getLastStepVehicleIDs(lid))
        w_north = len(warmup_counts["total_north_left"] | warmup_counts["total_north_right"])
        w_south = len(warmup_counts["total_south_left"] | warmup_counts["total_south_right"])
        print(f"  Warm-up counts: north={w_north} south={w_south}")
        for lid in ALL_LOOPS:
            print(f"    {lid}: {len(warmup_counts[lid])}")
        all_phases.append({
            "data_point": data_idx, "phase": "warmup",
            "north": w_north, "south": w_south,
            "per_loop": {lid: len(v) for lid, v in warmup_counts.items()},
        })

        # Remove warm-up vehicles so they don't bleed into RL steps
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.remove(veh_id)

        # Phase 2-5: RL steps
        for step_i in range(STEPS_PER_DATA):
            step_counts = {lid: set() for lid in ALL_LOOPS}
            for _ in range(PERIOD_STEPS):
                traci.simulationStep()
                for lid in ALL_LOOPS:
                    step_counts[lid].update(traci.inductionloop.getLastStepVehicleIDs(lid))
            s_north = len(step_counts["total_north_left"] | step_counts["total_north_right"])
            s_south = len(step_counts["total_south_left"] | step_counts["total_south_right"])
            exp_north = ou_val / 12.0
            exp_south = os_val / 12.0
            print(f"  RL step {step_i+1}: north={s_north} (exp={exp_north:.1f}) "
                  f"south={s_south} (exp={exp_south:.1f})")
            for lid in ALL_LOOPS:
                print(f"    {lid}: {len(step_counts[lid])}")
            all_phases.append({
                "data_point": data_idx, "phase": f"rl_step_{step_i+1}",
                "north": s_north, "south": s_south,
                "expected_north": exp_north, "expected_south": exp_south,
                "per_loop": {lid: len(v) for lid, v in step_counts.items()},
            })

    finally:
        force_kill_sumo()

ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"stepB_{ts}.json")
with open(log_path, "w") as f:
    json.dump(all_phases, f, indent=2)
print(f"\nLog saved to {log_path}")

print("\n=== WARMUP vs RL STEP 1 ===")
prev_phase = None
for p in all_phases:
    label = f"DP{p['data_point']} {p['phase']:12s}"
    print(f"  {label}  north={p['north']:4d}  south={p['south']:4d}")
    if "rl_step_1" in p["phase"] and prev_phase and "warmup" in prev_phase["phase"]:
        bleed_north = p["north"] - (sum(
            x["north"] for x in all_phases
            if x["data_point"] == p["data_point"] and "warmup" not in x["phase"] and x != p
        ))  # Simplified: just compare warmup vs first rl step
    prev_phase = p

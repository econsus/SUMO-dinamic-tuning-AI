"""
Step D: Vehicle-to-loop mapping test.
Track which vehicle IDs cross which induction loops during simulation.
This reveals if flows are going through the expected loops.
"""
import os, sys, datetime, json, tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from src.data_repo import DataRepo

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

STEP_LENGTH = 0.05
WARMUP_STEPS = int(60.0 / STEP_LENGTH) * 5
PERIOD_STEPS = int(60.0 / STEP_LENGTH) * 5

ROU_XML = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="f_sl"  begin="0.00" from="675098743#0" to="675098743#2" end="3600.00" vehsPerHour="390"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2" end="3600.00" vehsPerHour="204"/>
    <flow id="f_ul"  begin="0.00" color="50,255,0" from="585303113#0" to="585303113#2" end="3600.00" vehsPerHour="384"/>
    <flow id="f_upt" begin="0.00" color="255,53,0" from="585303113#0" to="675098743#2" end="3600.00" vehsPerHour="36"/>
</routes>"""

ALL_LOOPS = ["total_north_left", "total_north_right",
             "total_south_left", "total_south_right"]

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

tmpdir = tempfile.mkdtemp(prefix="stepD_")
rou_path = os.path.join(tmpdir, "flows.rou.xml")
with open(rou_path, "w") as f:
    f.write(ROU_XML)

repo = DataRepo()
cmd = [
    "sumo",
    "-c", str(repo.sumo_config_path),
    "--route-files", os.path.abspath(rou_path),
    "--step-length", str(STEP_LENGTH),
    "--no-warnings",
    "--start",
]

loop_vehicles = {lid: set() for lid in ALL_LOOPS}

try:
    traci.start(cmd)

    # Run warm-up + 1 RL period (10 minutes total = 12000 steps)
    total_steps = WARMUP_STEPS + PERIOD_STEPS
    for step in range(total_steps):
        traci.simulationStep()
        for lid in ALL_LOOPS:
            veh_ids = traci.inductionloop.getLastStepVehicleIDs(lid)
            # getLastStepVehicleIDs returns list of vehicle IDs on this loop
            if isinstance(veh_ids, (list, tuple)):
                loop_vehicles[lid].update(veh_ids)

except Exception as e:
    print(f"Error: {e}")
finally:
    force_kill_sumo()

print(f"\n=== Vehicle-Loop Mapping (over {total_steps} steps = 10 sim-min) ===")
for lid in ALL_LOOPS:
    vehs = loop_vehicles[lid]
    print(f"  {lid:20s}: {len(vehs):3d} unique vehicles")

# Cross-reference: check which flow IDs produced which vehicles
# Vehicle IDs in SUMO are flow-based: flow_XXX where XXX is a counter
# f_sl_0, f_sl_1, ... f_upt_15, etc.
flow_loop_map = {}
for lid in ALL_LOOPS:
    for veh_id in loop_vehicles[lid]:
        # Vehicle ID format: {flow_id}_{number}
        parts = veh_id.rsplit(".", 1)
        if len(parts) == 2:
            flow_prefix = parts[0]
            if flow_prefix not in flow_loop_map:
                flow_loop_map[flow_prefix] = {l: set() for l in ALL_LOOPS}
            flow_loop_map[flow_prefix][lid].add(veh_id)

print("\n  Flow -> Loop mapping:")
for flow_id in ["f_sl", "f_spt", "f_ul", "f_upt"]:
    entry = flow_loop_map.get(flow_id, {})
    north = len(entry.get("total_north_left", set())) + len(entry.get("total_north_right", set()))
    south = len(entry.get("total_south_left", set())) + len(entry.get("total_south_right", set()))
    total = north + south
    print(f"    {flow_id:5s}: north_loops={north:3d}  south_loops={south:3d}  (total={total} vehicles)")

    # Per-loop breakdown
    for lid in ALL_LOOPS:
        count = len(entry.get(lid, set()))
        if count > 0:
            print(f"      -> {lid}: {count} vehicles")

ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"stepD_{ts}.json")
with open(log_path, "w") as f:
    json.dump({
        "total_steps": total_steps,
        "loop_vehicle_counts": {lid: len(v) for lid, v in loop_vehicles.items()},
        "flow_loop_map": {
            fid: {lid: list(vehs) for lid, vehs in loops.items()}
            for fid, loops in flow_loop_map.items()
        },
    }, f, indent=2, default=str)
print(f"\nLog saved to {log_path}")

"""
Step A: Counting sanity test.
Start SUMO with flows at known vehsPerHour, run 5 min, compare expected vs actual loop counts.
Expected: each flow at 360 vehs/hour → ~30 vehicles per 5 min per relevant loop.
"""
import os, sys, tempfile, datetime

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
PERIOD_MINUTES = 5
PERIOD_STEPS = int(60.0 / STEP_LENGTH) * PERIOD_MINUTES  # 6000
KNOWN_RATE = 360  # vehs/hour → 30 in 5 min

def make_route(sl, spt, ul, upt):
    flows = []
    def add_flow(fid, frm, to, rate, color=None):
        if rate > 0:
            color_attr = f' color="{color}"' if color else ""
            flows.append(f'    <flow id="{fid}" begin="0.00"{color_attr} from="{frm}" to="{to}" end="3600.00" vehsPerHour="{int(rate)}"/>')
    add_flow("f_sl",  "675098743#0", "675098743#2", sl)
    add_flow("f_spt", "675098743#0", "585303113#2", spt)
    add_flow("f_ul",  "585303113#0", "585303113#2", ul, "50,255,0")
    add_flow("f_upt", "585303113#0", "675098743#2", upt, "255,53,0")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n' + "\n".join(flows) + "\n</routes>"

NORTH_LOOPS = ["total_north_left", "total_north_right"]
SOUTH_LOOPS = ["total_south_left", "total_south_right"]
ALL_LOOPS = NORTH_LOOPS + SOUTH_LOOPS

def run_test(test_label, sl, spt, ul, upt):
    print(f"\n{'='*60}")
    print(f"Test: {test_label}")
    print(f"  Flows: SL={sl}  SPT={spt}  UL={ul}  UPT={upt} (vehs/hour)")
    expected_north = (sl + upt) / 12.0
    expected_south = (spt + ul) / 12.0
    print(f"  Expected in {PERIOD_MINUTES} min: north={expected_north:.1f}  south={expected_south:.1f}")

    tmpdir = tempfile.mkdtemp(prefix="stepA_")
    rou_path = os.path.join(tmpdir, "flows.rou.xml")
    with open(rou_path, "w") as f:
        f.write(make_route(sl, spt, ul, upt))

    repo = DataRepo()
    cmd = [
        "sumo",
        "-c", str(repo.sumo_config_path),
        "--route-files", os.path.abspath(rou_path),
        "--step-length", str(STEP_LENGTH),
        "--no-warnings",
        "--start",
    ]

    counts = {lid: set() for lid in ALL_LOOPS}
    try:
        traci.start(cmd)
        for step in range(PERIOD_STEPS):
            traci.simulationStep()
            for lid in ALL_LOOPS:
                counts[lid].update(traci.inductionloop.getLastStepVehicleIDs(lid))
    finally:
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

    sim_north = len(counts["total_north_left"] | counts["total_north_right"])
    sim_south = len(counts["total_south_left"] | counts["total_south_right"])
    rn = f"{sim_north/expected_north:.1f}x" if expected_north > 0 else "N/A"
    rs = f"{sim_south/expected_south:.1f}x" if expected_south > 0 else "N/A"
    print(f"  Simulated: north={sim_north}  south={sim_south}")
    print(f"  Ratio: north={rn}  south={rs}")
    for lid in ALL_LOOPS:
        print(f"    {lid}: {len(counts[lid])}")

    return {
        "test": test_label,
        "flows": {"SL": sl, "SPT": spt, "UL": ul, "UPT": upt},
        "expected_north": expected_north,
        "expected_south": expected_south,
        "sim_north": sim_north,
        "sim_south": sim_south,
        "ratio_north": round(sim_north / expected_north, 1) if expected_north > 0 else None,
        "ratio_south": round(sim_south / expected_south, 1) if expected_south > 0 else None,
        "per_loop": {lid: len(v) for lid, v in counts.items()},
    }

results = []
results.append(run_test("All 4 flows at 360 each", 360, 360, 360, 360))
results.append(run_test("Only SL=360, others 0", 360, 0, 0, 0))
results.append(run_test("Only SPT=360, others 0", 0, 360, 0, 0))
results.append(run_test("Only UL=360, others 0", 0, 0, 360, 0))
results.append(run_test("Only UPT=360, others 0", 0, 0, 0, 360))

# Write summary log
import json
ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"stepA_{ts}.json")
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nLog saved to {log_path}")

print("\n=== SUMMARY ===")
for r in results:
    rn_str = f"x{r['ratio_north']:5.1f}" if r['ratio_north'] is not None else "N/A    "
    rs_str = f"x{r['ratio_south']:5.1f}" if r['ratio_south'] is not None else "N/A    "
    print(f"  {r['test']:30s}  north={r['sim_north']:4d} ({rn_str})  "
          f"south={r['sim_south']:4d} ({rs_str})")

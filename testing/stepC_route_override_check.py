"""
Step C: Route-files override test.
Check whether --route-files properly REPLACES the config's route-files or APPENDS.
Detection: compare departed vehicle count + vehicle ID prefixes against expected.
"""
import os, sys, datetime, json, tempfile
import xml.etree.ElementTree as ET

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
RUN_STEPS = 2000  # 100 sim-seconds

OUR_RATES = {"f_sl": 777, "f_spt": 888, "f_ul": 999, "f_upt": 111}
OUR_TOTAL_RATE = sum(OUR_RATES.values())  # 2775

ROU_XML = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="f_sl"  begin="0.00" from="675098743#0" to="675098743#2" end="3600.00" vehsPerHour="777"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2" end="3600.00" vehsPerHour="888"/>
    <flow id="f_ul"  begin="0.00" color="50,255,0" from="585303113#0" to="585303113#2" end="3600.00" vehsPerHour="999"/>
    <flow id="f_upt" begin="0.00" color="255,53,0" from="585303113#0" to="675098743#2" end="3600.00" vehsPerHour="111"/>
</routes>"""

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

# Read original route file for reference
original_rou_path = os.path.join(_proj_root, "data", "sumo_files", "map_suhat_netedit.rou.xml")
original_flows = {}
tree = ET.parse(original_rou_path)
for f in tree.iter("flow"):
    original_flows[f.get("id")] = int(f.get("vehsPerHour"))

print("Original route file flows:")
for fid, rate in original_flows.items():
    print(f"  {fid}: vehsPerHour={rate}")
print(f"  Total rate: {sum(original_flows.values())} veh/hr")
print(f"Our override total rate: {OUR_TOTAL_RATE} veh/hr")

# Expected vehicles in 100 sim-seconds at each rate
our_expected = OUR_TOTAL_RATE / 3600 * 100
orig_expected = sum(original_flows.values()) / 3600 * 100
print(f"\nExpected departed in 100s: override={our_expected:.1f}  original={orig_expected:.1f}")

# Run SUMO with temp route
tmpdir = tempfile.mkdtemp(prefix="stepC_")
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

results = {"vehicle_ids": []}
try:
    traci.start(cmd)
    for _ in range(RUN_STEPS):
        traci.simulationStep()

    departed = traci.simulation.getDepartedNumber()
    loaded = traci.simulation.getLoadedNumber()
    veh_ids = traci.vehicle.getIDList()

    results["departed"] = departed
    results["loaded"] = loaded
    results["vehicle_count"] = len(veh_ids)
    results["vehicle_ids"] = list(veh_ids)

    print(f"\nAfter {RUN_STEPS} steps ({RUN_STEPS * STEP_LENGTH:.0f} sim-seconds):")
    print(f"  Loaded: {loaded}")
    print(f"  Departed: {departed}")
    print(f"  Active vehicles: {len(veh_ids)}")

    # Analyze vehicle ID prefixes
    prefixes = {}
    for vid in veh_ids:
        prefix = vid.rsplit(".", 1)[0] if "." in vid else vid
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    print(f"\n  Vehicle ID prefixes: {prefixes}")

    # Check for any non-override prefixes
    our_prefixes = set(OUR_RATES.keys())
    unknown_prefixes = {p for p in prefixes if p not in our_prefixes}
    results["prefix_counts"] = prefixes
    results["unknown_prefixes"] = list(unknown_prefixes)

    if unknown_prefixes:
        print(f"  UNEXPECTED prefixes (from original routes?): {unknown_prefixes}")
    else:
        print(f"  All vehicle prefixes match our override flows")

except Exception as e:
    print(f"Error: {e}")
    results["error"] = str(e)
finally:
    force_kill_sumo()

ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"stepC_{ts}.json")
with open(log_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nLog saved to {log_path}")

print("\n=== VERDICT ===")
actual = results.get("departed", 0)
# Allow margin: override should give ~77, original ~200
if actual < our_expected * 1.5:
    print(f"  Departed={actual} ~= expected for override ({our_expected:.0f}) -> OVERRIDE WORKS")
else:
    print(f"  Departed={actual} >> expected for override ({our_expected:.0f}) -> APPENDING? (original would give {orig_expected:.0f})")

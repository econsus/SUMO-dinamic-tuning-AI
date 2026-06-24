"""
Step E: Verify SUMO lane-changing parameters propagate and affect simulation output.

Test A — Key name verification
  Set param with different key formats, read back, see which stores/retrieves.

Test B — Extreme parameter comparison (env-accurate: sets on existing vehicles only)
  Run 5 min with (assertive=0.01, coop=0.01) vs (assertive=5.0, coop=5.0).

Test B+ — Same but sets on ALL vehicles (including newly spawned mid-period)
  Tests whether the fix "set params on every new vehicle" changes anything.

Test C — Positive control: speedFactor (has proper TraCI API)
  Confirms the test methodology works.
"""
import os, sys, json, datetime, tempfile

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
PERIOD_STEPS = int(60.0 / STEP_LENGTH) * PERIOD_MINUTES
NORTH_LOOPS = ["total_north_left", "total_north_right"]
SOUTH_LOOPS = ["total_south_left", "total_south_right"]
ALL_LOOPS = NORTH_LOOPS + SOUTH_LOOPS

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

def start_sumo(route_xml=None):
    tmpdir = tempfile.mkdtemp(prefix="stepE_")
    rou_path = os.path.join(tmpdir, "flows.rou.xml")
    with open(rou_path, "w") as f:
        f.write(route_xml or make_route(360, 360, 360, 360))
    repo = DataRepo()
    cmd = [
        "sumo",
        "-c", str(repo.sumo_config_path),
        "--route-files", os.path.abspath(rou_path),
        "--step-length", str(STEP_LENGTH),
        "--no-warnings",
        "--start",
    ]
    traci.start(cmd)
    return tmpdir, rou_path

def stop_sumo():
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


# ============================================================
# TEST A — Key name verification
# ============================================================
def test_a():
    print("\n" + "="*60)
    print("TEST A: Parameter key name verification")
    print("="*60)

    results = {}

    tmpdir, _ = start_sumo(make_route(60, 60, 60, 60))

    # Let vehicles start spawning
    for _ in range(10):
        traci.simulationStep()

    # ---- A1: Set on DEFAULT_VEHTYPE with "laneChangeModel." prefix ----
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative", "3.14")
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive", "2.71")

    read_coop_lcm = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative")
    read_assert_lcm = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive")
    read_coop_plain = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "lcCooperative")
    read_assert_plain = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "lcAssertive")

    print("\n  Set on DEFAULT_VEHTYPE with key 'laneChangeModel.lcXxx' = (3.14, 2.71)")
    print(f"    Read back 'laneChangeModel.lcCooperative': [{read_coop_lcm}]")
    print(f"    Read back 'laneChangeModel.lcAssertive':   [{read_assert_lcm}]")
    print(f"    Read back 'lcCooperative' (no prefix):     [{read_coop_plain}]")
    print(f"    Read back 'lcAssertive' (no prefix):       [{read_assert_plain}]")

    results["A1_set_on_type_with_prefix"] = {
        "set": {"laneChangeModel.lcCooperative": "3.14", "laneChangeModel.lcAssertive": "2.71"},
        "read_laneChangeModel.lcCooperative": read_coop_lcm,
        "read_laneChangeModel.lcAssertive": read_assert_lcm,
        "read_lcCooperative": read_coop_plain,
        "read_lcAssertive": read_assert_plain,
    }

    # ---- A2: Set on DEFAULT_VEHTYPE WITHOUT prefix ----
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "lcCooperative", "1.23")
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "lcAssertive", "4.56")

    read_coop_plain2 = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "lcCooperative")
    read_assert_plain2 = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "lcAssertive")
    read_coop_lcm2 = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative")
    read_assert_lcm2 = traci.vehicletype.getParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive")

    print("\n  Set on DEFAULT_VEHTYPE with key 'lcXxx' (no prefix) = (1.23, 4.56)")
    print(f"    Read back 'lcCooperative':                       [{read_coop_plain2}]")
    print(f"    Read back 'lcAssertive':                         [{read_assert_plain2}]")
    print(f"    Read back 'laneChangeModel.lcCooperative':       [{read_coop_lcm2}]")
    print(f"    Read back 'laneChangeModel.lcAssertive':         [{read_assert_lcm2}]")

    results["A2_set_on_type_without_prefix"] = {
        "set": {"lcCooperative": "1.23", "lcAssertive": "4.56"},
        "read_lcCooperative": read_coop_plain2,
        "read_lcAssertive": read_assert_plain2,
        "read_laneChangeModel.lcCooperative": read_coop_lcm2,
        "read_laneChangeModel.lcAssertive": read_assert_lcm2,
    }

    # ---- A3: Set on an existing vehicle ----
    veh_ids = traci.vehicle.getIDList()
    if veh_ids:
        vid = veh_ids[0]
        traci.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", "7.77")
        traci.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", "8.88")

        veh_read_lcm_c = traci.vehicle.getParameter(vid, "laneChangeModel.lcCooperative")
        veh_read_lcm_a = traci.vehicle.getParameter(vid, "laneChangeModel.lcAssertive")
        veh_read_plain_c = traci.vehicle.getParameter(vid, "lcCooperative")
        veh_read_plain_a = traci.vehicle.getParameter(vid, "lcAssertive")

        print(f"\n  Set on vehicle '{vid}' with 'laneChangeModel.lcXxx' = (7.77, 8.88)")
        print(f"    Read back 'laneChangeModel.lcCooperative': [{veh_read_lcm_c}]")
        print(f"    Read back 'laneChangeModel.lcAssertive':   [{veh_read_lcm_a}]")
        print(f"    Read back 'lcCooperative' (no prefix):     [{veh_read_plain_c}]")
        print(f"    Read back 'lcAssertive' (no prefix):       [{veh_read_plain_a}]")

        results["A3_set_on_vehicle_with_prefix"] = {
            "vehicle": vid,
            "set": {"laneChangeModel.lcCooperative": "7.77", "laneChangeModel.lcAssertive": "8.88"},
            "read_laneChangeModel.lcCooperative": veh_read_lcm_c,
            "read_laneChangeModel.lcAssertive": veh_read_lcm_a,
            "read_lcCooperative": veh_read_plain_c,
            "read_lcAssertive": veh_read_plain_a,
        }
    else:
        print("\n  No vehicles yet - try again")
        results["A3_set_on_vehicle_with_prefix"] = {"error": "no vehicles"}

    # ---- A4: Spawn a NEW vehicle after setting DEFAULT_VEHTYPE ----
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative", "9.99")
    traci.vehicletype.setParameter("DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive", "1.11")

    existing_ids = set(traci.vehicle.getIDList())
    found_new = False
    for _ in range(200):
        traci.simulationStep()
        current_ids = set(traci.vehicle.getIDList())
        new_vids = current_ids - existing_ids
        if new_vids:
            nvid = list(new_vids)[0]
            new_read_c = traci.vehicle.getParameter(nvid, "laneChangeModel.lcCooperative")
            new_read_a = traci.vehicle.getParameter(nvid, "laneChangeModel.lcAssertive")
            new_read_cp = traci.vehicle.getParameter(nvid, "lcCooperative")
            new_read_ap = traci.vehicle.getParameter(nvid, "lcAssertive")
            print(f"\n  New vehicle '{nvid}' spawned AFTER setting DEFAULT_VEHTYPE:")
            print(f"    Read 'laneChangeModel.lcCooperative': [{new_read_c}]")
            print(f"    Read 'laneChangeModel.lcAssertive':   [{new_read_a}]")
            print(f"    Read 'lcCooperative' (no prefix):     [{new_read_cp}]")
            print(f"    Read 'lcAssertive' (no prefix):       [{new_read_ap}]")
            results["A4_new_vehicle_inherits"] = {
                "vehicle": nvid,
                "read_laneChangeModel.lcCooperative": new_read_c,
                "read_laneChangeModel.lcAssertive": new_read_a,
                "read_lcCooperative": new_read_cp,
                "read_lcAssertive": new_read_ap,
            }
            found_new = True
            break
    if not found_new:
        print("\n  ERROR: No new vehicle spawned after 200 steps")
        results["A4_new_vehicle_inherits"] = {"error": "no new vehicle spawned"}

    stop_sumo()

    # Verdict
    print("\n  --- VERDICT A ---")
    if results.get("A4_new_vehicle_inherits", {}).get("read_laneChangeModel.lcCooperative") == "9.99":
        print("  New vehicles DO inherit from DEFAULT_VEHTYPE (setParameter works!)")
        results["verdict"] = "NEW_VEHICLES_INHERIT"
    else:
        print("  New vehicles DO NOT inherit from DEFAULT_VEHTYPE")
        print("  -> setParameter on vType creates generic <param> not attribute inheritance")
        results["verdict"] = "NEW_VEHICLES_DO_NOT_INHERIT"

    return results


# ============================================================
# TEST B — Extreme params (env-accurate: existing vehicles only)
# ============================================================
def run_sim(label, lc_coop, lc_assert, set_on_all_vehicles, sl=360, spt=360, ul=360, upt=360):
    """If set_on_all_vehicles=True, set params on every vehicle (incl. mid-period spawns)."""
    print(f"\n  --- {label} ---")
    print(f"    lcCooperative={lc_coop}  lcAssertive={lc_assert}  set_on_all={set_on_all_vehicles}")
    tmpdir, _ = start_sumo(make_route(sl, spt, ul, upt))

    # Set on DEFAULT_VEHTYPE (both key formats)
    for key in ["laneChangeModel.lcCooperative", "lcCooperative"]:
        traci.vehicletype.setParameter("DEFAULT_VEHTYPE", key, str(lc_coop))
    for key in ["laneChangeModel.lcAssertive", "lcAssertive"]:
        traci.vehicletype.setParameter("DEFAULT_VEHTYPE", key, str(lc_assert))

    # Set on already-existing vehicles
    for vid in traci.vehicle.getIDList():
        traci.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", str(lc_coop))
        traci.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", str(lc_assert))

    seen_ids = set(traci.vehicle.getIDList())
    north_ids = set()
    south_ids = set()

    for step in range(PERIOD_STEPS):
        traci.simulationStep()

        # Optionally set on all vehicles (including new spawns)
        if set_on_all_vehicles:
            for vid in traci.vehicle.getIDList():
                traci.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", str(lc_coop))
                traci.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", str(lc_assert))

        for lid in NORTH_LOOPS:
            north_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))
        for lid in SOUTH_LOOPS:
            south_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))

    sim_north = len(north_ids)
    sim_south = len(south_ids)
    print(f"    sim_north={sim_north}  sim_south={sim_south}")

    stop_sumo()
    return {
        "label": label,
        "lcCooperative": lc_coop,
        "lcAssertive": lc_assert,
        "set_on_all_vehicles": set_on_all_vehicles,
        "sim_north": sim_north,
        "sim_south": sim_south,
    }


def test_b():
    print("\n" + "="*60)
    print("TEST B: Extreme params comparison")
    print("="*60)

    variants = [
        ("Timid (env-accurate: existing only)", 0.01, 0.01, False),
        ("Assertive (env-accurate: existing only)", 5.0, 5.0, False),
    ]

    results = []
    for label, coop, assertv, all_v in variants:
        results.append(run_sim(label, coop, assertv, all_v))

    r1, r2 = results[0], results[1]
    dn = r2["sim_north"] - r1["sim_north"]
    ds = r2["sim_south"] - r1["sim_south"]
    print(f"\n  Difference (Assertive - Timid): north={dn:+d}  south={ds:+d}")
    if dn == 0 and ds == 0:
        print("  -> ZERO effect (params don't change sim output)")
    else:
        print("  -> Params DO affect sim counts")
    results.append({"label": "diff_assertive_minus_timid", "delta_north": dn, "delta_south": ds})

    return results


# ============================================================
# TEST B+ — Same but set on ALL vehicles (incl. mid-period spawns)
# ============================================================
def test_bplus():
    print("\n" + "="*60)
    print("TEST B+: All-vehicle setting (fix for inheritance bug)")
    print("="*60)

    variants = [
        ("Timid (ALL vehicles)", 0.01, 0.01, True),
        ("Assertive (ALL vehicles)", 5.0, 5.0, True),
    ]

    results = []
    for label, coop, assertv, all_v in variants:
        results.append(run_sim(label, coop, assertv, all_v))

    r1, r2 = results[0], results[1]
    dn = r2["sim_north"] - r1["sim_north"]
    ds = r2["sim_south"] - r1["sim_south"]
    print(f"\n  Difference (Assertive - Timid): north={dn:+d}  south={ds:+d}")
    if dn == 0 and ds == 0:
        print("  -> STILL zero effect even when ALL vehicles get the params")
    else:
        print("  -> FIXED! Setting on all vehicles makes the params work")
    results.append({"label": "diff_assertive_minus_timid_all", "delta_north": dn, "delta_south": ds})

    return results


# ============================================================
# TEST C — Positive control: speedFactor
# ============================================================
def test_c():
    print("\n" + "="*60)
    print("TEST C: Positive control - speedFactor")
    print("="*60)

    results = []

    for sf, label in [(0.5, "Slow (speedFactor=0.5)"), (2.0, "Fast (speedFactor=2.0)")]:
        print(f"\n  --- {label} ---")
        tmpdir, _ = start_sumo(make_route(360, 360, 360, 360))

        seen = set()
        north_ids = set()
        south_ids = set()

        for _ in range(PERIOD_STEPS):
            traci.simulationStep()

            # Set speedFactor on ALL vehicles (including new spawns)
            for vid in traci.vehicle.getIDList():
                traci.vehicle.setSpeedFactor(vid, sf)
                seen.add(vid)

            for lid in NORTH_LOOPS:
                north_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))
            for lid in SOUTH_LOOPS:
                south_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))

        sim_north = len(north_ids)
        sim_south = len(south_ids)
        print(f"    sim_north={sim_north}  sim_south={sim_south}")

        stop_sumo()
        results.append({
            "label": label,
            "speedFactor": sf,
            "sim_north": sim_north,
            "sim_south": sim_south,
        })

    if len(results) >= 2:
        dn = results[1]["sim_north"] - results[0]["sim_north"]
        ds = results[1]["sim_south"] - results[0]["sim_south"]
        print(f"\n  Difference (Fast - Slow): north={dn:+d}  south={ds:+d}")
        if dn != 0 or ds != 0:
            print("  -> speedFactor DOES change sim output (methodology works)")
        else:
            print("  -> speedFactor also has no effect (something else is wrong?)")
        results.append({"label": "diff_fast_minus_slow", "delta_north": dn, "delta_south": ds})

    return results


# ============================================================
# TEST D — XML attribute approach (the fix)
# ============================================================
ROU_XML_VTYPE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="DEFAULT_VEHTYPE" lcCooperative="{coop}" lcAssertive="{assertv}"/>
    <flow id="f_sl" begin="0.00" from="675098743#0" to="675098743#2" end="3600.00" vehsPerHour="{sl}"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2" end="3600.00" vehsPerHour="{spt}"/>
    <flow id="f_ul" begin="0.00" color="50,255,0" from="585303113#0" to="585303113#2" end="3600.00" vehsPerHour="{ul}"/>
    <flow id="f_upt" begin="0.00" color="255,53,0" from="585303113#0" to="675098743#2" end="3600.00" vehsPerHour="{upt}"/>
</routes>"""

def make_route_with_vtype(sl, spt, ul, upt, coop, assertv):
    return ROU_XML_VTYPE_TEMPLATE.format(sl=sl, spt=spt, ul=ul, upt=upt, coop=coop, assertv=assertv)


def run_with_xml_attrs(label, lc_coop, lc_assert, sl=360, spt=360, ul=360, upt=360):
    print(f"\n  --- {label} ---")
    print(f"    lcCooperative={lc_coop}  lcAssertive={lc_assert} (as XML attributes)")
    route_xml = make_route_with_vtype(sl, spt, ul, upt, lc_coop, lc_assert)
    tmpdir, _ = start_sumo(route_xml)

    north_ids = set()
    south_ids = set()

    for _ in range(PERIOD_STEPS):
        traci.simulationStep()
        for lid in NORTH_LOOPS:
            north_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))
        for lid in SOUTH_LOOPS:
            south_ids.update(traci.inductionloop.getLastStepVehicleIDs(lid))

    sim_north = len(north_ids)
    sim_south = len(south_ids)
    print(f"    sim_north={sim_north}  sim_south={sim_south}")

    stop_sumo()
    return {
        "label": label,
        "lcCooperative": lc_coop,
        "lcAssertive": lc_assert,
        "sim_north": sim_north,
        "sim_south": sim_south,
    }


def test_d():
    print("\n" + "="*60)
    print("TEST D: XML attribute approach (vType in route)")
    print("="*60)

    results = []

    r1 = run_with_xml_attrs("Timid via XML (0.01, 0.01)", 0.01, 0.01)
    r2 = run_with_xml_attrs("Assertive via XML (5.0, 5.0)", 5.0, 5.0)

    results.append(r1)
    results.append(r2)

    dn = r2["sim_north"] - r1["sim_north"]
    ds = r2["sim_south"] - r1["sim_south"]
    print(f"\n  Difference (Assertive - Timid): north={dn:+d}  south={ds:+d}")
    if dn == 0 and ds == 0:
        print("  -> STILL zero effect - lane-changing params genuinely don't affect routing here")
    else:
        print("  -> FIX WORKS! XML attribute approach changes sim output")
    results.append({"label": "diff_assertive_minus_timid_via_xml", "delta_north": dn, "delta_south": ds})

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = {}
    all_results["test_a"] = test_a()
    all_results["test_b"] = test_b()
    all_results["test_bplus"] = test_bplus()
    all_results["test_c"] = test_c()
    all_results["test_d"] = test_d()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"stepE_{ts}.json")
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nLog saved to {log_path}")

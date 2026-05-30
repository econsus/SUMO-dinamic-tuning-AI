import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from data_repo import DataRepo
from SumoSimulation import SUMOSimulation
from run_recorder import RunRecorder


def main():
    repo = DataRepo()

    sumo_config = [
        'sumo-gui',
        '-c', str(repo.sumo_config_path),
        '--step-length', '0.05',
        '--delay', '1000',
        '--lateral-resolution', '0.1',
        '--start',
        '--quit-on-end'
    ]

    loops = repo.get_induction_loops()
    loop_ids = [l['id'] for l in loops]

    recorder = RunRecorder(loop_ids, repo.BASE_DIR / "logs")

    try:
        with SUMOSimulation(sumo_config) as sim:
            sim.run(
                simulation_duration=600,
                step_callback=recorder.step_callback
            )

        report_path = recorder.save_report()
        print(f"\nReport saved to {report_path}")

    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
from datetime import datetime
from pathlib import Path
import traci


class RunRecorder:
    def __init__(self, loop_ids, logs_dir, minute_interval=60, step_length=0.05):
        self.loop_ids = loop_ids
        self.minute_interval = minute_interval
        self.step_length = step_length
        self.steps_per_minute = int(minute_interval / step_length)

        self.minute_counts = {lid: [] for lid in loop_ids}
        self.total_counts = {lid: 0 for lid in loop_ids}
        self.minute_accum = {lid: 0 for lid in loop_ids}
        self.step_counter = 0

        start_time = datetime.now()
        folder_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(logs_dir) / folder_name
        os.makedirs(self.run_dir, exist_ok=True)

    def step_callback(self):
        for lid in self.loop_ids:
            count = traci.inductionloop.getLastStepVehicleNumber(lid)
            self.minute_accum[lid] += count
        self.step_counter += 1
        if self.step_counter >= self.steps_per_minute:
            self._flush_minute()

    def _flush_minute(self):
        for lid in self.loop_ids:
            self.minute_counts[lid].append(self.minute_accum[lid])
            self.total_counts[lid] += self.minute_accum[lid]
            self.minute_accum[lid] = 0
        self.step_counter = 0

    def save_report(self):
        for lid in self.loop_ids:
            if self.minute_accum[lid] > 0:
                self.minute_counts[lid].append(self.minute_accum[lid])
                self.total_counts[lid] += self.minute_accum[lid]
                self.minute_accum[lid] = 0

        report_path = self.run_dir / "induction_report.txt"
        with open(report_path, 'w') as f:
            f.write("Induction Loop Vehicle Count Report\n")
            f.write("=" * 40 + "\n\n")
            for lid in self.loop_ids:
                f.write(f"Loop: {lid}\n")
                f.write("-" * 30 + "\n")
                for i, count in enumerate(self.minute_counts[lid], 1):
                    f.write(f"  Minute {i}: {count}\n")
                f.write(f"\n  Total: {self.total_counts[lid]}\n\n")

        return report_path

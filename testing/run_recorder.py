import csv
import json
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

        report = {}
        for lid in self.loop_ids:
            report[lid] = {
                "per_minute": self.minute_counts[lid],
                "total": self.total_counts[lid]
            }

        json_path = self.run_dir / "induction_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        csv_path = self.run_dir / "induction_report.csv"
        max_minutes = max(len(v["per_minute"]) for v in report.values()) if report else 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["Minute"]
            for lid in self.loop_ids:
                header.append(f"{lid}")
            writer.writerow(header)
            for i in range(max_minutes):
                row = [i + 1]
                for lid in self.loop_ids:
                    counts = report[lid]["per_minute"]
                    row.append(counts[i] if i < len(counts) else 0)
                writer.writerow(row)
            total_row = ["Total"]
            for lid in self.loop_ids:
                total_row.append(report[lid]["total"])
            writer.writerow(total_row)

        return json_path, csv_path

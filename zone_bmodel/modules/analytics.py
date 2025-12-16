import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.settings import Config


class Analytics:
    """
    Collects per-person data, computes aggregated stats, and emits reports/visualisations.
    """

    def __init__(self) -> None:
        self.people_data: Dict[int, Dict] = {}
        self.stats: Dict = {
            "total_people": 0,
            "bypassers": 0,
            "current_inside": 0,
            "max_concurrent": 0,
            "in_count": 0,  # People who crossed entry line
            "out_count": 0,  # People who crossed exit line
        }

    def update_person_data(self, person_id: int, track_data: Dict) -> None:
        is_new = person_id not in self.people_data
        was_entry_crossed = self.people_data.get(person_id, {}).get("track_data", {}).get("entry_crossed", False)
        was_exit_crossed = self.people_data.get(person_id, {}).get("track_data", {}).get("exit_crossed", False)
        
        self.people_data[person_id] = {
            "track_data": track_data.copy(),
            "classification": track_data.get("classification", "unknown"),
        }
        
        if is_new:
            self.stats["total_people"] += 1
        
        # Count IN: when entry is crossed for the first time
        if track_data.get("entry_crossed", False) and not was_entry_crossed:
            self.stats["in_count"] += 1
        
        # Count OUT: when exit is crossed for the first time
        if track_data.get("exit_crossed", False) and not was_exit_crossed:
            self.stats["out_count"] += 1
        
        self.stats["bypassers"] = len([p for p in self.people_data.values() if p["classification"] == "bypasser"])
        # Current inside: only people who entered but haven't exited yet
        self.stats["current_inside"] = len(
            [p for p in self.people_data.values() 
             if p["track_data"].get("entry_crossed", False) and not p["track_data"].get("exit_crossed", False)]
        )
        self.stats["max_concurrent"] = max(self.stats["max_concurrent"], self.stats["current_inside"])

    def get_current_stats(self) -> Dict:
        return self.stats.copy()

    def get_final_stats(self) -> Dict:
        stats = self.stats.copy()
        
        # Calculate total from IN + OUT (more accurate than total_people)
        stats["total_from_in_out"] = stats["in_count"] + stats["out_count"]
        
        dwell_times = [
            p["track_data"].get("dwell_time")
            for p in self.people_data.values()
            if p["track_data"].get("dwell_time") is not None
        ]
        stats["avg_dwell_time"] = sum(dwell_times) / len(dwell_times) if dwell_times else 0
        return stats


    def save_report(self, report: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(Config.REPORT_OUTPUT_DIR, f"report_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self._save_csv(timestamp)
        self._generate_visuals(timestamp)
        print(f"Reports saved under timestamp {timestamp}")

    def _save_csv(self, timestamp: str) -> None:
        rows: List[Dict] = []
        for person_id, data in self.people_data.items():
            track = data["track_data"]
            rows.append(
                {
                    "person_id": person_id,
                    "classification": data["classification"],
                    "dwell_time_seconds": track.get("dwell_time", 0),
                    "entry_time": track.get("entry_time"),
                    "exit_time": track.get("exit_time"),
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(Config.REPORT_OUTPUT_DIR, f"detailed_data_{timestamp}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")

    def _generate_visuals(self, timestamp: str) -> None:
        try:
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Customer Traffic Analysis", fontsize=16)

            # Movement classification pie
            labels = ["Bypassers", "Others"]
            values = [
                self.stats["bypassers"],
                max(0, self.stats["total_people"] - self.stats["bypassers"]),
            ]
            axes[0].pie(
                [v for v in values if v > 0],
                labels=[labels[i] for i, v in enumerate(values) if v > 0],
                autopct="%1.1f%%",
            )
            axes[0].set_title("Movement Classification")

            # Key metrics
            metrics = ["Total", "Bypassers"]
            stats = self.get_final_stats()
            values = [
                stats["total_people"],
                stats["bypassers"],
            ]
            bars = axes[1].bar(metrics, values, color=["steelblue", "tomato", "seagreen", "gold"])
            for bar, value in zip(bars, values):
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{value:.1f}", ha="center")
            axes[1].set_title("Key Metrics")

            plt.tight_layout()
            plot_path = os.path.join(Config.REPORT_OUTPUT_DIR, f"visualization_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(f"Warning: unable to generate visualizations ({exc})")


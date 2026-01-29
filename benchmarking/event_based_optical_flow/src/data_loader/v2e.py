import glob
import logging
import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from . import DataLoaderBase

logger = logging.getLogger(__name__)


class SimpleEventDataLoader(DataLoaderBase):
    NAME = "v2e"

    def __init__(self, config: Dict | None = None):
        if config is None:
            config = {}
        self._EVAL_HZ: float = config.get("eval_hz", 30.0)
        self.load_gt_flow: bool = config.get("load_gt_flow", False)
        self.gt_flow_dir: str | None = config.get("gt_flow_dir")
        super().__init__(config)

        self.events: Dict[str, np.ndarray] = {}
        self.ts: Optional[np.ndarray] = None

        # gt flow
        self.gt_flow_available: bool = True
        self._gt_flow_paths: list[str] = []  # list of .flo files
        self._dt = 1.0 / self._EVAL_HZ

        self.eval_times: Optional[np.ndarray] = None
        self.eval_event_start_idxs: Optional[np.ndarray] = None
        self.eval_event_end_idxs: Optional[np.ndarray] = None

        self.min_ts: float = 0.0
        self.max_ts: float = 0.0

        # self.events: Optional[np.ndarray] = None
        # self.ts: Optional[np.ndarray] = None
        # self.min_ts: float = 0.0
        # self.max_ts: float = 0.0
        # self.eval_times: Optional[np.ndarray] = None

    def set_sequence(self, sequence_name: str) -> None:
        super().set_sequence(sequence_name)

        event_file = self.dataset_files["event"]
        if not os.path.isfile(event_file):
            raise FileNotFoundError(f"{event_file} does not exist.")
        logger.info("Reading events from %s", event_file)

        with h5py.File(event_file, "r") as f:
            raw = np.asarray(f["events"], dtype=np.int64)  # [t, x, y, p]

        self.events = {
            "t": raw[:, 0].astype(np.float64) * 1e-6,
            "x": raw[:, 1].astype(np.int16),
            "y": raw[:, 2].astype(np.int16),
            "p": (raw[:, 3] > 0),
        }
        self.ts = self.events["t"]

        # eval timeline
        self.min_ts = float(self.ts.min())
        self.max_ts = float(self.ts.max())
        dt = 1.0 / self._EVAL_HZ
        self.eval_times = np.arange(
            self.min_ts, self.max_ts, dt, dtype=np.float64
        )

        logger.info("Pre-computing eval event indices")
        self.eval_event_start_idxs = np.searchsorted(
            self.ts, self.eval_times, side="left"
        )
        self.eval_event_end_idxs = np.searchsorted(
            self.ts, self.eval_times, side="left"
        )
        logger.info(
            "Loaded %d events (%.2f s - %.2f s ).",
            len(self.ts),
            self.min_ts,
            self.max_ts,
        )

        # gt flow
        if self.load_gt_flow and self.gt_flow_dir:

            # seq_dir = os.path.join(self.gt_flow_dir, sequence_name)
            flo_dir = self.gt_flow_dir
            self._gt_flow_paths = sorted(
                glob.glob(os.path.join(flo_dir, "*.flo"))
            )
            if len(self._gt_flow_paths) >= len(self.eval_times) - 1:
                self.gt_flow_available = True
                logger.info(
                    "Found %d gt flow files in %s",
                    len(self._gt_flow_paths),
                    flo_dir,
                )
            else:
                logger.warning("gt flow files missing or not enough.")
                logger.info(
                    "GT-flow files found: %d, expected ≥ %d",
                    len(self._gt_flow_paths),
                    len(self.eval_times) - 1,
                )
                self.gt_flow_available = False

        # Split & convert
        # t_us = raw[:, 0].astype(np.float64)
        # x = raw[:, 1].astype(np.int16)
        # y = raw[:, 2].astype(np.int16)
        # p_raw = raw[:, 3].astype(np.int8)
        #
        # t = t_us * 1e-6  # seconds
        # p = np.where(p_raw > 0, 1, -1)  # ensure –1 / +1
        #
        # # keep the same ordering as the MVSEC loader: [y, x, t, p]
        # self.events = np.stack((y, x, t, p), axis=1).astype(np.float64)
        # self.ts = self.events[:, 2]
        #
        # # Statistics & evaluation timeline
        # self.min_ts = float(self.ts.min())
        # self.max_ts = float(self.ts.max())
        # step = 1.0 / self._EVAL_HZ  # 0.0333… for 30 Hz
        # self.eval_times = np.arange(self.min_ts, self.max_ts, step, dtype=np.float64)
        #
        # logger.info("Loaded %d events (%.1f s – %.1f s).", len(self.events), self.min_ts, self.max_ts)

    def get_sequence(self, sequence_name: str) -> Dict[str, str]:
        event_path = os.path.join(self.dataset_dir, f"{sequence_name}.h5")
        return {"event": event_path}

    def load_event(
        self, start_index: int, end_index: int, cam: str = "left", *_
    ) -> np.ndarray:
        if self.events is None:
            raise RuntimeError("Sequence not loaded – call set_sequence()")
        if start_index < 0 or end_index > len(self.ts):
            raise IndexError(
                f"Requested [{start_index}:{end_index}] for {len(self.ts)} events."
            )
        y = self.events["y"][start_index:end_index]
        x = self.events["x"][start_index:end_index]
        t = self.events["t"][start_index:end_index]
        p = np.where(self.events["p"][start_index:end_index], 1, -1)
        return np.column_stack((y, x, t, p)).astype(np.float64)
        # return self.events[start_index:end_index].copy()

    def index_to_time(self, index: int) -> float:
        return float(self.ts[index])

    def time_to_index(self, time: float) -> int:
        # Right side search gives the first index after time,  subtract 1
        return int(np.searchsorted(self.ts, time, side="right") - 1)

    def eval_frame_time_list(self) -> np.ndarray:
        return self.eval_times

    def get_eval_event_indices(self, idx: int) -> tuple[int, int]:
        return int(self.eval_event_start_idx[idx]), int(
            self.eval_event_end_idxs[idx]
        )

    def __len__(self) -> int:
        return 0 if self.ts is None else len(self.ts)

    def load_optical_flow(self, t1: float, t2: float) -> np.ndarray:
        if not self.gt_flow_available:
            raise RuntimeError("gt flow not available")

        idx = int(round((t1 - self.min_ts) / self._dt))
        if idx < 0 or idx >= len(self._gt_flow_paths):
            raise IndexError(f"No gt flow file for t=1{t1:.6f}s (index {idx}).")

        flow = self._read_flo(self._gt_flow_paths[idx])  # H x W x 2 (u,v)
        # convert (u,v) -> (dy, dx)
        flow_h = flow[..., 1]
        flow_w = flow[..., 0]
        return np.stack((flow_h, flow_w), axis=2).astype(np.float32)

    @staticmethod
    def _read_flo(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            tag = np.fromfile(f, np.float32, count=1)[0]
            if tag != 202021.25:
                raise IOError(f"{path} is not a valid .flo file (wrong tag)")
            w = int(np.fromfile(f, np.int32, count=1)[0])
            h = int(np.fromfile(f, np.int32, count=1)[0])
            data = np.fromfile(f, np.float32, count=2 * w * h)
        return np.reshape(data, (h, w, 2))

    def load_calib(self) -> Dict:
        """identity mapping"""
        return None

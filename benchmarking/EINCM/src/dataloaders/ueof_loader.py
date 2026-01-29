from pathlib import Path

import cv2 as cv
import numpy as np
from dataloaders.reader_utils.hdf5_file_reader import HDF5FileReader


class UEOFDataset:
    def __init__(self, root_dir, sequence_name):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.images_dir = self.root_dir / f"{sequence_name}/images"
        self.image_timestamps_path = (
            self.root_dir / f"{sequence_name}/image_timestamps.txt"
        )
        self.events_h5_path = self.root_dir / f"{sequence_name}/events.h5"
        self.flow_gt_dir = self.root_dir / f"{sequence_name}/gt_flow"
        self.flow_gt_timestamps_path = (
            self.root_dir / f"{sequence_name}/gt_flow_timestamps.txt"
        )
        self.eval_timestamps_path = (
            self.root_dir / f"{sequence_name}/evaluation_timestamps.txt"
        )


class UEOFDataLoader:
    def __init__(
        self,
        root_dir,
        sequence_name,
        delta_idx,  # TODO
        event_sensor_size=(720, 1280),
        des_n_events=90000,
        load_more_images=False,  # TODO
        prefer_latest_events=True,
    ):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.delta_idx = delta_idx
        self.des_n_events = des_n_events
        self.load_more_images = load_more_images
        self.prefer_latest_events = prefer_latest_events

        self.n_event_deficiency = None

        # set event sensor size
        self.height = event_sensor_size[0]
        self.width = event_sensor_size[1]
        self.sensor_size = event_sensor_size

        self.dataset = UEOFDataset(root_dir, sequence_name)
        self._DATA_LOADED = False

    def get_ready(self):
        print(f"Loading ({self.sequence_name}) data...")
        self.load_data()
        self.precompute_eval_event_indices()
        self.precompute_eval_image_indices()
        self.precompute_eval_flow_gt_indices()
        print(f'\nReady to load {self.sequence_name} datasamples.\n{"":-^80}')

    def load_data(self):
        # load events from h5 file
        with HDF5FileReader(self.dataset.events_h5_path) as h5_rdr:
            events_np = h5_rdr.read_dataset("events")  # uint32

        self.events = {
            "t": events_np[:, 0].astype("float64")
            / 1e6,  # microseconds to seconds
            "x": events_np[:, 1].astype("int16"),
            "y": events_np[:, 2].astype("int16"),
            "p": events_np[:, 3].astype("bool"),
        }  # 1/0 to True/False

        # load image and flow_gt paths from image directory
        self.image_paths = sorted(
            [
                str(p)
                for p in self.dataset.images_dir.iterdir()
                if str(p).endswith(".png")
            ]
        )
        self.flow_gt_paths = sorted(
            [
                str(p)
                for p in self.dataset.flow_gt_dir.iterdir()
                if str(p).endswith(".flo")
            ]
        )

        # load image, eval and gt timestamps from txt files
        self.image_ts = (
            np.loadtxt(
                self.dataset.image_timestamps_path, skiprows=0, dtype=np.float64
            )
            / 1e3
        )
        self.flow_gt_ts = (
            np.loadtxt(
                self.dataset.flow_gt_timestamps_path,
                skiprows=0,
                dtype=np.float64,
            )
            / 1e3
        )
        self.eval_ts = (
            np.loadtxt(
                self.dataset.eval_timestamps_path, skiprows=0, dtype=np.float64
            )
            / 1e3
        )

        self._DATA_LOADED = True

    def precompute_eval_event_indices(self):
        print("Pre-computing eval event indices for efficiency... ")
        self.eval_event_start_idxs = np.searchsorted(
            self.events["t"], self.eval_ts[:, 0], side="left"
        )
        self.eval_event_end_idxs = np.searchsorted(
            self.events["t"], self.eval_ts[:, 1], side="left"
        )
        print("\bDone.")

    def precompute_eval_image_indices(self):
        print("Pre-computing eval image indices for efficiency... ")
        self.eval_image_start_idxs = np.searchsorted(
            self.image_ts, self.eval_ts[:, 0], side="left"
        )
        self.eval_image_end_idxs = np.searchsorted(
            self.image_ts, self.eval_ts[:, 1], side="left"
        )
        print("\bDone.")

    def precompute_eval_flow_gt_indices(self):
        print("Pre-computing eval flow gt indices for efficiency... ")
        self.eval_flow_gt_start_idxs = np.searchsorted(
            self.image_ts, self.eval_ts[:, 0], side="left"
        )
        self.eval_flow_gt_end_idxs = np.searchsorted(
            self.image_ts, self.eval_ts[:, 1], side="left"
        )
        print("\bDone.")

    def get_sample(self, eval_idx):
        # prepare image samples
        idx_img_start, idx_img_end = (
            self.eval_image_start_idxs[eval_idx],
            self.eval_image_end_idxs[eval_idx],
        )
        sampled_images_paths = self.image_paths[idx_img_start : idx_img_end + 1]
        sampled_images = [
            cv.imread(im_path, cv.IMREAD_GRAYSCALE)
            for im_path in sampled_images_paths
        ]

        # prepare event samples
        idx_evt_start, idx_evt_end = (
            self.eval_event_start_idxs[eval_idx],
            self.eval_event_end_idxs[eval_idx],
        )
        orig_n_events = idx_evt_end - idx_evt_start
        if self.des_n_events is not None:
            # make sure we have desired num of events (corner cases not handled)
            self.n_event_deficiency = self.des_n_events - orig_n_events
            if self.n_event_deficiency > 0:
                idx_evt_start -= np.ceil(self.n_event_deficiency / 2).astype(
                    int
                )
                idx_evt_end += np.floor(self.n_event_deficiency / 2).astype(int)
                idx_evt_start = max(0, idx_evt_start)
                idx_evt_end = min(idx_evt_end, len(self.events["x"]))
            elif self.n_event_deficiency < 0:
                # TODO
                #  if there are more events then keep the boundary events
                # and remove randomly from the middle section
                if self.prefer_latest_events:
                    idx_evt_start = idx_evt_end - self.des_n_events
                else:
                    idx_evt_end = idx_evt_start + self.des_n_events

        sampled_events = {
            "x": self.events["x"][idx_evt_start:idx_evt_end],
            "y": self.events["y"][idx_evt_start:idx_evt_end],
            "t": self.events["t"][idx_evt_start:idx_evt_end],
            "p": self.events["p"][idx_evt_start:idx_evt_end],
        }

        # prepare flow gt samples
        flow_gt = self.read_flo_file(self.flow_gt_paths[eval_idx])

        return {
            "images": sampled_images,
            "events": sampled_events,
            "flow_gt": flow_gt,
            "image_ts": self.image_ts[idx_img_start : idx_img_end + 1],
            "event_ts": self.events["t"][idx_evt_start:idx_evt_end],
            "flow_gt_ts": self.flow_gt_ts[eval_idx],
            "eval_ts": self.eval_ts[eval_idx],
            "n_event_deficiency": self.n_event_deficiency,
            "orig_n_events": orig_n_events,
        }

    def read_flo_file(self, flo_file_path):
        """Adapted from https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py#L18"""
        with open(flo_file_path, "rb") as flo_file:
            flo_number = np.fromfile(flo_file, dtype=np.float32, count=1)[0]
            assert (
                flo_number == 202021.25
            ), f"Flow number {flo_number} incorrect. Invalid .flo file"

            width = np.fromfile(flo_file, dtype=np.int32, count=1)[0]
            height = np.fromfile(flo_file, dtype=np.int32, count=1)[0]
            data = np.fromfile(flo_file, np.float32, count=2 * width * height)
            flow = np.resize(data, (height, width, 2))

        return flow

    def __len__(self):
        if not self._DATA_LOADED:
            raise RuntimeError("Data not loaded yet. Call get_ready() first.")
        return len(self.eval_ts)

    def __getitem__(self, idx):
        if not self._DATA_LOADED:
            raise RuntimeError("Data not loaded yet. Call get_ready() first.")
        return self.get_sample(idx)

import math
from pathlib import Path
from typing import Dict, Tuple
import weakref

import cv2
import h5py
from numba import jit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import visualization as visu
from matplotlib import pyplot as plt
from utils import transformers
import os
import imageio

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float

VISU_INDEX = 1

class EventSlicer:
    """Supports DSEC + V2E formats

    DSEC layout 
        events/p, events/x, events/y, events/t

    V2E (packed) layout:
        /events  - [t, x, y, p]
    """

    def __init__(self, h5f: h5py.File, time_block_us: int = 100_000):
        self.h5f = h5f
        self.time_block_us = time_block_us

        # Detect layout 
        std_ev_keys = {k for k in h5f.keys() if k.startswith("events/")}
        self._packed = False
        if "events" in h5f and isinstance(h5f["events"], h5py.Dataset):
            if not {"events/p", "events/x", "events/y", "events/t"}.issubset(std_ev_keys):
                self._packed = True

        if self._packed:
            # V2E layout
            self._ds = h5f["events"]     
            self._t = np.asarray(self._ds[:, 0], dtype="int64")  
            self.first_event_ts = int(self._t[0])
            self.t_final = int(self._t[-1])
            self.t_offset = 0      

            # build ms→idx 
            n_ms = self.t_final // 1000 + 1
            self.ms_to_idx = np.searchsorted(
                self._t, np.arange(n_ms, dtype="int64") * 1000, side="left"
            )

            print(
                f"[DEBUG][EventSlicer] Packed dataset detected: first={self.first_event_ts} µs, "
                f"last={self.t_final} µs, events={self._t.size}, ms_to_idx len={self.ms_to_idx.size}"
            )
        else:
            # DSEC layout
            self.events = {d: h5f[f"events/{d}"] for d in ["p", "x", "y", "t"]}
            self.ms_to_idx = np.asarray(h5f["ms_to_idx"], dtype="int64")
            self.t_offset = int(h5f["t_offset"][()])
            self.first_event_ts = int(self.events["t"][0]) + self.t_offset
            self.t_final = int(self.events["t"][-1]) + self.t_offset

            print(
                f"[DEBUG][EventSlicer] Standard DSEC dataset: first={self.first_event_ts} µs, "
                f"last={self.t_final} µs, ms_to_idx len={self.ms_to_idx.size}"
            )

    def get_final_time_us(self):
        return self.t_final

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us: int) -> Tuple[int, int]:
        assert ts_end_us > ts_start_us
        return math.floor(ts_start_us / 1000), math.ceil(ts_end_us / 1000)

    @staticmethod
    @jit(nopython=True)
    def _find_inner_indices(time_array: np.ndarray, time_start_us: int, time_end_us: int) -> Tuple[int, int]:
        """Return relative slice [inner_start:inner_end] such that
        time_start_us ≤ t < time_end_us."""
        if time_array.size == 0:
            return 0, 0
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size

        idx_start = -1
        for i in range(time_array.size):
            if time_array[i] >= time_start_us:
                idx_start = i
                break
        if idx_start == -1:
            return 0, 0  # no events in window

        idx_end = time_array.size
        for i in range(time_array.size - 1, -1, -1):
            if time_array[i] >= time_end_us:
                idx_end = i
            else:
                break
        return idx_start, idx_end

    def ms2idx(self, time_ms: int):
        if time_ms >= self.ms_to_idx.size:
            return None
        return int(self.ms_to_idx[time_ms])

    def get_events(self, t_start_us: int, t_end_us: int):
        """Return dict of numpy arrays p,x,y,t in [t_start_us, t_end_us)."""
        if t_start_us >= t_end_us:
            raise ValueError("t_start_us must be < t_end_us")

        # clamp negative start to 0 
        if t_start_us < 0:
            print(f"[DEBUG][EventSlicer] Clamping negative t_start {t_start_us} to 0")
            t_start_us = 0
        if t_end_us <= 0:
            return None

        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        idx_start = self.ms2idx(t_start_ms)
        idx_end = self.ms2idx(t_end_ms)
        if idx_start is None or idx_end is None or idx_start == idx_end:
            print(f"[DEBUG][EventSlicer] Window {t_start_us}-{t_end_us} outside range")
            return None

        if self._packed:
            block = self._ds[idx_start:idx_end]
            t_block = block[:, 0]
            inner_start, inner_end = self._find_inner_indices(t_block, t_start_us, t_end_us)
            if inner_start == inner_end:
                print(
                    f"[DEBUG][EventSlicer] No events in window {t_start_us}-{t_end_us} (packed)"
                )
                return None
            sub = block[inner_start:inner_end]
            return {
                "p": sub[:, 3].astype("uint8"),
                "x": sub[:, 1].astype("uint16"),
                "y": sub[:, 2].astype("uint16"),
                "t": sub[:, 0] + self.t_offset,
            }
        else:
            # DSEC Layout
            time_arr_consv = np.asarray(self.events["t"][idx_start:idx_end])
            inner_start, inner_end = self._find_inner_indices(time_arr_consv, t_start_us, t_end_us)
            if inner_start == inner_end:
                print(
                    f"[DEBUG][EventSlicer] No events in window {t_start_us}-{t_end_us} (std)"
                )
                return None
            global_start = idx_start + inner_start
            global_end = idx_start + inner_end
            ev_out = {
                "t": time_arr_consv[inner_start:inner_end] + self.t_offset,
            }
            for k in ["p", "x", "y"]:
                ev_out[k] = np.asarray(self.events[k][global_start:global_end])
            return ev_out

class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = "test", delta_t_ms: int = 100,
                 num_bins: int = 15, transforms=None, name_idx=0, visualize=False):
        assert num_bins >= 1
        assert delta_t_ms == 100
        assert seq_path.is_dir()
        assert mode in {"train", "test"}
        """
        Directory Structure:

        Dataset
        └── test
            ├── scene_x
            │   ├── events_left/events.h5
            │   ├── image_timestamps.txt
            │   └── test_forward_flow_timestamps.csv
        """

        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        # Test timestamp file just used for visualization indices
        file = np.genfromtxt(seq_path / "test_forward_flow_timestamps.csv", delimiter=",")
        self.idx_to_visualize = file[:, 2] if file.ndim == 2 else []
        # Save output dimensions
        self.height = 540
        self.width = 960
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        assert self.num_bins==15

        # Set event representation
        self.voxel_grid = None
        if representation_type == RepresentationType.VOXEL:
            self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)


        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        #Load and compute timestamps and indices
        timestamps_images = np.loadtxt(seq_path / 'image_timestamps.txt', dtype='int64')
        image_indices = np.arange(len(timestamps_images))
        # But only use every second one because we train at 10 Hz, and we leave away the 1st & last one
        # Changed to ::3 for 30fps WIRN
        self.timestamps_flow = timestamps_images[2:-2]
        self.indices = image_indices[2:-2]

        # Left events only
        ev_dir_location = seq_path / 'events_left'
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        self.rectify_ev_map = None
        if ev_rect_file.is_file():
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                rmap = h5_rect['rectify_map'][()]
                # Use the map only if its resolution matches what the network expects
                if rmap.shape == (self.height, self.width, 2):
                    self.rectify_ev_map = rmap
                else:
                    print(f"[DEBUG][Sequence] Ignoring rectify_map – "
                          f"shape is {rmap.shape}, expected {(self.height, self.width, 2)}")
        else:
            print(f"[DEBUG][Sequence] No rectify_map.h5 found – skipping")

        #
        # with h5py.File(str(ev_rect_file), 'r') as h5_rect:
        #     self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, p, t, x, y, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        """Return rectified (x,y).  If no map is available, pass points through."""
        if self.rectify_ev_map is None:
            return np.stack([x, y], axis=1)       # identity

        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]
    #
    # def rectify_events(self, x: np.ndarray, y: np.ndarray):
    #     # assert location in self.locations
    #     # From distorted to undistorted
    #     rectify_map = self.rectify_ev_map
    #     assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
    #     assert x.max() < self.width
    #     assert y.max() < self.height
    #     return rectify_map[y, x]
    #
    def get_data_sample(self, index, crop_window=None, flip=None):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_start = [self.timestamps_flow[index] - self.delta_t_us, self.timestamps_flow[index]]
        ts_end = [self.timestamps_flow[index], self.timestamps_flow[index] + self.delta_t_us]

        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index]
        }
        # Save sample for benchmark submission
        output['save_submission'] = file_index in self.idx_to_visualize
        output['visualize'] = self.visualize_samples

        for i in range(len(names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i])

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x']-2) & (x_rect < crop_window['start_x']+crop_window['crop_width']+2)
                y_mask = (y_rect >= crop_window['start_y']-2) & (y_rect < crop_window['start_y']+crop_window['crop_height']+2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                output[names[i]] = event_representation
            output['name_map']=self.name_idx
        return output

    def __getitem__(self, idx):
        sample =  self.get_data_sample(idx)
        return sample


class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_ms: int=100,
                 num_bins: int=15, transforms=None, sequence_length=1, name_idx=0, visualize=False):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_ms, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize)
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow)-self.sequence_length+1):
                diff = self.timestamps_flow[i+self.sequence_length-1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow)-1):
                diff = self.timestamps_flow[i+1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        # Valid index is the actual index we want to load, which guarantees a continuous sequence length
        valid_idx = self.valid_indices[idx]

        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        # Add first sample
        sample = self.get_data_sample(j)
        sequence.append(sample)

        # Data augmentation according to first sample
        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length-1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert(ts_cur-ts_old < 100000 + 1000)
            sample = self.get_data_sample(j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        # Check if the current sample is the first sample of a continuous sequence
        if idx==0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
            print("Timestamp {} is the first one of the next seq!".format(self.timestamps_flow[self.valid_indices[idx]]))
        else:
            sequence[0]['new_sequence'] = 0
        return sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int=100, num_bins=15,
                 type='standard', config=None, visualize=False):
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert test_path.is_dir(), str(test_path)
        assert delta_t_ms == 100
        self.config=config
        self.name_mapper_test = []

        test_sequences = list()
        for child in test_path.iterdir():
            self.name_mapper_test.append(str(child).split("/")[-1])
            if type == 'standard':
                test_sequences.append(Sequence(child, representation_type, 'test', delta_t_ms, num_bins,
                                               transforms=[],
                                               name_idx=len(self.name_mapper_test)-1,
                                               visualize=visualize))
            elif type == 'warm_start':
                test_sequences.append(SequenceRecurrent(child, representation_type, 'test', delta_t_ms, num_bins,
                                                        transforms=[], sequence_length=1,
                                                        name_idx=len(self.name_mapper_test)-1,
                                                        visualize=visualize))
            else:
                raise Exception('Please provide a valid subtype [standard/warm_start] in config file!')

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_test_dataset(self):
        return self.test_dataset


    def get_name_mapping_test(self):
        return self.name_mapper_test

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(self.test_dataset.datasets[0].num_bins), True)


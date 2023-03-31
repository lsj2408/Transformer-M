from functools import lru_cache

from ogb.lsc import PCQM4Mv2Evaluator
import numpy as np
import torch
from torch.nn import functional as F
from fairseq.data import FairseqDataset, BaseWrapperDataset, data_utils

from .wrapper import MyPygPCQM4MDataset, MyPygPCQM4MPosDataset
from .collator import collator, collator_3d, collator_3d_qm9
from .ogb_datasets import OGBDatasetLookupTable
from .pyg_datasets import PYGDatasetLookupTable

class PCQPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset"):
        super().__init__()

        assert dataset_name in [
            "PCQM4M-LSC-V2",
            "PCQM4M-LSC-V2-TOY",
            "PCQM4M-LSC-V2-3D"
        ], "Only support PCQM4M-LSC-V2 or PCQM4M-LSC-V2-POS"
        self.dataset_name = dataset_name
        if dataset_name == 'PCQM4M-LSC-V2-3D':
            self.dataset = MyPygPCQM4MPosDataset(root=dataset_path)
        else:
            self.dataset = MyPygPCQM4MDataset(root=dataset_path)
        self.setup()

    def setup(self, stage: str = None):
        split_idx = self.dataset.get_idx_split()
        if self.dataset_name in ["PCQM4M-LSC-V2", "PCQM4M-LSC-V2-3D"]:
            self.train_idx = split_idx["train"]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]
        elif self.dataset_name == "PCQM4M-LSC-V2-TOY":
            self.train_idx = split_idx["train"][:5000]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)

        self.max_node = 256
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class PYGPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset", seed=42, task_idx=4):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_name, seed=seed, dataset_path=dataset_path, task_idx=task_idx)
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data

        self.max_node = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class OGBPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset", seed=42):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_name, seed=seed)
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data

class BatchedDataDataset(FairseqDataset):
    def __init__(self, dataset, dataset_version="2D", max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        self.dataset_version = dataset_version
        assert self.dataset_version in ["2D", "3D", "3D_QM9"]

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.dataset_version == "3D":
            collator_fn = collator_3d
        elif self.dataset_version == "3D_QM9":
            collator_fn = collator_3d_qm9
        else:
            collator_fn = collator
        return collator_fn(samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max)

class CacheAllDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.dataset[index]

    def collater(self, samples):
        return self.dataset.collater(samples)

class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, size, seed):
        super().__init__(dataset)
        self.size = size
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.size)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)
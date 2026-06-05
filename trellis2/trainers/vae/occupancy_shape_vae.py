from typing import Tuple, Dict
import os
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler


class OccupancyShapeVaeTrainer(BasicTrainer):
    """
    Trainer for a sparse occupancy shape VAE.

    This trainer is intentionally structure-focused: it trains the VAE decoder
    to predict sparse subdivision decisions and uses the latent features later
    as coordinate-aligned shape conditioning.
    """

    def __init__(
        self,
        *args,
        num_workers: int = None,
        lambda_recon: float = 1e-4,
        lambda_subdiv: float = 1.0,
        lambda_kl: float = 1e-6,
        **kwargs,
    ):
        self.num_workers = num_workers
        super().__init__(*args, **kwargs)
        self.lambda_recon = lambda_recon
        self.lambda_subdiv = lambda_subdiv
        self.lambda_kl = lambda_kl

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers if self.num_workers is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x: sp.SparseTensor,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        z, mean, logvar = self.training_models['encoder'](x, sample_posterior=True, return_raw=True)
        y, subs_gt, subs = self.training_models['decoder'](z)

        terms = edict(loss=0.0)

        if self.lambda_recon > 0:
            terms['recon'] = F.binary_cross_entropy_with_logits(y.feats.flatten(), x.feats.flatten())
            terms['loss'] = terms['loss'] + self.lambda_recon * terms['recon']

        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f'bce_sub{i}'] = F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            terms['loss'] = terms['loss'] + self.lambda_subdiv * terms[f'bce_sub{i}']

        terms['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms['loss'] = terms['loss'] + self.lambda_kl * terms['kl']

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        snapshot_dataset = copy.deepcopy(self.dataset)
        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        gt_images = []
        rec_images = []
        pred_subdiv_images = []

        self.models['encoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)

            z = self.models['encoder'](args['x'])

            self.models['decoder'].train()
            y_guided = self.models['decoder'](z)[0]

            z.clear_spatial_cache()
            self.models['decoder'].eval()
            y_pred = self.models['decoder'](z)

            gt_images.append(self.dataset.visualize_sample({'x': args['x']})['occupancy'][:batch])
            rec_images.append(self.dataset.visualize_sample({'x': y_guided})['occupancy'][:batch])
            pred_subdiv_images.append(self.dataset.visualize_sample({'x': y_pred})['occupancy'][:batch])

        self.models['encoder'].train()
        self.models['decoder'].train()

        return {
            'gt_occupancy': {'value': torch.cat(gt_images, dim=0)[:num_samples] * 2 - 1, 'type': 'image'},
            'rec_occupancy': {'value': torch.cat(rec_images, dim=0)[:num_samples] * 2 - 1, 'type': 'image'},
            'pred_subdiv_occupancy': {'value': torch.cat(pred_subdiv_images, dim=0)[:num_samples] * 2 - 1, 'type': 'image'},
        }

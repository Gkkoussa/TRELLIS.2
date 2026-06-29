from typing import *
import json
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                for root_key, r in root.items():
                    if root_key.startswith('_'):
                        continue
                    metadata = metadata.combine_first(pd.read_csv(os.path.join(r, 'metadata.csv')).set_index('sha256'))
                self._stats[key]['Total'] = len(metadata)
                metadata = self._apply_metadata_filter_csv(
                    metadata,
                    root.get('_metadata_filter_csv', None),
                    self._stats[key],
                )
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])

    @staticmethod
    def _read_filter_sha256s(path: str) -> Set[str]:
        if os.path.isdir(path):
            path = os.path.join(path, 'metadata.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Metadata filter CSV not found: {path}')
        metadata = pd.read_csv(path, usecols=['sha256'])
        return set(metadata['sha256'].astype(str).values)

    def _apply_metadata_filter_csv(
        self,
        metadata: pd.DataFrame,
        filter_csv: Optional[str],
        stats: Dict[str, int],
    ) -> pd.DataFrame:
        if filter_csv is None or str(filter_csv).strip() == '':
            return metadata
        keep = None
        paths = [p.strip() for p in str(filter_csv).split(',') if p.strip()]
        for path in paths:
            path_keep = self._read_filter_sha256s(path)
            keep = path_keep if keep is None else keep.intersection(path_keep)
            stats[f'Metadata filter {os.path.basename(path)}'] = len(path_keep)
        keep = keep or set()
        before = len(metadata)
        filtered = metadata[metadata.index.astype(str).isin(keep)]
        stats['Metadata filters applied'] = len(paths)
        stats['Metadata filter kept'] = len(filtered)
        stats['Metadata filter removed'] = before - len(filtered)
        return filtered
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(f'Error loading {instance}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack

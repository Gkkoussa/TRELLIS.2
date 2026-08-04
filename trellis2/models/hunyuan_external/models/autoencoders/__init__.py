"""Namespace adapter for Hunyuan autoencoder components."""

import os

from ... import source_package_dir


__path__.append(os.path.join(source_package_dir(), 'models', 'autoencoders'))


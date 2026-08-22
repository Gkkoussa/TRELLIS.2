"""Expose Hunyuan model source without importing its full pipelines package."""

import os

from .. import source_package_dir


__path__.append(os.path.join(source_package_dir(), 'models'))


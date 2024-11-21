import logging
import sys

import demes
import jax
import platformdirs
import sparse
from jax.tree_util import register_pytree_node

from .momi import Momi3  # noqa: F401

__all__ = ["Momi3"]

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", platformdirs.user_cache_dir("momi3"))
logging.getLogger("jax").setLevel(logging.INFO)

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


register_pytree_node(
    demes.Graph,
    lambda g: ((), g.asdict()),
    lambda aux_data, _: demes.Graph.fromdict(aux_data),
)

register_pytree_node(
    sparse.COO,
    lambda sp: ((sp.coords, sp.data), sp.shape),
    lambda aux_data, children: sparse.COO(
        coords=children[0], data=children[1], shape=aux_data
    ),
)

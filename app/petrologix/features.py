
from petrologix import prepare_features  # noqa: F401
"""Feature engineering is NOT implemented here -- deliberately.

It lives in the `petrologix` wheel (`petrologix.features`), which is the same
module the training pipeline uses. Re-implementing or copying it here would let
serving drift from training, which produces confidently wrong predictions rather
than errors -- the worst failure mode this system has.

This module re-exports the entry point for callers that want it directly.
"""

__all__ = ["prepare_features"]

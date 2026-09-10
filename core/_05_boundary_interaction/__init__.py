"""Boundary-interaction framework for bed, surface, dry-cell, and domain-exit logic."""

from core._05_boundary_interaction.bed_boundary import *  # noqa: F401,F403
from core._05_boundary_interaction.domain_exit_boundary import *  # noqa: F401,F403
from core._05_boundary_interaction.dry_cell_boundary import *  # noqa: F401,F403
from core._05_boundary_interaction.surface_boundary import *  # noqa: F401,F403


def validate_boundary_policies(
    *,
    surface_policy: str,
    bed_policy: str,
    dry_policy: str,
    outside_policy: str,
) -> None:
    """Validate the configured boundary-interaction policies."""
    validate_outside_policy(outside_policy)
    validate_dry_policy(dry_policy)
    if surface_policy not in SURFACE_POLICIES:
        raise ValueError(
            'surfacePolicy must be '
            '"always_stick", "always_reflect", '
            '"deterministic_detachment", or "probabilistic_detachment".'
        )
    if bed_policy not in BED_POLICIES:
        raise ValueError(
            'bedPolicy must be '
            '"always_deposit", "always_reflect", '
            '"reflect_if_ustar_gt_crit", or "probabilistic_entrainment".'
        )

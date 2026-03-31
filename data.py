"""Compatibility exports for legacy top-level imports.

Primary implementations live in ``static.data``.
"""

from static.data import (
    generate_data_li_ov,
    generate_data_second_order,
    generate_parametric,
)

__all__ = [
    "generate_data_li_ov",
    "generate_data_second_order",
    "generate_parametric",
]

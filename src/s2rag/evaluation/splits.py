from __future__ import annotations

import hashlib


def fixed_partition(examples, validation_fraction: float = 0.15, test_fraction: float = 0.15):
    """Stable ID-based split; model seeds never change held-out examples."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    groups = {"train": [], "validation": [], "test": []}
    validation_end = 1 - validation_fraction - test_fraction
    test_start = 1 - test_fraction
    for example in examples:
        digest = hashlib.sha256(str(example.example_id).encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        name = "train" if value < validation_end else "validation" if value < test_start else "test"
        groups[name].append(example)
    if any(not items for items in groups.values()):
        raise ValueError("dataset is too small to produce non-empty fixed partitions")
    return groups

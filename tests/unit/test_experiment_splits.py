from dataclasses import dataclass

from qmshe.evaluation.splits import fixed_partition


@dataclass
class Example:
    example_id: str


def test_fixed_partition_is_independent_of_input_order():
    examples = [Example(str(index)) for index in range(100)]
    forward = fixed_partition(examples)
    reverse = fixed_partition(list(reversed(examples)))
    for name in ("train", "validation", "test"):
        assert {item.example_id for item in forward[name]} == {
            item.example_id for item in reverse[name]
        }
    assert set().union(*(
        {item.example_id for item in forward[name]}
        for name in ("train", "validation", "test")
    )) == {item.example_id for item in examples}

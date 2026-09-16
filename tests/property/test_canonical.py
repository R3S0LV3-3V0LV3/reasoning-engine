import pytest
from hypothesis import given
from hypothesis import strategies as st

from fre.domain.common import canonical_hash


@pytest.mark.property
@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=20))
def test_hash_is_independent_of_mapping_insertion_order(value: dict[str, int]) -> None:
    assert canonical_hash(value) == canonical_hash(dict(reversed(tuple(value.items()))))

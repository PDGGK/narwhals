from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError


@pytest.mark.parametrize("size", [1, 3])
@pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
def test_empty_frame_accepts_series(size: int, dtype: str) -> None:
    pd = pytest.importorskip("pandas")
    source = pd.DataFrame()
    source.columns.name = "features"
    native = pd.Series(range(size), dtype=dtype, index=range(5, 5 + size), name="a")
    result = (
        nw.from_native(source)
        .with_columns(nw.from_native(native, series_only=True))
        .to_native()
    )
    pd.testing.assert_frame_equal(
        result, native.to_frame().rename_axis("features", axis=1)
    )
    assert source.shape == (0, 0)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("scalar_first", [True, False])
def test_empty_frame_aligns_series_and_scalar(size: int, *, scalar_first: bool) -> None:
    pd = pytest.importorskip("pandas")
    index = list(range(8, 8 + size))
    a = pd.Series(range(size), index=index, name="a")
    b = pd.Series(range(3, 3 + size), index=index[::-1], name="b")
    columns = [nw.from_native(a, series_only=True), nw.from_native(b, series_only=True)]
    scalar = nw.lit(7).alias("c")
    expressions = [scalar, *columns] if scalar_first else [*columns, scalar]
    result = nw.from_native(pd.DataFrame()).with_columns(expressions).to_native()
    expected = pd.DataFrame(
        {"a": range(size), "b": range(3, 3 + size), "c": [7] * size}, index=index
    )
    if scalar_first:
        expected = expected[["c", "a", "b"]]
    pd.testing.assert_frame_equal(result, expected)
    assert b.index.tolist() == index[::-1]


@pytest.mark.parametrize("data", [{"old": []}, {"old": [1, 2]}])
def test_existing_columns_still_reject_mismatch(data: dict[str, list[int]]) -> None:
    pd = pytest.importorskip("pandas")
    with pytest.raises(ShapeError):
        nw.from_native(pd.DataFrame(data)).with_columns(
            nw.new_series("a", [1, 2, 3], backend="pandas")
        )


def test_empty_frame_existing_empty_behavior() -> None:
    pd = pytest.importorskip("pandas")
    frame = nw.from_native(pd.DataFrame())
    assert frame.with_columns().shape == (0, 0)
    assert frame.with_columns(nw.lit(1).alias("a")).shape == (0, 1)
    assert frame.with_columns(
        nw.new_series("a", [], dtype=nw.Int64, backend="pandas")
    ).shape == (0, 1)


@pytest.mark.parametrize("other", [[], [4], [4, 5, 6]])
def test_empty_frame_rejects_unequal_series(other: list[int]) -> None:
    pd = pytest.importorskip("pandas")
    with pytest.raises(ShapeError):
        nw.from_native(pd.DataFrame()).with_columns(
            nw.new_series("a", [1, 2], backend="pandas"),
            nw.new_series("b", other, dtype=nw.Int64, backend="pandas"),
        )


def test_zero_column_frame_keeps_nonempty_index() -> None:
    pd = pytest.importorskip("pandas")
    frame = nw.from_native(pd.DataFrame(index=[8, 9]))
    result = frame.with_columns(nw.new_series("a", [1, 2], backend="pandas")).to_native()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [1, 2]}, index=[8, 9]))

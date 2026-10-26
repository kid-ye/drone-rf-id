"""Smallest checks that fail if the label decode or feature shape breaks."""
import numpy as np

from preprocess import FAMILY, MODE, M, Q, segment_features


def test_labels():
    assert FAMILY["00000"] == 0 and FAMILY["11000"] == 3
    assert MODE["00000"] == 0 and MODE["11000"] == 9
    assert len(set(MODE.values())) == 10  # 10 distinct modes


def test_segment_shape_and_stitch():
    x = np.random.randn(100_000)
    xf = segment_features(x)
    assert xf.shape == (M // 2,)
    yf = segment_features(np.random.randn(100_000))
    yf = yf * (xf[-Q:].mean() / yf[:Q].mean())
    assert np.isfinite(yf).all()


if __name__ == "__main__":
    test_labels()
    test_segment_shape_and_stitch()
    print("ok")

import numpy as np
import pandas as pd

from shared.model.training import _cross_fit_probability_calibrator, predict_probabilities


class _DummyModel:
    """Return a constant positive-class probability so routing tests can stay deterministic."""

    def __init__(self, prob: float) -> None:
        self._prob = float(prob)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: ANN001 - sklearn-like interface
        probs = np.full(len(X), self._prob, dtype=float)
        return np.column_stack([1.0 - probs, probs])


def test_cross_fit_probability_calibrator_returns_finite_predictions():
    raw = np.linspace(0.01, 0.99, 200)
    y = pd.Series(([0, 1] * 100), dtype=int)
    calibrator, calibrated = _cross_fit_probability_calibrator(raw, y, folds=5)
    assert len(calibrated) == len(raw)
    assert np.isfinite(calibrated).all()
    assert np.isfinite(calibrator.calibrate(raw[:10])).all()


def test_cross_fit_probability_calibrator_time_windows_random_sampling_is_seeded():
    rng = np.random.default_rng(123)
    raw = rng.uniform(0.01, 0.99, 400)
    y = pd.Series((rng.random(400) < raw).astype(int), dtype=int)
    sample_index = pd.Index(np.arange(len(raw)))
    windows = [sample_index[i : i + 80] for i in range(0, len(raw), 80)]

    _, calibrated_a = _cross_fit_probability_calibrator(
        raw,
        y,
        sample_index=sample_index,
        time_windows=windows,
        randomize_within_windows=True,
        window_sample_fraction=0.5,
        random_state=7,
    )
    _, calibrated_b = _cross_fit_probability_calibrator(
        raw,
        y,
        sample_index=sample_index,
        time_windows=windows,
        randomize_within_windows=True,
        window_sample_fraction=0.5,
        random_state=7,
    )
    _, calibrated_c = _cross_fit_probability_calibrator(
        raw,
        y,
        sample_index=sample_index,
        time_windows=windows,
        randomize_within_windows=True,
        window_sample_fraction=0.5,
        random_state=11,
    )

    assert np.all(np.isfinite(calibrated_a))
    assert np.allclose(calibrated_a, calibrated_b)
    assert not np.allclose(calibrated_a, calibrated_c)


def test_predict_probabilities_leaves_unknown_bucket_unscored_without_fallback():
    feature_df = pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 1, "market_type": "WIN", "x": 1.0},
            {"market_id": "1.2", "selection_id": 2, "market_type": "PLACE", "x": 2.0},
        ]
    )
    model_bundle = {
        "models": {"WIN": _DummyModel(0.4)},
        "fallback": None,
        "feature_columns": ["x"],
    }
    calibrator_bundle = {"calibrators": {}, "fallback": None}

    probs = predict_probabilities(
        model_bundle,
        calibrator_bundle,
        feature_df,
        apply_plackett_luce=False,
    )
    assert probs.loc[feature_df.index[0]] == 0.4
    assert pd.isna(probs.loc[feature_df.index[1]])

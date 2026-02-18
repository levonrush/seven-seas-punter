import numpy as np
import pandas as pd

from shared.model.tab_translation import estimate_tab_odds_quantiles, quantile_to_column


class _ConstantPredictor:
    """Return a fixed odds value so translation tests can validate model wiring deterministically."""

    def __init__(self, value: float) -> None:
        """Store one constant output so each test can simulate a quantile-specific model."""
        self.value = float(value)

    def predict(self, X):  # noqa: ANN001 - mirrors sklearn-style estimator signature
        """Produce a constant-length vector so joblib-loaded models emulate simple quantile regressors."""
        return np.full(len(X), self.value, dtype=float)


def test_estimate_tab_odds_quantiles_fallback_is_monotonic():
    features = pd.DataFrame(
        {
            "market_id": ["1.1", "1.2"],
            "selection_id": [101, 202],
            "back_price_t10": [5.0, np.nan],
            "back_price_t5": [np.nan, 3.0],
        }
    )

    quantiles = [0.10, 0.50, 0.90]
    translated = estimate_tab_odds_quantiles(
        feature_df=features,
        cutoff_minutes=10,
        quantiles=quantiles,
        model_path=None,
        fallback_haircut=0.10,
        fallback_spread=0.10,
    )

    q10 = quantile_to_column(0.10)
    q50 = quantile_to_column(0.50)
    q90 = quantile_to_column(0.90)
    assert translated["tab_price_source"].eq("fallback_haircut").all()
    assert (translated[q10] <= translated[q50]).all()
    assert (translated[q50] <= translated[q90]).all()


def test_estimate_tab_odds_quantiles_uses_model_bundle(tmp_path):
    import joblib

    model_path = tmp_path / "tab_translation.joblib"
    bundle = {
        "feature_columns": ["feature_a"],
        "quantile_models": {
            "q10": _ConstantPredictor(2.2),
            "q50": _ConstantPredictor(2.5),
            "q90": _ConstantPredictor(2.9),
        },
    }
    joblib.dump(bundle, model_path)

    features = pd.DataFrame({"feature_a": [1.0, 2.0], "back_price_t10": [3.1, 4.2]})
    translated = estimate_tab_odds_quantiles(
        feature_df=features,
        cutoff_minutes=10,
        quantiles=[0.10, 0.50, 0.90],
        model_path=str(model_path),
    )

    assert translated["tab_price_source"].eq("model_quantiles").all()
    assert np.allclose(translated[quantile_to_column(0.10)].to_numpy(), 2.2)
    assert np.allclose(translated[quantile_to_column(0.50)].to_numpy(), 2.5)
    assert np.allclose(translated[quantile_to_column(0.90)].to_numpy(), 2.9)

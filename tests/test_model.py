"""
tests/test_model.py
Unit tests cho src/model.py
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.model import (
    evaluate,
    get_roc_data,
    load_model,
    save_model,
    train_baselines,
    train_random_forest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """
    Tạo dataset tổng hợp nhỏ (300 mẫu, 8 features)
    để test nhanh, không cần tải dữ liệu thật.
    """
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        random_state=42,
    )
    split = 240
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture(scope="module")
def trained_rf(synthetic_data):
    """Random Forest đã train sẵn để tái sử dụng trong nhiều test."""
    X_train, _, y_train, _ = synthetic_data
    return train_random_forest(X_train, y_train, tune=False)


# ── Tests: train_random_forest ────────────────────────────────────────────────

class TestTrainRandomForest:

    @pytest.mark.unit
    def test_returns_rf_classifier(self, trained_rf):
        assert isinstance(trained_rf, RandomForestClassifier)

    @pytest.mark.unit
    def test_model_is_fitted(self, trained_rf):
        """Model đã fit phải có thuộc tính estimators_."""
        assert hasattr(trained_rf, "estimators_")
        assert len(trained_rf.estimators_) > 0

    @pytest.mark.unit
    def test_can_predict_after_train(self, trained_rf, synthetic_data):
        _, X_test, _, _ = synthetic_data
        preds = trained_rf.predict(X_test)
        assert len(preds) == len(X_test)

    @pytest.mark.unit
    def test_predictions_are_binary(self, trained_rf, synthetic_data):
        _, X_test, _, _ = synthetic_data
        preds = trained_rf.predict(X_test)
        assert set(preds).issubset({0, 1})


# ── Tests: train_baselines ────────────────────────────────────────────────────

class TestTrainBaselines:

    @pytest.mark.unit
    def test_returns_dict(self, synthetic_data):
        X_train, _, y_train, _ = synthetic_data
        result = train_baselines(X_train, y_train)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_contains_both_models(self, synthetic_data):
        X_train, _, y_train, _ = synthetic_data
        result = train_baselines(X_train, y_train)
        assert "Logistic Regression" in result
        assert "Decision Tree" in result

    @pytest.mark.unit
    def test_all_models_fitted(self, synthetic_data):
        X_train, X_test, y_train, _ = synthetic_data
        baselines = train_baselines(X_train, y_train)
        for name, model in baselines.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{name} không dự đoán được"


# ── Tests: evaluate ───────────────────────────────────────────────────────────

class TestEvaluate:

    @pytest.mark.unit
    def test_returns_dict_with_required_keys(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "RF Test")
        for key in ["name", "accuracy", "f1", "auc", "confusion_matrix", "y_pred", "y_prob"]:
            assert key in result, f"Thiếu key '{key}' trong kết quả evaluate"

    @pytest.mark.unit
    def test_accuracy_in_valid_range(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "RF Test")
        assert 0.0 <= result["accuracy"] <= 1.0

    @pytest.mark.unit
    def test_f1_in_valid_range(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "RF Test")
        assert 0.0 <= result["f1"] <= 1.0

    @pytest.mark.unit
    def test_auc_in_valid_range(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "RF Test")
        assert 0.0 <= result["auc"] <= 1.0

    @pytest.mark.unit
    def test_confusion_matrix_shape(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "RF Test")
        assert result["confusion_matrix"].shape == (2, 2)

    @pytest.mark.unit
    def test_model_name_preserved(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        result = evaluate(trained_rf, X_test, y_test, "MyModel")
        assert result["name"] == "MyModel"


# ── Tests: get_roc_data ───────────────────────────────────────────────────────

class TestGetRocData:

    @pytest.mark.unit
    def test_returns_fpr_tpr(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        y_prob = trained_rf.predict_proba(X_test)[:, 1]
        fpr, tpr = get_roc_data(y_test, y_prob)
        assert len(fpr) > 0
        assert len(tpr) > 0

    @pytest.mark.unit
    def test_fpr_tpr_same_length(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        y_prob = trained_rf.predict_proba(X_test)[:, 1]
        fpr, tpr = get_roc_data(y_test, y_prob)
        assert len(fpr) == len(tpr)

    @pytest.mark.unit
    def test_values_in_0_1_range(self, trained_rf, synthetic_data):
        _, X_test, _, y_test = synthetic_data
        y_prob = trained_rf.predict_proba(X_test)[:, 1]
        fpr, tpr = get_roc_data(y_test, y_prob)
        assert np.all((fpr >= 0) & (fpr <= 1))
        assert np.all((tpr >= 0) & (tpr <= 1))


# ── Tests: save_model / load_model ───────────────────────────────────────────

class TestSaveLoadModel:

    @pytest.mark.unit
    def test_save_and_load_roundtrip(self, trained_rf, synthetic_data, tmp_path):
        """Model lưu xuống rồi load lên phải cho kết quả dự đoán giống nhau."""
        _, X_test, _, _ = synthetic_data
        path = str(tmp_path / "test_model.pkl")

        save_model(trained_rf, path)
        loaded = load_model(path)

        np.testing.assert_array_equal(
            trained_rf.predict(X_test),
            loaded.predict(X_test),
        )

    @pytest.mark.unit
    def test_saved_file_exists(self, trained_rf, tmp_path):
        import os
        path = str(tmp_path / "model.pkl")
        save_model(trained_rf, path)
        assert os.path.exists(path)
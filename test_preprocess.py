"""
tests/test_preprocess.py
Unit tests cho src/preprocess.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import fix_invalid_zeros, split_features_target, preprocess


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame mẫu có chứa giá trị 0 bất hợp lệ."""
    return pd.DataFrame({
        "Pregnancies":  [1, 2, 3, 0, 1],
        "Glucose":      [120, 0, 85, 150, 0],      # 2 giá trị 0 bất hợp lệ
        "BloodPressure":[70, 80, 0, 90, 60],        # 1 giá trị 0 bất hợp lệ
        "SkinThickness":[20, 0, 15, 30, 25],
        "Insulin":      [80, 100, 0, 0, 120],
        "BMI":          [28.5, 0.0, 32.1, 25.0, 30.0],
        "DiabetesPedigreeFunction": [0.5, 0.3, 0.8, 0.2, 0.6],
        "Age":          [30, 45, 25, 50, 35],
        "Outcome":      [1, 0, 0, 1, 0],
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Lưu sample_df ra CSV tạm để test hàm preprocess()."""
    path = tmp_path / "diabetes_test.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


# ── Tests: fix_invalid_zeros ──────────────────────────────────────────────────

class TestFixInvalidZeros:

    @pytest.mark.unit
    def test_zeros_replaced_in_glucose(self, sample_df):
        """Giá trị 0 trong cột Glucose phải được thay bằng median."""
        result = fix_invalid_zeros(sample_df)
        assert (result["Glucose"] == 0).sum() == 0, "Không được còn giá trị 0 trong Glucose"

    @pytest.mark.unit
    def test_zeros_replaced_with_median(self, sample_df):
        """Kiểm tra giá trị thay thế đúng là median của các giá trị != 0."""
        result = fix_invalid_zeros(sample_df)
        expected_median = sample_df.loc[sample_df["Glucose"] != 0, "Glucose"].median()
        # Các vị trí ban đầu là 0 phải có giá trị bằng median
        zero_positions = sample_df["Glucose"] == 0
        assert all(result.loc[zero_positions, "Glucose"] == expected_median)

    @pytest.mark.unit
    def test_non_zero_values_unchanged(self, sample_df):
        """Giá trị hợp lệ (khác 0) không bị thay đổi."""
        result = fix_invalid_zeros(sample_df)
        valid_mask = sample_df["Glucose"] != 0
        pd.testing.assert_series_equal(
            result.loc[valid_mask, "Glucose"],
            sample_df.loc[valid_mask, "Glucose"],
        )

    @pytest.mark.unit
    def test_original_df_not_modified(self, sample_df):
        """Hàm phải hoạt động trên bản copy, không sửa DataFrame gốc."""
        original_zeros = (sample_df["Glucose"] == 0).sum()
        fix_invalid_zeros(sample_df)
        assert (sample_df["Glucose"] == 0).sum() == original_zeros

    @pytest.mark.unit
    def test_pregnancies_zeros_kept(self, sample_df):
        """Cột Pregnancies KHÔNG nằm trong ZERO_INVALID_COLS → 0 được giữ nguyên."""
        result = fix_invalid_zeros(sample_df)
        assert (result["Pregnancies"] == 0).sum() == (sample_df["Pregnancies"] == 0).sum()


# ── Tests: split_features_target ─────────────────────────────────────────────

class TestSplitFeaturesTarget:

    @pytest.mark.unit
    def test_target_column_removed_from_X(self, sample_df):
        X, y = split_features_target(sample_df)
        assert "Outcome" not in X.columns

    @pytest.mark.unit
    def test_y_equals_outcome_column(self, sample_df):
        X, y = split_features_target(sample_df)
        pd.testing.assert_series_equal(y, sample_df["Outcome"])

    @pytest.mark.unit
    def test_X_has_correct_feature_count(self, sample_df):
        X, y = split_features_target(sample_df)
        # 9 cột tổng - 1 cột Outcome = 8 features
        assert X.shape[1] == sample_df.shape[1] - 1

    @pytest.mark.unit
    def test_row_count_preserved(self, sample_df):
        X, y = split_features_target(sample_df)
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)


# ── Tests: preprocess pipeline ───────────────────────────────────────────────

class TestPreprocessPipeline:

    @pytest.mark.integration
    def test_returns_six_values(self, sample_csv):
        result = preprocess(sample_csv)
        assert len(result) == 6, "preprocess() phải trả về 6 giá trị"

    @pytest.mark.integration
    def test_output_shapes_consistent(self, sample_csv):
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
            sample_csv, test_size=0.4
        )
        # Số features phải khớp
        assert X_train.shape[1] == X_test.shape[1]
        # Số samples phải khớp với label
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)

    @pytest.mark.integration
    def test_feature_names_correct(self, sample_csv):
        _, _, _, _, feature_names, _ = preprocess(sample_csv)
        assert "Outcome" not in feature_names
        assert len(feature_names) == 8

    @pytest.mark.integration
    def test_scaled_output_is_numpy(self, sample_csv):
        X_train, X_test, y_train, y_test, _, _ = preprocess(sample_csv)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)

    @pytest.mark.integration
    def test_scaler_returned_when_scale_true(self, sample_csv):
        from sklearn.preprocessing import StandardScaler
        _, _, _, _, _, scaler = preprocess(sample_csv, scale=True)
        assert scaler is not None
        assert isinstance(scaler, StandardScaler)

    @pytest.mark.integration
    def test_scaler_none_when_scale_false(self, sample_csv):
        _, _, _, _, _, scaler = preprocess(sample_csv, scale=False)
        assert scaler is None
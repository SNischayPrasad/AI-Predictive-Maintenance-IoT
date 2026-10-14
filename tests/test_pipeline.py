"""
Unit tests - run with:  pytest -v
Tests show recruiters you write reliable code, not just notebooks.
"""

import pandas as pd
import pytest

from src import config
from src.alert_system import get_alert_level
from src.data_simulator import simulate_sensor_data
from src.feature_engineering import add_features
from src.preprocessing import clean_data


@pytest.fixture(scope="module")
def small_df():
    return simulate_sensor_data(n_samples=2000, seed=1)


def test_simulator_shape_and_columns(small_df):
    assert len(small_df) == 2000
    for col in [config.COL_AIR_TEMP, config.COL_TORQUE, config.TARGET] + config.FAILURE_MODES:
        assert col in small_df.columns


def test_failure_rate_is_realistic(small_df):
    # Real factories: failures are rare (AI4I benchmark ~3.4%)
    assert 0.01 < small_df[config.TARGET].mean() < 0.08


def test_machine_failure_matches_failure_modes(small_df):
    any_mode = small_df[config.FAILURE_MODES].max(axis=1)
    assert (any_mode == small_df[config.TARGET]).all()


def test_cleaning_removes_leakage_columns(small_df):
    cleaned = clean_data(small_df)
    for col in config.DROP_COLUMNS:
        assert col not in cleaned.columns


def test_cleaning_fixes_sensor_glitches(small_df):
    dirty = small_df.copy()
    dirty.loc[0, config.COL_AIR_TEMP] = 9999      # impossible reading
    dirty.loc[1, config.COL_TORQUE] = None        # dropped packet
    cleaned = clean_data(dirty)
    assert cleaned[config.COL_AIR_TEMP].max() < 350
    assert cleaned[config.COL_TORQUE].notnull().all()


def test_feature_engineering_physics():
    raw = pd.DataFrame([{config.COL_TYPE: "L", config.COL_AIR_TEMP: 300.0,
                         config.COL_PROCESS_TEMP: 310.0, config.COL_RPM: 1500,
                         config.COL_TORQUE: 40.0, config.COL_TOOL_WEAR: 100,
                         config.TARGET: 0}])
    feats = add_features(raw).iloc[0]
    assert feats["Temp difference [K]"] == pytest.approx(10.0)
    assert feats["Strain [minNm]"] == pytest.approx(4000.0)
    assert feats["Power [W]"] == pytest.approx(40 * 1500 * 2 * 3.14159265 / 60, rel=1e-4)


@pytest.mark.parametrize("prob,level", [(0.05, "NORMAL"), (0.45, "WARNING"), (0.9, "CRITICAL")])
def test_alert_levels(prob, level):
    assert get_alert_level(prob) == level

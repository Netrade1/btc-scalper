import pandas as pd
from btc_scalper.data import generate_synthetic_ohlcv, load_csv, save_csv


def test_csv_round_trip(tmp_path):
    df = generate_synthetic_ohlcv(n_bars=200, seed=1)
    path = tmp_path / "bars.csv"
    save_csv(df, str(path))
    back = load_csv(str(path), 300)
    assert len(back) == len(df)
    assert (back.index.asi8 == df.index.as_unit("us").asi8).all()
    assert abs(back["close"] - df["close"].values).max() < 1e-9


def test_resample_sums_flow_and_drops_partial_buckets():
    from btc_scalper.data import resample_bars

    df = generate_synthetic_ohlcv(n_bars=12 * 10 + 5, seed=2)  # 10 full hours + 5 bars
    df = df.iloc[3:]  # first hour now partial too
    h = resample_bars(df, 3600)
    assert len(h) == 9
    first = df.loc[h.index[0]: h.index[0] + pd.Timedelta("55min")]
    assert len(first) == 12
    assert h["open"].iloc[0] == first["open"].iloc[0]
    assert h["close"].iloc[0] == first["close"].iloc[-1]
    assert h["high"].iloc[0] == first["high"].max()
    assert abs(h["buy_volume"].iloc[0] - first["buy_volume"].sum()) < 1e-9

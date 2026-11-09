from btc_scalper.data import generate_synthetic_ohlcv, load_csv, save_csv


def test_csv_round_trip(tmp_path):
    df = generate_synthetic_ohlcv(n_bars=200, seed=1)
    path = tmp_path / "bars.csv"
    save_csv(df, str(path))
    back = load_csv(str(path), 300)
    assert len(back) == len(df)
    assert (back.index.asi8 == df.index.as_unit("us").asi8).all()
    assert abs(back["close"] - df["close"].values).max() < 1e-9

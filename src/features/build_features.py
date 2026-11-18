from pathlib import Path
import pandas as pd


def build_features(df):
    """Build point-in-time-safe aggregate features for a chronological frame."""
    required = {"TransactionDT", "TransactionID", "TransactionAmt"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    frame = df.sort_values("TransactionDT").copy()
    frame["event_timestamp"] = pd.to_datetime(frame["TransactionDT"], unit="s", origin="1970-01-01")
    entity = frame["card1"].astype(str) if "card1" in frame else frame["TransactionID"].astype(str)
    frame["entity_id"] = entity

    grouped = frame.groupby("entity_id", sort=False)["TransactionAmt"]
    frame["velocity_count_1h"] = grouped.transform(lambda s: s.expanding().count().shift(1).fillna(0))
    frame["velocity_count_24h"] = frame["velocity_count_1h"]
    frame["velocity_count_7d"] = frame["velocity_count_1h"]
    frame["spend_1h"] = grouped.transform(lambda s: s.expanding().sum().shift(1).fillna(0))
    frame["spend_24h"] = frame["spend_1h"]
    frame["amt_vs_avg"] = frame["TransactionAmt"] / (frame["spend_1h"] / frame["velocity_count_1h"].clip(lower=1)).clip(lower=1)
    frame["rolling_max"] = grouped.transform(lambda s: s.expanding().max().shift(1).fillna(s.iloc[0]))
    frame["seconds_since_last"] = (
        frame.groupby("entity_id")["TransactionDT"].diff().fillna(0).clip(lower=0)
    )
    if "DeviceID" in frame:
        previous_device = frame.groupby("entity_id")["DeviceID"].shift()
        frame["new_device_flag"] = previous_device.isna().astype(int)  # first event is new
        frame.loc[previous_device.notna(), "new_device_flag"] = (
            frame.loc[previous_device.notna(), "DeviceID"] != previous_device.loc[previous_device.notna()]
        ).astype(int)
    else:
        frame["new_device_flag"] = 0
    if "P_emaildomain" in frame:
        previous_email = frame.groupby("entity_id")["P_emaildomain"].shift()
        frame["new_email_domain_flag"] = previous_email.isna().astype(int)
        frame.loc[previous_email.notna(), "new_email_domain_flag"] = (
            frame.loc[previous_email.notna(), "P_emaildomain"] != previous_email.loc[previous_email.notna()]
        ).astype(int)
    else:
        frame["new_email_domain_flag"] = 0
    columns = [
        "entity_id", "event_timestamp", "velocity_count_1h", "velocity_count_24h",
        "velocity_count_7d", "spend_1h", "spend_24h", "amt_vs_avg",
        "rolling_max", "seconds_since_last", "new_device_flag", "new_email_domain_flag",
    ]
    return frame[columns].copy()


def save_offline_features(df, output="data/features/offline_features.parquet"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = build_features(df)
    features.to_parquet(output, index=False)
    return features


def get_training_df(df):
    return build_features(df)


def get_online_features(entity_id, store=None):
    if store is not None and hasattr(store, "get_online_features"):
        return store.get_online_features(entity_rows=[{"entity_id": str(entity_id)}])
    return {"entity_id": str(entity_id), "feature_status": "missing-online-store"}

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64

account = Entity(name="entity_id", join_keys=["entity_id"])
offline_source = FileSource(
    path="data/features/offline_features.parquet",
    timestamp_field="event_timestamp",
)

account_features = FeatureView(
    name="account_features",
    entities=[account],
    ttl=None,
    schema=[
        Field(name="velocity_count_1h", dtype=Float32),
        Field(name="velocity_count_24h", dtype=Float32),
        Field(name="velocity_count_7d", dtype=Float32),
        Field(name="spend_1h", dtype=Float32),
        Field(name="spend_24h", dtype=Float32),
        Field(name="amt_vs_avg", dtype=Float32),
        Field(name="rolling_max", dtype=Float32),
        Field(name="seconds_since_last", dtype=Float32),
        Field(name="new_device_flag", dtype=Int64),
        Field(name="new_email_domain_flag", dtype=Int64),
    ],
    source=offline_source,
)

fraud_v1 = FeatureService(
    name="fraud_v1",
    features=[account_features],
)

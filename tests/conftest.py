import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append(
            {
                "monthly_fee": 20.0 + i * 0.5,
                "usage_hours": 150.0 + i,
                "support_requests": 0,
                "account_age_months": 24 + i,
                "failed_payments": 0,
                "region": ["europe", "asia", "america", "africa"][i % 4],
                "device_type": ["mobile", "desktop", "tablet"][i % 3],
                "payment_method": ["card", "paypal", "crypto"][i % 3],
                "autopay_enabled": 1,
                "churn": 0,
            }
        )
    for i in range(40):
        rows.append(
            {
                "monthly_fee": 40.0 + i * 0.5,
                "usage_hours": 5.0 + i * 0.2,
                "support_requests": 6 + (i % 4),
                "account_age_months": 1 + (i % 6),
                "failed_payments": 2 + (i % 3),
                "region": ["europe", "asia", "america", "africa"][i % 4],
                "device_type": ["mobile", "desktop", "tablet"][i % 3],
                "payment_method": ["card", "paypal", "crypto"][i % 3],
                "autopay_enabled": 0,
                "churn": 1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def valid_client_payload() -> dict:
    return {
        "monthly_fee": 25.5,
        "usage_hours": 120.0,
        "support_requests": 2,
        "account_age_months": 14,
        "failed_payments": 0,
        "autopay_enabled": 1,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
    }
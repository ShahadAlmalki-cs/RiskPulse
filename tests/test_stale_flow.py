import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json


def _fingerprint(values):
    """Mirror of app.inputs_fingerprint (importing app.py requires Streamlit runtime)."""
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def test_fingerprint_changes_when_loan_changes():
    base = {
        "person_age": 30, "person_income": 60000, "person_emp_length": 5.0,
        "person_home_ownership": "RENT", "cb_person_cred_hist_length": 5,
        "cb_person_default_on_file": "N", "loan_amnt": 10000, "loan_intent": "PERSONAL",
        "customer_name": "Sara",
    }
    changed = {**base, "loan_amnt": 30000}
    assert _fingerprint(base) != _fingerprint(changed)
    assert _fingerprint(base) == _fingerprint(dict(base))  # stable for identical inputs


def test_fingerprint_is_order_insensitive():
    a = {"loan_amnt": 10000, "person_age": 30}
    b = {"person_age": 30, "loan_amnt": 10000}
    assert _fingerprint(a) == _fingerprint(b)

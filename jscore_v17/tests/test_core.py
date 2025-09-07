import math
from app import compute_apri, compute_fib4, interpret_j_score, is_bt_normal, bucket_k

def test_compute_apri():
    assert round(compute_apri(40, 40, 200), 4) == 0.5

def test_compute_fib4():
    assert round(compute_fib4(50, 100, 50, 200), 4) == round((50*100)/(200*math.sqrt(50)), 4)

def test_is_bt_normal():
    assert is_bt_normal(3.0) and is_bt_normal(12.0)
    assert not is_bt_normal(2.9) and not is_bt_normal(12.1)

def test_bucket_k():
    assert bucket_k(1.0) == 1.0
    assert bucket_k(1.4) == 1.5
    assert bucket_k(2.1) == 2.0

def test_interpret_normal_k_toggle():
    assert interpret_j_score(0.4, 1.2, 5.0, 1.0).startswith("Normale")
    assert interpret_j_score(0.4, 1.2, 5.0, 1.5).startswith("Normale")
    assert interpret_j_score(0.4, 1.2, 5.0, 2.0).startswith("Consulter")

def test_interpret_abnormal_any_k():
    assert interpret_j_score(0.6, 1.2, 5.0, 1.0).startswith("Consulter")
    assert interpret_j_score(0.4, 2.0, 5.0, 1.0).startswith("Consulter")

def test_interpret_fibrose_bt_anormale():
    assert interpret_j_score(1.0, 2.0, 20.0, 1.0).startswith("Consulter")

def test_interpret_cirrhose_bt_anormale():
    assert interpret_j_score(3.0, 4.0, 20.0, 1.0).startswith("Urgemment")

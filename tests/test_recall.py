import numpy as np
from metrics import recall_at_k, kth_dist2

K = 3
Q = np.zeros((1, 1), dtype=np.float32)

def setup(vals):
    db = np.array(vals, dtype=np.float32)[:, None]
    d2 = (db[:, 0] ** 2)
    exact = np.argsort(d2, kind="stable")[None, :K]
    return db, kth_dist2(Q, db, exact, K)

def rec(ids, vals):
    db, kd = setup(vals)
    return recall_at_k(np.array([ids]), Q, db, kd, k=K)

def test_perfect():
    assert rec([0, 1, 2], [1, 2, 3, 9, 8, 7]) == 1.0

def test_all_wrong():
    assert rec([3, 4, 5], [1, 2, 3, 9, 8, 7]) == 0.0

def test_partial():
    assert abs(rec([0, 4, 5], [1, 2, 3, 9, 8, 7]) - 1 / 3) < 1e-9

def test_tie_at_kth_neighbour():
    # distances 1,2,3,3,4: ids 2 and 3 tie for 3rd place, either is correct
    vals = [1, 2, 3, -3, 4]
    assert rec([0, 1, 2], vals) == 1.0
    assert rec([0, 1, 3], vals) == 1.0

def test_duplicates_count_once():
    assert abs(rec([0, 0, 0], [1, 2, 3, 9]) - 1 / 3) < 1e-9

def test_minus_one_is_a_miss():
    # db[-1] would be a true neighbour (value 1), so a buggy negative index would score a hit
    assert rec([-1, -1, -1], [2, 3, 4, 1]) == 0.0
    assert abs(rec([3, -1, -1], [2, 3, 4, 1]) - 1 / 3) < 1e-9
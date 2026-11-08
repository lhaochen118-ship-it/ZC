#!/usr/bin/env python3
"""src/sqlesc.py, the <sql> escape (docs/08 WP12): a SELECT over the canonical tables gives the toolkit's number (revenue of a
month, order count, top item) on several shops; the answer type follows the answer column's name, then the expected type, then
the shape; nothing but one reading query runs (no files, no writes, no second statement); the tables are not modified.
Usage: test_sqlesc.py <train_split> <scratch>"""
import os, sys, pickle, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sqlesc, dm


def main():
    split, S = sys.argv[1], sys.argv[2]
    shops = sorted(glob.glob(os.path.join(S, "canon", "*.pkl")))[:4]
    assert shops, "no canonical tables in <scratch>/canon"
    n = 0
    for p in shops:
        can = pickle.load(open(p, "rb")); C = can["C"]
        f = sqlesc.make_sql_answer(C, {"Bảng 1": C["orders"].head(3)})
        day = str(C["orders"].order_date.max().date())
        api = dm.api(dm.Shop(C, day, {"data_end": can.get("data_end")}), day)
        m = C["orders"].order_date.max()
        y, mo = m.year, m.month
        start, end = f"{y}-{mo:02d}-01", (f"{y + (mo == 12)}-{mo % 12 + 1:02d}-01")
        r = f(f"SELECT SUM(revenue) AS number FROM orders WHERE valid AND order_date >= '{start}' AND order_date < '{end}'")
        want = api["revenue"](api["month"](y, mo))
        assert r["type"] == "number" and abs(r["answer"] - want) < 1, (p, r, want)
        r = f(f"SELECT COUNT(*) FROM orders WHERE valid AND order_date >= '{start}' AND order_date < '{end}'")
        assert r == {"type": "number", "answer": api["n_orders"](api["month"](y, mo))}, (p, r)
        r = f("SELECT COUNT(*) AS percent FROM raw_bảng_1"); assert r == {"type": "percent", "answer": 3}, r
        n += 1
    r = f("SELECT 2 AS x", "percent"); assert r == {"type": "percent", "answer": 2}, r
    r = f("SELECT order_id FROM orders LIMIT 2"); assert r["type"] == "list" and len(r["answer"]) == 2, r
    r = f("SELECT order_id FROM orders LIMIT 1", "list"); assert r["type"] == "list" and len(r["answer"]) == 1, r
    r = f("SELECT order_id, revenue FROM orders LIMIT 2"); assert r["type"] == "table" and len(r["answer"]["rows"]) == 2, r
    r = f("SELECT MAX(order_date) AS date FROM orders"); assert r["type"] == "date" and len(r["answer"]) == 10, r
    r = f("-- top\nWITH t AS (SELECT 1 AS a) SELECT a = 1 FROM t"); assert r == {"type": "boolean", "answer": True}, r
    before = len(C["orders"])
    for bad in ["SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM '/etc/passwd'", "COPY orders TO 'x.csv'", "DROP TABLE orders",
                "SELECT 1; DELETE FROM orders", "ATTACH 'x.db'", "INSTALL httpfs", "SELECT * FROM glob('/*')"]:
        try: f(bad); raise AssertionError(f"not refused: {bad}")
        except AssertionError: raise
        except Exception: pass
    assert len(C["orders"]) == before
    print(f"{n} shops: SQL answers = toolkit; typing and refusals ok")
    print("test_sqlesc PASS")


if __name__ == "__main__":
    main()

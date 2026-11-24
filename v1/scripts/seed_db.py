"""Create data/acme.db with a fake `employees` table. Safe to re-run (rebuilds from scratch)."""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/acme.db")

EMPLOYEES = [
    # id, name,              email,                       department,    level, location,    hire_date,    manager_id, salary_inr
    (1,  "Ananya Rao",       "ananya.rao@acme.example",    "Engineering", "L7", "Bengaluru", "2016-02-15", None, 6500000),
    (2,  "Priya Sharma",     "priya.sharma@acme.example",  "Engineering", "L5", "Bengaluru", "2019-06-10", 1,    3800000),
    (3,  "Rahul Verma",      "rahul.verma@acme.example",   "Engineering", "L3", "Hyderabad", "2026-08-01", 2,    1400000),
    (4,  "Arjun Nair",       "arjun.nair@acme.example",    "Engineering", "L4", "Bengaluru", "2022-01-17", 2,    2200000),
    (5,  "Meera Iyer",       "meera.iyer@acme.example",    "Engineering", "L4", "Hyderabad", "2021-09-06", 2,    2300000),
    (6,  "Kavya Reddy",      "kavya.reddy@acme.example",   "Sales",       "L6", "Mumbai",    "2018-04-02", None, 4200000),
    (7,  "Vikram Singh",     "vikram.singh@acme.example",  "Sales",       "L4", "Delhi NCR", "2023-03-20", 6,    1800000),
    (8,  "Sneha Kulkarni",   "sneha.kulkarni@acme.example","Sales",       "L3", "Mumbai",    "2026-06-15", 6,    1100000),
    (9,  "Rohan Das",        "rohan.das@acme.example",     "Sales",       "L5", "Bengaluru", "2020-11-09", 6,    3000000),
    (10, "Fatima Khan",      "fatima.khan@acme.example",   "HR",          "L6", "Hyderabad", "2017-07-24", None, 3900000),
    (11, "Deepak Menon",     "deepak.menon@acme.example",  "HR",          "L3", "Hyderabad", "2024-10-01", 10,   1000000),
    (12, "Aisha Thomas",     "aisha.thomas@acme.example",  "Finance",     "L5", "Mumbai",    "2019-12-02", None, 3500000),
    (13, "Karthik Pillai",   "karthik.pillai@acme.example","Finance",     "L4", "Mumbai",    "2022-05-16", 12,   2000000),
    (14, "Nisha Gupta",      "nisha.gupta@acme.example",   "IT",          "L5", "Bengaluru", "2020-02-03", None, 3100000),
    (15, "Sameer Joshi",     "sameer.joshi@acme.example",  "IT",          "L3", "Bengaluru", "2026-09-14", 14,   1200000),
]

if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)          # rebuild from scratch every time
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE employees (
                id          INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL UNIQUE,
                department  TEXT NOT NULL,      -- Engineering, Sales, HR, Finance, IT
                level       TEXT NOT NULL,      -- L3 (junior) ... L7 (senior)
                location    TEXT NOT NULL,
                hire_date   TEXT NOT NULL,      -- ISO date YYYY-MM-DD
                manager_id  INTEGER REFERENCES employees(id),
                salary_inr  INTEGER NOT NULL    -- RESTRICTED: must never be readable by the bot
            )""")
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EMPLOYEES)
    print(f"Created {DB_PATH} with {len(EMPLOYEES)} employees")
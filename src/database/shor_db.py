import os

DB_PATH = os.path.join("shor_database.txt")

def append_shor_record(N, p, q, a, r):
    """
    Append a Shor result entry to the database file.
    Format: N=?, p=?, q=?, a=?, r=?
    """
    line = f"N={N}, p={p}, q={q}, a={a}, r={r}\n"

    with open(DB_PATH, "a", encoding="utf-8") as db:
        db.write(line)

    return line

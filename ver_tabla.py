import sqlite3
conn = sqlite3.connect("control_acceso.db")
row = conn.execute("SELECT sql FROM sqlite_master WHERE name='accesos'").fetchone()
print(row[0])
conn.close()
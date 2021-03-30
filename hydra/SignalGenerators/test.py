import duckdb

cursor = duckdb.connect(database=f"output\signals\XBTUSD Aroon.entry.join.duck")

cursor.execute("""DESCRIBE aroon_the_world;""")
print(cursor.fetchall())
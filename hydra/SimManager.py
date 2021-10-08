import pyarrow.parquet as pq


def get_simulation_id(id_base, entry, exit):
    return entry * id_base + exit
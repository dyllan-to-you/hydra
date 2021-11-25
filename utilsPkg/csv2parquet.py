import glob, os
import sys
import pathlib
import pyarrow as pa
from pyarrow import csv
import pyarrow.parquet as pq


print(str(sys.argv[1]))
for file in glob.glob(str(sys.argv[1])):
    path = pathlib.Path(file).resolve()
    name = path.stem
    dir = path.parent
    print(file)
    table = csv.read_csv(path)
    pq.write_table(table, dir.joinpath('parq', f"{name}.parq"))

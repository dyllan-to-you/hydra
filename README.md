Scalene w/html outputs

```sh
poetry run scalene --html --outfile scalene.html <filename>
```

Recursive folder comparison
```sh
# Compares existence and hash
diff -qr enviroem enviroemOneRun

# Compares file sizes too
diff -y <(cd enviroem && du -a | awk '{ print $2 " " $1}' | sort -k1) <(cd enviroemOneRun && du -a | awk '{ print $2 " " $1}' | sort -k1)
```

DF Comparison
```py
import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def parse(output):
    split = output.split()
    file1 = outputDir / split[1]
    file2 = outputDir / split[3]
    return pd.read_parquet(file1).sort_index(), pd.read_parquet(file2).sort_index()

def compareDfs(args):
    (df1, df2) = args
    return df1.compare(df2)

outputDir = Path('./output')

string = "Files enviro-chunky-7/year=2017/month=9/day=9/96.xtrp.parq and enviro-origin-7/year=2017/month=9/day=9/96.xtrp.parq differ"

df1, df2 = parse(string)
res = compareDfs((df1, df2))

```
import path from "path";
// @ts-ignore
import duckdb from "duckdb";

console.log(new Date(), "Initializing");
const root = path.dirname(require?.main?.filename ?? "");
const dbPath = path.join(root, "../../data/prices.duckpq");
console.log(dbPath);
const db = new duckdb.Database(":memory:");
db.exec(`IMPORT DATABASE '${dbPath}'`);
console.log(new Date(), "Initialized");
main();

async function runQuery(query: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    db.all(query, (err: any, res: any) => {
      if (err) {
        console.error(`Error:`, err);
        reject(err);
      }
      console.log(new Date(), res.length);
      resolve(res);
    });
  });
}

async function main() {
  console.log(new Date(), "q1");
  await runQuery(
    "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-01-01' and timestamp < '2019-01-02'"
  );
  console.log(new Date(), "q2");
  await runQuery(
    "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-06-05' and timestamp < '2019-06-07'"
  );
  console.log(new Date(), "q3");
  await runQuery(
    "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-03-01' and timestamp < '2019-03-01 12:00'"
  );
  console.log(new Date(), "q4");
  await runQuery(
    "SELECT * from binance WHERE ticker='XBTUSD' AND timestamp > '2020-06-01' and timestamp < '2020-06-03 12:00'"
  );

  //@ts-ignore
  db.close();
}

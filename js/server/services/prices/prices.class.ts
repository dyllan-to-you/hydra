import path from "path";
import duckdb from "duckdb";
import { Knex } from "knex";
import { BadRequest, MethodNotAllowed } from "@feathersjs/errors";
import { Id, NullableId, Params, ServiceMethods } from "@feathersjs/feathers";
import { Service as KnexService, KnexServiceOptions } from "feathers-knex";
import { Application } from "../../declarations";

type Data = any;

const tickerPairs = [{ binance: "BTCUSD", kraken: "XBTUSD" }];
const tickerMap = tickerPairs.reduce((a, e) => {
  const { binance, kraken } = e;
  a.set(binance, e);
  a.set(kraken, e);
  return a;
}, new Map());

export class Prices implements ServiceMethods<any> {
  app: Application;

  binance: KnexService;
  kraken: KnexService;

  db: duckdb.Database;
  //eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(options: Partial<KnexServiceOptions>, app: Application) {
    this.app = app;
    this.binance = new KnexService({
      ...options,
      name: "binance",
    });
    this.kraken = new KnexService({
      ...options,
      name: "kraken",
    });

    const root = path.dirname(require?.main?.filename ?? "");
    const dbPath = path.join(root, "../../data/prices.duckpq");
    console.log(dbPath);
    this.db = new duckdb.Database(":memory:");

    this.db.exec(`IMPORT DATABASE '${dbPath}'`);
  }

  async find(params?: Params): Promise<Data> {
    // console.log("prices::find", this.createQuery(params).toString());
    throw new MethodNotAllowed("Currency pair is required");
  }

  createQuery(params: Partial<Params> = {}) {
    const { filters, query } = this.binance.filterQuery(params);
    let q: Knex | Knex.QueryBuilder = this.binance.db(params);

    // $select uses a specific find syntax, so it has to come first.
    q = filters.$select // always select the id field, but make sure we only select it once
      ? q.select([...new Set(filters.$select)])
      : q.select([`*`]);

    // build up the knex query out of the query params
    // @ts-ignore: use untyped method
    this.binance.knexify(q, query);

    // Handle $sort
    if (filters.$sort) {
      Object.keys(filters.$sort).forEach((key) => {
        q = q.orderBy(key, filters.$sort[key] === 1 ? "asc" : "desc");
      });
    }

    return q;
  }

  runQuery(query: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      this.db.all(query, (err: any, res: any) => {
        if (err) {
          console.error(`Error:`, err);
          reject(err);
        }
        resolve(res);
      });
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async get(id: Id, params?: Params): Promise<Data[]> {
    const tickers = tickerMap.get(id);
    if (tickers == null) {
      throw new BadRequest("Invalid ticker " + id);
    }
    const { binance: binanceTicker, kraken: krakenTicker } = tickers;

    let resolution = 1;

    if (params?.query?.resolution) {
      resolution = params.query.resolution;
      delete params.query.resolution;
    }

    const query = this.createQuery(params);

    if (params?.query?.timestamp == null) {
      const now = new Date();
      const then = new Date(now.getTime() - 60 * 60 * 1000);
      query.whereBetween("timestamp", [then, now]);
    }

    const runQuery = (name: string, ticker: string) => {
      const queryStr = query
        .clone()
        .from(name)
        .where({ ticker })
        .toString()
        .replace(/`/g, '"');

      console.log(queryStr);
      return this.runQuery(queryStr);
    };

    console.log("prices::get", id);
    console.log(
      "between",
      await this.runQuery(
        `SELECT DISTINCT ticker from kraken 
        where "timestamp" > '2019-01-01' and "timestamp" < '2019-01-02' `
      )
    );
    console.log(
      "global",
      await this.runQuery(`SELECT DISTINCT ticker from kraken`)
    );
    // console.log(
    //   await this.runQuery(
    //     `SELECT * from kraken ORDER BY timestamp DESC limit 1;`
    //   )
    // );
    console.log((await runQuery("binance", binanceTicker)).length);
    console.log((await runQuery("kraken", krakenTicker)).length);

    throw new MethodNotAllowed("Invalid method for prices");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async create(data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for prices");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async update(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for prices");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async patch(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for prices");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async remove(id: NullableId, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for prices");
  }
}

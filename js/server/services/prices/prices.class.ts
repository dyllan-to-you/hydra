import path from "path";
import duckdb from "duckdb";
import { Knex } from "knex";
import { BadRequest, MethodNotAllowed } from "@feathersjs/errors";
import { Id, NullableId, Params, ServiceMethods } from "@feathersjs/feathers";
import { Service as KnexService, KnexServiceOptions } from "feathers-knex";
import { Application } from "../../declarations";

interface Data {
  ts: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const tickerPairs = [{ binance: "BTCUSD", kraken: "XBTUSD" }];
const tickerMap = tickerPairs.reduce((a, e) => {
  const { binance, kraken } = e;
  a.set(binance, e);
  a.set(kraken, e);
  return a;
}, new Map());

export class Prices implements ServiceMethods<any> {
  app: Application;

  knexServ: KnexService;

  db: duckdb.Database;
  //eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(options: Partial<KnexServiceOptions>, app: Application) {
    this.app = app;
    this.knexServ = new KnexService({
      ...options,
      name: "binance",
    });

    const dbPath = path.join(__dirname, "../../../../data/prices.duckpq");
    console.log(dbPath);
    this.db = new duckdb.Database(":memory:");

    this.db.exec(`IMPORT DATABASE '${dbPath}'`);
  }

  async find(params?: Params): Promise<Data> {
    // console.log("prices::find", this.createQuery(params).toString());
    throw new MethodNotAllowed("Currency pair is required");
  }

  createQuery(params: Partial<Params> = {}) {
    const { filters, query } = this.knexServ.filterQuery(params);
    let q: Knex | Knex.QueryBuilder = this.knexServ.db(params);

    // $select uses a specific find syntax, so it has to come first.
    q = filters.$select // always select the id field, but make sure we only select it once
      ? q.select([...new Set(filters.$select)])
      : q.select([`*`]);

    // build up the knex query out of the query params
    // @ts-expect-error: use untyped method
    this.knexServ.knexify(q, query);

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

    console.log("prices::get", id);

    const queries = Object.entries(tickers).map(([name, ticker]) =>
      query.clone().from(name).where({ ticker })
    );
    const [firstQuery, ...remQueries] = queries;
    const unionQuery = firstQuery.unionAll(remQueries);

    const resampled = this.knexServ.knex
      .queryBuilder()
      .with("unioned", unionQuery)
      .with(
        "resampled",
        this.knexServ.knex
          .queryBuilder()
          .select(
            "timestamp",
            this.knexServ.knex.raw(`
          to_timestamp(
            CAST(
              floor(
                EXTRACT(epoch FROM timestamp) 
                / EXTRACT(epoch FROM interval '${resolution} minutes')
              ) * EXTRACT(epoch FROM interval '${resolution} minutes') 
            AS BIGINT)
          ) as ts
        `),
            "open",
            "high",
            "low",
            "close",
            "volume"
          )
          .from("unioned")
          .orderBy("timestamp", "asc")
      );

    const fin = resampled
      .select(
        "ts",
        this.knexServ.knex.raw("first(open) as open"),
        this.knexServ.knex.raw("max(high) as high"),
        this.knexServ.knex.raw("min(low) as low"),
        this.knexServ.knex.raw("last(close) as close"),
        this.knexServ.knex.raw("sum(volume) as volume")
      )
      .from("resampled")
      .groupBy("ts");

    const queryStr = fin.toString().replace(/`/g, '"');

    // console.log(queryStr);
    const result = await this.runQuery(queryStr);
    // console.log(result);

    // throw new MethodNotAllowed("Invalid method for prices");
    return result;
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

import path from "path";
import { MethodNotAllowed } from "@feathersjs/errors";
import {
  Id,
  NullableId,
  Paginated,
  Params,
  ServiceMethods,
} from "@feathersjs/feathers";
import duckdb from "duckdb";
import { Knex } from "knex";
import { Service as KnexService, KnexServiceOptions } from "feathers-knex";
import { Application } from "../../declarations";

const outputSet = "enviro-chunky-90-365";
const outputPath = path.resolve(__dirname, "../../../../output", outputSet);
const outputs = path.join(
  outputPath,
  "year=*",
  "month=*",
  "day=*",
  "*[!.xtrp].parq"
);

export interface Data {
  minPerCycle: number;
  deviance: number;
  ifft_extrapolated_wavelength: number;
  ifft_extrapolated_amplitude: number;
  ifft_extrapolated_deviance: number;
  first_extrapolated: number;
  first_extrapolated_date: Date;
  first_extrapolated_isup: boolean;
  startDate: Date;
  endDate: Date;
  window: number;
  window_original: number;
  trend_deviance: number;
  trend_slope: number;
  trend_intercept: number;
  rootNumber: number;
}

interface ServiceOptions {}

function duckdbInit(db: duckdb.Database, exportTable = null): Promise<true> {
  return new Promise((resolve, reject) => {
    let query = `CREATE TABLE fft_indicator AS SELECT
      "minPerCycle",
      "deviance",
      "ifft_extrapolated_wavelength",
      "ifft_extrapolated_amplitude",
      "ifft_extrapolated_deviance",
      "first_extrapolated",
      "first_extrapolated_date",
      "first_extrapolated_isup",
      "startDate",
      "endDate",
      "window",
      "window_original",
      "trend_deviance",
      "trend_slope",
      "trend_intercept",
      "rootNumber"
    FROM parquet_scan(['${outputs}']);
    CREATE INDEX fft_indicator_idx ON fft_indicator(first_extrapolated_date, ifft_extrapolated_wavelength, rootNumber);
    `;
    if (exportTable != null) {
      query += `EXPORT DATABASE '${outputPath}-duckdb-parq' (${
        typeof exportTable === "string" ? exportTable : ""
      });
      `;
    }
    db.exec(query, (err: any, res: any) => (err ? reject(err) : resolve(true)));
  });
}

export class FftIndicator implements ServiceMethods<Data> {
  app: Application;
  options: ServiceOptions;
  knexServ: KnexService;
  db: duckdb.Database;
  dbInitPromise: Promise<boolean>;

  constructor(options: Partial<KnexServiceOptions>, app: Application) {
    this.options = options;
    this.app = app;
    this.knexServ = new KnexService({
      ...options,
      name: "fft_indicator",
    });

    console.log(outputs);

    this.db = new duckdb.Database(":memory:");
    this.dbInitPromise = duckdbInit(this.db);
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

  runQuery(query: Knex | Knex.QueryBuilder): Promise<any[]> {
    const queryStr = query.toString().replace(/`/g, '"');
    console.log("fft-indicator", queryStr);
    return new Promise((resolve, reject) => {
      this.db.all(queryStr, (err: any, res: any) => {
        if (err) {
          console.error(`Error:`, err);
          reject(err);
        }
        resolve(res);
      });
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async find(params?: Params): Promise<Data[] | Paginated<Data>> {
    await this.dbInitPromise;

    let resolution = 1;

    if (params?.query?.resolution) {
      resolution = params.query.resolution;
      delete params.query.resolution;
    }

    const query = this.createQuery(params);

    // if (params?.query?.timestamp == null) {
    //   const now = new Date();
    //   const then = new Date(now.getTime() - 60 * 60 * 1000);
    //   query.whereBetween("timestamp", [then, now]);
    // }

    const result = await this.runQuery(query);
    // console.log(result);

    // throw new MethodNotAllowed("Invalid method for prices");
    return result;
  }

  async get(id: string, params?: Params): Promise<any> {
    switch (id) {
      case "info":
      default: {
        const [rootNumbers, wavelengths] = await Promise.all([
          this.runQuery(
            this.createQuery(params)
              .clear("select")
              .distinct("rootNumber")
              .orderBy("rootNumber")
          ).then((data) => data.map((row) => row["rootNumber"])),
          this.runQuery(
            this.createQuery(params)
              .clear("select")
              .distinct("ifft_extrapolated_wavelength")
              .orderBy("ifft_extrapolated_wavelength")
          ).then((data) =>
            data.map((row) => row["ifft_extrapolated_wavelength"])
          ),
        ]);

        // console.log({
        //   rootNumbers,
        //   wavelengths,
        // });
        return {
          rootNumbers,
          wavelengths,
        };
        break;
      }
    }
    throw new MethodNotAllowed("Invalid method for fft-indicator");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async create(data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for fft-indicator");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async update(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for fft-indicator");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async patch(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for fft-indicator");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async remove(id: NullableId, params?: Params): Promise<Data> {
    throw new MethodNotAllowed("Invalid method for fft-indicator");
  }
}

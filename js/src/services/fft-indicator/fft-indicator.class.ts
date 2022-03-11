import path from "path";
import * as dfd from "danfojs-node";
import { MethodNotAllowed } from "@feathersjs/errors";
import {
  Id,
  NullableId,
  Paginated,
  Params,
  ServiceMethods,
} from "@feathersjs/feathers";
import duckdb from "duckdb";
import { Application } from "../../declarations";

const METHODS = {
  $or: "orWhere",
  $and: "andWhere",
  $ne: "whereNot",
  $in: "whereIn",
  $nin: "whereNotIn",
};

const OPERATORS: Record<string, string> = {
  $lt: "<",
  $lte: "<=",
  $gt: ">",
  $gte: ">=",
  $like: "like",
  $notlike: "not like",
  $ilike: "ilike",
};

export interface Data {
  minPerCycle: any;
  deviance: any;
  ifft_extrapolated_wavelength: any;
  ifft_extrapolated_amplitude: any;
  ifft_extrapolated_deviance: any;
  first_extrapolated: any;
  first_extrapolated_date: any;
  first_extrapolated_isup: any;
  startDate: any;
  endDate: any;
  window: any;
  window_original: any;
  trend_deviance: any;
  trend_slope: any;
  trend_intercept: any;
  rootNumber: any;
}

interface ServiceOptions {}

const outputSet = "enviro-chunky-90-365";
const outputPath = path.resolve(__dirname, "../../../../output", outputSet);
const outputs = path.join(
  outputPath,
  "year=*",
  "month=*",
  "day=*",
  "*[!.xtrp].parq"
);

function danfo(): Promise<dfd.DataFrame> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const db = new duckdb.Database(":memory:"); // or a file name for a persistent DB

    db.all(
      `SELECT 
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
      `,
      (err: any, res: any) => {
        if (err) {
          console.error("ERROR", err);
          reject(err);
        } else {
          const df = new dfd.DataFrame(res, {}).setIndex({
            column: "first_extrapolated_date",
          });
          console.log("Memory Load", (Date.now() - start) / 1000);
          df.head(10).print();

          const queryStart = Date.now();
          const qResult = df.loc({ rows: [`2018-01-01:2018-02-01`] }); //df.query(df.index.gt('2018-01-01'))
          console.log(qResult, (Date.now() - queryStart) / 1000);

          resolve(df);
        }
      }
    );
  });
}

function duckdbInit(
  path = ":memory:",
  exportTable = null
): Promise<duckdb.Database> {
  return new Promise(async (resolve, reject) => {
    const db = new duckdb.Database(path);
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
    db.exec(query, (err: any, res: any) => (err ? reject(err) : resolve(db)));
  });
}

export class FftIndicator implements ServiceMethods<Data> {
  app: Application;
  options: ServiceOptions;
  // df: Promise<dfd.DataFrame>;
  db: Promise<duckdb.Database>;

  constructor(options: ServiceOptions = {}, app: Application) {
    this.options = options;
    this.app = app;

    /*
      path.join(outputPath, "accumulated_parents.parq")
      path.join(outputPath, "year=2018", "month=1", "day=1", "1.parq")
      path.join(outputPath, "**", "*.parq")
    */
    console.log(outputs);

    // memory load into danfo
    // this.df = danfo();

    // this.db = Promise.resolve(new duckdb.Database(`${outputPath}-duckdb-parq`));

    this.db = duckdbInit();
  }

  objConditionParser(
    column: string,
    condition: Record<keyof typeof OPERATORS, string | number> | string | number
  ) {
    if (typeof condition != "object") {
      condition = { $eq: condition };
    }

    const clause = Object.entries(condition).map(([op, val]) => {
      const operator = OPERATORS[op] ?? "=";
      return `${column} ${operator} ${this.segmentParser(val)}`;
    });

    console.log("parser", clause);
    return `(${clause.join(" and ")})`;
  }

  segmentParser(value: any) {
    switch (typeof value) {
      case "string":
        return `'${value}'`;
      case "number":
        return `${value}`;
      default:
        return value;
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async find(params?: Params): Promise<Data[] | Paginated<Data>> {
    const { query } = params ?? {};
    const { $select, ...remQueries } = query ?? {};

    const select = $select ? $select.join(", ") : "*";

    let clause = Object.entries(remQueries)
      .map((e) => {
        const [prop, condition] = e;
        console.log(prop, condition);
        return this.objConditionParser(prop, condition);
      })
      .filter((e) => e != null)
      .join(" AND ");

    console.log("clause", clause);
    if (clause == null || clause == "") {
      clause = `first_extrapolated_date BETWEEN '2018-01-01' AND '2018-01-02'`;
    }

    const connection = (await this.db).connect();

    const sqlQuery = `SELECT ${select} FROM fft_indicator WHERE ${clause}`;
    console.debug(sqlQuery);

    return new Promise((resolve, reject) => {
      connection.all(sqlQuery, (err: Error, res: any[]) =>
        err ? reject(err) : resolve(res)
      );
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async get(id: Id, params?: Params): Promise<Data> {
    throw MethodNotAllowed;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async create(data: Data, params?: Params): Promise<Data> {
    throw MethodNotAllowed;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async update(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw MethodNotAllowed;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async patch(id: NullableId, data: Data, params?: Params): Promise<Data> {
    throw MethodNotAllowed;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async remove(id: NullableId, params?: Params): Promise<Data> {
    throw MethodNotAllowed;
  }
}

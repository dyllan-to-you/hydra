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
import { Application } from "../../declarations";

interface Data {}

interface ServiceOptions {}

export class FftIndicator implements ServiceMethods<Data> {
  app: Application;
  options: ServiceOptions;
  db: duckdb.Database;

  constructor(options: ServiceOptions = {}, app: Application) {
    this.options = options;
    this.app = app;

    const outputSet = "enviro-chunky-90-365";
    const outputPath = path.resolve(__dirname, "../../../../output", outputSet);
    const outputs = path.join(outputPath, "**", "*.parq");

    /*
      path.join(outputPath, "accumulated_parents.parq")
      path.join(outputPath, "year=2018", "month=1", "day=1", "1.parq")
      path.join(outputPath, "**", "*.parq")
    */
    console.log(outputs);
    this.db = new duckdb.Database(":memory:"); // or a file name for a persistent DB
    this.db.all(
      `SELECT * FROM parquet_scan(['${outputs}']);`,
      (err: any, res: any) => {
        if (err) {
          console.error("ERROR", err);
        } else {
          console.log(res);
        }
      }
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async find(params?: Params): Promise<Data[] | Paginated<Data>> {
    return [];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async get(id: Id, params?: Params): Promise<Data> {
    return {
      id,
      text: `A new message with ID: ${id}!`,
    };
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

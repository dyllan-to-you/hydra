// type callback = (err: Error, res: any) => any;
declare module "duckdb" {
  const duckdb: any;
  export default duckdb;
}
//  {
//   class Database {
//     constructor(path: string);
//     each: (sql: string, cb: callback) => this;
//     all: (sql: string, cb: callback) => this;
//     exec: (sql: string, cb: callback) => this;
//     register: () => this;
//     unregister: () => this;
//     get: () => never;
//   }
// }

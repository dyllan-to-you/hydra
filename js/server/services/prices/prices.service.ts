// Initializes the `prices` service on path `/prices`
import { ServiceAddons } from "@feathersjs/feathers";
import { Application } from "../../declarations";
import { Prices } from "./prices.class";
import createModel from "./prices.model";
import hooks from "./prices.hooks";

// Add this service to the service type index
declare module "../../declarations" {
  interface ServiceTypes {
    prices: Prices & ServiceAddons<any>;
  }
}

export default function (app: Application): void {
  const options = {
    Model: createModel(app),
    paginate: app.get("paginate"),
  };

  // Initialize our service with any options it requires
  app.use("/prices", new Prices(options, app));

  // Get our initialized service so that we can register hooks
  const service = app.service("prices");

  service.hooks(hooks);
}

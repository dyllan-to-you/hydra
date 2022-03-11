// Initializes the `fft-indicator` service on path `/fft-indicator`
import { ServiceAddons } from "@feathersjs/feathers";
import { Application } from "../../declarations";
import { FftIndicator } from "./fft-indicator.class";
import hooks from "./fft-indicator.hooks";

// Add this service to the service type index
declare module "../../declarations" {
  interface ServiceTypes {
    "fft-indicator": FftIndicator & ServiceAddons<any>;
  }
}

export default function (app: Application): void {
  const options = {
    paginate: app.get("paginate"),
  };

  // Initialize our service with any options it requires
  app.use("/fft-indicator", new FftIndicator(options, app));

  // Get our initialized service so that we can register hooks
  const service = app.service("fft-indicator");

  service.hooks(hooks);
}

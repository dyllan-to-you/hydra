import { Application } from "../declarations";
import fftIndicator from "./fft-indicator/fft-indicator.service";
import prices from './prices/prices.service';
// Don't remove this comment. It's needed to format import lines nicely.

export default function (app: Application): void {
  app.configure(fftIndicator);
  app.configure(prices);
}

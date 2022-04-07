import app from "../../server/app";
import { Data as FftIndicatorData } from "../../server/services/fft-indicator/fft-indicator.class";

function getMax(arr: number[]) {
  let len = arr.length;
  let max = -Infinity;

  while (len--) {
    max = arr[len] > max ? arr[len] : max;
  }
  return max;
}

function getMin(arr: number[]) {
  let len = arr.length;
  let min = Infinity;

  while (len--) {
    min = arr[len] < min ? arr[len] : min;
  }
  return min;
}

describe("'fft-indicator' service", () => {
  it("registered the service", () => {
    const service = app.service("fft-indicator");
    expect(service).toBeTruthy();
  });

  it("returns a month of data", async () => {
    const service = app.service("fft-indicator");
    const query = {
      first_extrapolated_date: {
        $gt: "2018-01-01",
        $lt: "2018-02-01",
      },
    };
    const res = <FftIndicatorData[]>await service.find({
      query,
    });

    Object.entries(query).forEach(([prop, condition]) => {
      console.log(prop, condition, typeof condition);
      switch (typeof condition) {
        case "object":
          Object.entries(condition).forEach(([op, val]) => {
            switch (op) {
              case "$gt":
                expect(
                  getMax(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeGreaterThan(new Date(val).getTime());
                break;
              case "$lt":
                expect(
                  getMin(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeLessThan(new Date(val).getTime());
                break;
            }
          });
          break;
        default:
          res.forEach((e: any) => expect(e[prop]).toEqual(condition));
      }
    });
  });

  it("returns a week of data", async () => {
    const service = app.service("fft-indicator");
    const query = {
      first_extrapolated_date: {
        $gt: "2018-01-01",
        $lt: "2018-01-08",
      },
    };

    const res = <FftIndicatorData[]>await service.find({
      query,
    });

    Object.entries(query).forEach(([prop, condition]) => {
      switch (typeof condition) {
        case "object":
          Object.entries(condition).forEach(([op, val]) => {
            switch (op) {
              case "$gt":
                expect(
                  getMax(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeGreaterThan(new Date(val).getTime());
                break;
              case "$lt":
                expect(
                  getMin(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeLessThan(new Date(val).getTime());
                break;
            }
          });
          break;
        default:
          res.forEach((e: any) => expect(e[prop]).toEqual(condition));
      }
    });
  });

  it("returns a day of data with rootNumber2 and 1100 < wavelength < 1400", async () => {
    const service = app.service("fft-indicator");
    const query = {
      first_extrapolated_date: {
        $gt: "2018-01-01",
        $lt: "2018-01-02",
      },
      rootNumber: 2,
      ifft_extrapolated_wavelength: {
        $gt: 1100,
        $lt: 1400,
      },
    };
    const res = <FftIndicatorData[]>await service.find({
      query,
    });

    Object.entries(query).forEach(([prop, condition]) => {
      switch (typeof condition) {
        case "object":
          Object.entries(condition).forEach(([op, val]) => {
            switch (op) {
              case "$gt":
                expect(
                  getMax(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeGreaterThan(new Date(val).getTime());
                break;
              case "$lt":
                expect(
                  getMin(res.map((e: any) => new Date(e[prop]).getTime()))
                ).toBeLessThan(new Date(val).getTime());
                break;
            }
          });
          break;
        default:
          res.forEach((e: any) => expect(e[prop]).toEqual(condition));
      }
    });
  });
});

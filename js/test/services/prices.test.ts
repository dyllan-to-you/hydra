import app from "../../server/app";

describe("'prices' service", () => {
  it("registered the service", () => {
    const service = app.service("prices");
    expect(service).toBeTruthy();
  });
  it("retrieves BTCUSD data", async () => {
    const service = app.service("prices");
    expect(await service.get("BTCUSD")).toBeTruthy();
  });
});

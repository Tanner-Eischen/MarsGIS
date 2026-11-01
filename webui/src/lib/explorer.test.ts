import { afterEach, describe, expect, it, vi } from "vitest";
import { planExplorerRoute, pollExplorerRoute } from "./explorer";

afterEach(() => vi.unstubAllGlobals());
const reply = (body: unknown) => {
  const fetcher = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
};

describe("route transport identity", () => {
  it.each([
    { state: "running", request_id: "selection-1" },
    { id: "job-1", state: "queued", request_id: "selection-1" },
  ])("rejects malformed job envelopes", async (body) => {
    reply(body);
    await expect(
      planExplorerRoute({ lat: 0, lon: 0 }, { lat: 1, lon: 1 }, 25, {
        requestId: "selection-1",
      }),
    ).rejects.toThrow(/different or invalid/);
  });
  it("forwards POST cancellation and the client request identity", async () => {
    const fetcher = reply({
      id: "job-1",
      request_id: "selection-1",
      state: "running",
    });
    const signal = new AbortController().signal;
    const result = await planExplorerRoute(
      { lat: 0, lon: 0 },
      { lat: 1, lon: 1 },
      25,
      {
        signal,
        requestId: "selection-1",
      },
    );
    expect(result.request_id).toBe("selection-1");
    expect(fetcher.mock.calls[0][1].signal).toBe(signal);
    expect(JSON.parse(fetcher.mock.calls[0][1].body).request_id).toBe(
      "selection-1",
    );
  });

  it("rejects a response for a different server job", async () => {
    reply({ id: "job-other", request_id: "selection-1", state: "running" });
    await expect(
      pollExplorerRoute("job-1", undefined, "selection-1"),
    ).rejects.toThrow(/different or invalid/);
  });

  it("rejects a response for a different client request", async () => {
    reply({ id: "job-1", request_id: "selection-other", state: "running" });
    await expect(
      pollExplorerRoute("job-1", undefined, "selection-1"),
    ).rejects.toThrow(/different or invalid/);
  });

  it.each([
    { id: "other", request_id: "selection-1" },
    { id: "job-1", request_id: "other" },
  ])(
    "rejects a ready payload with mismatched nested identity",
    async (result) => {
      reply({ id: "job-1", request_id: "selection-1", state: "ready", result });
      await expect(
        pollExplorerRoute("job-1", undefined, "selection-1"),
      ).rejects.toThrow(/different or invalid/);
    },
  );

  it("rejects invalid identifiers before fetching", async () => {
    const fetcher = reply({});
    await expect(pollExplorerRoute("../outside")).rejects.toThrow(
      /Invalid route job/,
    );
    expect(fetcher).not.toHaveBeenCalled();
  });
});

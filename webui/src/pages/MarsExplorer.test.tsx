import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import MarsExplorer, { pointFromInputs } from "./MarsExplorer";
import {
  loadExplorer,
  planExplorerRoute,
  pollExplorerRoute,
} from "../lib/explorer";
import type { ExplorerMetadata, ExplorerRoute } from "../lib/explorer";

vi.mock("../lib/explorer", async (original) => ({
  ...(await original<typeof import("../lib/explorer")>()),
  loadExplorer: vi.fn(),
  planExplorerRoute: vi.fn(),
  pollExplorerRoute: vi.fn(),
}));
vi.mock("../components/MarsExplorerMap", () => ({
  default: (props: any) => (
    <div data-testid="map">
      <button
        onClick={() =>
          props.onPick(
            props.selection === "start"
              ? { lat: 0, lon: 0 }
              : { lat: 1, lon: 1 },
            props.selection,
          )
        }
      >
        Place selected point
      </button>
      <span data-testid="map-selection">{props.selection}</span>
      <span data-testid="map-command">{props.command?.kind}</span>
      <span data-testid="map-result">{props.result?.id}</span>
    </div>
  ),
}));
vi.mock("../components/TerrainScene3D", () => ({
  default: (props: any) => (
    <div data-testid="3d-scene">
      {props.result.id} · {props.exaggeration}×
    </div>
  ),
}));

const metadata: ExplorerMetadata = {
  global: {
    id: "mola16",
    label: "MOLA global",
    url: "https://example.org/mola",
    sha256: "a".repeat(64),
    radius_m: 3396000,
    native_resolution_m: 3704,
  },
  detail: {
    id: "ctx",
    label: "CTX Jezero",
    url: "https://example.org/ctx",
    sha256: "b".repeat(64),
    radius_m: 3396190,
    native_resolution_m: 20,
    bounds: [77.16, 18.21, 77.7, 18.72],
  },
  tiles: {
    mola: "/api/v1/explorer/tiles/mola/{z}/{x}/{y}.png",
    ctx: "/api/v1/explorer/tiles/ctx/{z}/{x}/{y}.png",
  },
  initial_view: { lat: 0, lon: 0, zoom: 1 },
  detail_view: { lat: 18.4, lon: 77.4, zoom: 9 },
  max_zoom: 12,
};
const result: ExplorerRoute = {
  id: "route-1",
  source: { ...metadata.global, analysis_resolution_m: 3704 },
  requested: { start: { lat: 0, lon: 0 }, end: { lat: 1, lon: 1 } },
  snapped: { start: { lat: 0, lon: 0 }, end: { lat: 1, lon: 1 } },
  path: [
    { lat: 0, lon: 0, elevation_m: 0, x_m: 0, y_m: 0 },
    { lat: 1, lon: 1, elevation_m: 100, x_m: 1000, y_m: 1000 },
  ],
  geojson: { type: "FeatureCollection", features: [] },
  distance_m: 85000,
  straight_line_m: 83800,
  terrain_cost_m_equivalent: 100000,
  mean_slope_deg: 2,
  max_slope_deg: 5,
  grid: { shape: [2, 2] },
  surface: {
    x_m: [
      [0, 1000],
      [0, 1000],
    ],
    y_m: [
      [0, 0],
      [1000, 1000],
    ],
    elevation_m: [
      [0, 20],
      [50, 100],
    ],
    display_step: 1,
  },
  elevation_profile: [
    { distance_m: 0, elevation_m: 0 },
    { distance_m: 85000, elevation_m: 100 },
  ],
};
const mount = () =>
  render(
    <MemoryRouter>
      <MarsExplorer />
    </MemoryRouter>,
  );
beforeEach(() => {
  vi.mocked(loadExplorer).mockResolvedValue(metadata);
  vi.mocked(planExplorerRoute).mockResolvedValue({
    id: "route-1",
    state: "running",
  });
  vi.mocked(pollExplorerRoute).mockResolvedValue({
    id: "route-1",
    state: "ready",
    result,
  });
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.useRealTimers();
});

describe("map-first two-point workflow", () => {
  it("lets map clicks choose A then B and preserves zero coordinates", async () => {
    mount();
    await screen.findByTestId("map");
    expect(
      (screen.getByRole("button", { name: "Plan route" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Place selected point" }),
    );
    expect(screen.getByTestId("map-selection").textContent).toBe("end");
    expect(
      (screen.getByLabelText("Start latitude") as HTMLInputElement).value,
    ).toBe("0.000000");
    fireEvent.click(
      screen.getByRole("button", { name: "Place selected point" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Plan route" }));
    await waitFor(() =>
      expect(planExplorerRoute).toHaveBeenCalledWith(
        { lat: 0, lon: 0 },
        { lat: 1, lon: 1 },
        25,
      ),
    );
    expect(
      (
        screen.getByRole("button", {
          name: "Calculating route…",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it("shows the computed path in map and 3D, adjusts exaggeration and can return to true scale", async () => {
    mount();
    await screen.findByTestId("map");
    fireEvent.click(screen.getByRole("button", { name: "Use Jezero example" }));
    fireEvent.click(screen.getByRole("button", { name: "Plan route" }));
    await screen.findByTestId("3d-scene", {}, { timeout: 4000 });
    expect(screen.getByTestId("map-result").textContent).toBe("route-1");
    expect(screen.getByTestId("3d-scene").textContent).toContain("5×");
    fireEvent.change(
      screen.getByRole("slider", { name: "3D vertical exaggeration" }),
      { target: { value: "10" } },
    );
    expect(
      (
        screen.getByRole("slider", {
          name: "Vertical exaggeration",
        }) as HTMLInputElement
      ).value,
    ).toBe("10");
    expect(screen.getByTestId("3d-scene").textContent).toContain("10×");
    fireEvent.change(
      screen.getByRole("slider", { name: "Vertical exaggeration" }),
      {
        target: { value: "20" },
      },
    );
    expect(screen.getByTestId("3d-scene").textContent).toContain("20×");
    fireEvent.click(screen.getByRole("button", { name: "True scale · 1×" }));
    expect(screen.getByTestId("3d-scene").textContent).toContain("1×");
    expect(
      screen
        .getByRole("link", { name: "Download this route" })
        .getAttribute("href"),
    ).toBe("/api/v1/explorer/routes/route-1/geojson");
    fireEvent.click(screen.getByRole("button", { name: "3D path" }));
    expect(document.querySelector(".mars-view-3d")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Map" }));
    expect(document.querySelector(".mars-view-map")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Map + 3D" }));
    expect(document.querySelector(".mars-view-split")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Fit route" }));
    expect(screen.getByTestId("map-command").textContent).toBe("route");
  });

  it("invalidates a stale completed route when points or constraints change", async () => {
    mount();
    await screen.findByTestId("map");
    fireEvent.click(screen.getByRole("button", { name: "Use Jezero example" }));
    fireEvent.click(screen.getByRole("button", { name: "Plan route" }));
    await screen.findByText("Route results", {}, { timeout: 4000 });
    fireEvent.change(screen.getByLabelText("Destination longitude"), {
      target: { value: "80" },
    });
    expect(screen.queryByText("Route results")).toBeNull();
    expect(screen.queryByTestId("3d-scene")).toBeNull();
    expect(screen.getByTestId("map-result").textContent).toBe("");
  });

  it("does not publish an in-flight result for an old point selection", async () => {
    mount();
    await screen.findByTestId("map");
    fireEvent.click(screen.getByRole("button", { name: "Use Jezero example" }));
    fireEvent.click(screen.getByRole("button", { name: "Plan route" }));
    await waitFor(() => expect(planExplorerRoute).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText("Destination latitude"), {
      target: { value: "19" },
    });
    await screen.findByText(
      /Your points changed during calculation/,
      {},
      { timeout: 4000 },
    );
    expect(screen.queryByText("Route results")).toBeNull();
    expect(screen.queryByTestId("3d-scene")).toBeNull();
  });

  it("allows global/Jezeroview commands without starting analysis and clears both points", async () => {
    mount();
    await screen.findByTestId("map");
    fireEvent.click(screen.getByRole("button", { name: "Whole planet" }));
    expect(screen.getByTestId("map-command").textContent).toBe("global");
    fireEvent.click(screen.getByRole("button", { name: "Jezero example" }));
    expect(screen.getByTestId("map-command").textContent).toBe("jezero");
    expect(
      (screen.getByLabelText("Start latitude") as HTMLInputElement).value,
    ).toBe("18.380010");
    expect(planExplorerRoute).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Clear points" }));
    expect(
      (screen.getByLabelText("Start latitude") as HTMLInputElement).value,
    ).toBe("");
    expect(
      (screen.getByLabelText("Destination longitude") as HTMLInputElement)
        .value,
    ).toBe("");
    expect(screen.getByTestId("map-selection").textContent).toBe("start");
  });

  it("keeps a source connection error visible when coordinate fields are edited, then retries", async () => {
    vi.mocked(loadExplorer).mockRejectedValueOnce(
      new Error("Planetary data is unavailable"),
    );
    mount();
    expect((await screen.findByRole("alert")).textContent).toContain(
      "Planetary data is unavailable",
    );
    fireEvent.change(screen.getByLabelText("Start latitude"), {
      target: { value: "0" },
    });
    expect(
      screen.getByRole("button", { name: "Retry connection" }),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }));
    await screen.findByTestId("map");
  });

  it("reports an unavailable route without substituting a straight line", async () => {
    vi.mocked(pollExplorerRoute).mockResolvedValue({
      id: "route-1",
      state: "failed",
      error: "No traversable route exists in the sampled window",
    });
    mount();
    await screen.findByTestId("map");
    fireEvent.click(screen.getByRole("button", { name: "Use Jezero example" }));
    fireEvent.click(screen.getByRole("button", { name: "Plan route" }));
    expect(
      (await screen.findByRole("alert", {}, { timeout: 4000 })).textContent,
    ).toContain("No traversable route");
    expect(screen.queryByText("Route results")).toBeNull();
    expect(screen.queryByTestId("3d-scene")).toBeNull();
  });
});

describe("geographic form values", () => {
  it.each([
    ["", "0"],
    ["0", ""],
    ["91", "0"],
    ["0", "181"],
    ["NaN", "0"],
    ["Infinity", "0"],
  ])("rejects invalid input %s/%s", (lat, lon) =>
    expect(pointFromInputs(lat, lon)).toBeNull(),
  );
  it("accepts measured origin and map limits", () => {
    expect(pointFromInputs("0", "0")).toEqual({ lat: 0, lon: 0 });
    expect(pointFromInputs("-90", "-180")).toEqual({ lat: -90, lon: -180 });
  });
});

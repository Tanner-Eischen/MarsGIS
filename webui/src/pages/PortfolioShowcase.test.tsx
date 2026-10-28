import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import PortfolioShowcase from "./PortfolioShowcase";
import {
  getPortfolioStatus,
  loadProbe,
  loadScene,
  startPortfolioRun,
  validateScene,
} from "../lib/portfolioData";

vi.mock("../lib/portfolioData", async (original) => ({
  ...(await original<typeof import("../lib/portfolioData")>()),
  getPortfolioStatus: vi.fn(),
  loadScene: vi.fn(),
  loadProbe: vi.fn(),
  startPortfolioRun: vi.fn(),
}));
// These are Node component tests using a virtual DOM, not browser/UI automation.
const scene = validateScene(
  JSON.parse(readFileSync(resolve("public/portfolio/scene.json"), "utf8")),
);
const bytes = readFileSync(resolve("public/portfolio/probe.bin"));
const buffer = bytes.buffer.slice(
  bytes.byteOffset,
  bytes.byteOffset + bytes.byteLength,
) as ArrayBuffer;

beforeEach(() => {
  vi.mocked(getPortfolioStatus).mockResolvedValue({
    state: "ready",
    data_available: true,
  });
  vi.mocked(loadScene).mockResolvedValue(scene);
  vi.mocked(loadProbe).mockResolvedValue(buffer);
  vi.mocked(startPortfolioRun).mockResolvedValue({
    state: "running",
    data_available: true,
  });
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("scientific website controls", () => {
  it("loads the actual scene, selects all layers and keeps scientific scope visible", async () => {
    render(<PortfolioShowcase />);
    await screen.findByText("One region. Five views.");
    expect(screen.getByText(/verified reference run/)).toBeTruthy();
    for (const layer of scene.layers) {
      const button = screen.getByRole("button", { name: layer.label });
      fireEvent.click(button);
      expect(button.getAttribute("aria-pressed")).toBe("true");
      expect(document.querySelector("svg image")?.getAttribute("href")).toBe(
        `/portfolio/${layer.image}`,
      );
    }
    expect(screen.getByText(/not safety probabilities/)).toBeTruthy();
    expect(
      screen
        .getByRole("link", { name: /USGS source description/ })
        .getAttribute("href"),
    ).toBe(scene.source.citation);
  });

  it("selects candidates with keyboard markers and toggles complete route geometry", async () => {
    render(<PortfolioShowcase />);
    const marker = await screen.findByRole("button", {
      name: "Select candidate 2 on map",
    });
    fireEvent.keyDown(marker, { key: "Enter" });
    expect(marker.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getAllByTestId(/^route-/)).toHaveLength(1);
    expect(screen.getByTestId("route-2").getAttribute("points")).toBe(
      scene.routes[1].points.map((point) => point.join(",")).join(" "),
    );
    expect((screen.getByLabelText("Row") as HTMLInputElement).value).toBe(
      String(scene.candidates[1].row),
    );
    const toggle = screen.getByRole("checkbox", { name: "Show all routes" });
    fireEvent.click(toggle);
    expect(screen.getAllByTestId(/^route-/)).toHaveLength(3);
    fireEvent.click(toggle);
    expect(screen.getAllByTestId(/^route-/)).toHaveLength(1);
    fireEvent.keyDown(
      screen.getByRole("button", { name: "Select candidate 1 on map" }),
      { key: " " },
    );
    expect(screen.getByTestId("route-1")).toBeTruthy();
  });

  it("zooms within limits and fits the exact source grid again", async () => {
    render(<PortfolioShowcase />);
    await screen.findByText("One region. Five views.");
    const svg = screen.getByRole("group", {
      name: "Jezero terrain map, local grid, north up",
    });
    const original = `0 0 ${scene.grid.shape[1]} ${scene.grid.shape[0]}`;
    expect(svg.getAttribute("viewBox")).toBe(original);
    expect(
      (screen.getByRole("button", { name: "Zoom out" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(svg.getAttribute("viewBox")).not.toBe(original);
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(
      (screen.getByRole("button", { name: "Zoom in" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    fireEvent.click(screen.getByRole("button", { name: "Fit region" }));
    expect(svg.getAttribute("viewBox")).toBe(original);
  });

  it("probes actual full-resolution raster values and marks invalid cells NoData", async () => {
    render(<PortfolioShowcase />);
    await screen.findByText("One region. Five views.");
    await waitFor(() =>
      expect(document.querySelectorAll(".probe-values dd")).toHaveLength(5),
    );
    fireEvent.change(screen.getByLabelText("Row"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Column"), {
      target: { value: "0" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Inspect cell" }));
    const expected = new DataView(buffer).getFloat32(0, true);
    const text = document.querySelector(".probe-values dd")?.textContent;
    if (Number.isFinite(expected))
      expect(text).toContain(
        expected.toLocaleString("en-US", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }),
      );
    else expect(text).toBe("NoData");
    fireEvent.change(screen.getByLabelText("Row"), { target: { value: "-1" } });
    fireEvent.submit(
      screen.getByRole("button", { name: "Inspect cell" }).closest("form")!,
    );
    expect(screen.getByRole("alert").textContent).toContain("inside this grid");
  });

  it("provides every original scientific download from the same run", async () => {
    const live = { ...scene, asset_base: "/api/v1/portfolio/files/test-run/" };
    vi.mocked(loadScene).mockResolvedValue(live);
    render(<PortfolioShowcase />);
    await screen.findByText("Download the analysis.");
    for (const download of scene.downloads) {
      const link = screen
        .getAllByRole("link")
        .find((item) => item.getAttribute("download") === download.path)!;
      expect(link.getAttribute("href")).toBe(
        `${live.asset_base}${download.path}`,
      );
      expect(link.getAttribute("download")).toBe(download.path);
    }
  });

  it("executes the server workflow and loads the completed run", async () => {
    const live = {
      ...scene,
      execution_mode: "server_computed_analysis" as const,
    };
    vi.mocked(loadScene).mockResolvedValueOnce(scene).mockResolvedValue(live);
    render(<PortfolioShowcase />);
    await screen.findByText("One region. Five views.");
    fireEvent.click(
      screen.getByRole("button", { name: "Run Jezero analysis" }),
    );
    await waitFor(() => expect(startPortfolioRun).toHaveBeenCalledTimes(1));
    expect(
      (
        screen.getByRole("button", {
          name: /Analysis running/,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    await screen.findByText(
      "Showing results computed by this analysis service.",
      {},
      { timeout: 4000 },
    );
    expect(loadScene).toHaveBeenCalledTimes(2);
    expect(
      screen.getByRole("button", { name: "Run Jezero analysis" }),
    ).toBeTruthy();
  });

  it("handles an unavailable API and retries without presenting fabricated results", async () => {
    vi.mocked(getPortfolioStatus).mockRejectedValueOnce(
      new Error("Analysis API returned HTTP 503"),
    );
    render(<PortfolioShowcase />);
    expect((await screen.findByRole("alert")).textContent).toContain("503");
    expect(loadScene).not.toHaveBeenCalled();
    expect(screen.queryByText("One region. Five views.")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }));
    await screen.findByText("One region. Five views.");
  });

  it("retains the scene and offers measurement retry when integrity verification fails", async () => {
    vi.mocked(loadProbe).mockRejectedValueOnce(
      new Error("Measurement checksum mismatch"),
    );
    render(<PortfolioShowcase />);
    expect((await screen.findByRole("alert")).textContent).toContain(
      "checksum mismatch",
    );
    expect(screen.getByText("Download the analysis.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry measurements" }));
    await waitFor(() =>
      expect(document.querySelectorAll(".probe-values dd")).toHaveLength(5),
    );
  });
});

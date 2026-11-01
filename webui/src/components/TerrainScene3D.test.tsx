import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import Plotly from "plotly.js/dist/plotly-gl3d.min.js";
import TerrainScene3D, { browserHasWebGL } from "./TerrainScene3D";
import type { ExplorerRoute } from "../lib/explorer";

const state = vi.hoisted(() => ({
  props: {} as any,
  resize: null as (() => void) | null,
  plotResize: vi.fn(),
  loseContext: vi.fn(),
}));
vi.mock("plotly.js/dist/plotly-gl3d.min.js", () => ({
  default: {
    relayout: vi.fn().mockResolvedValue(undefined),
    Plots: { resize: state.plotResize },
  },
}));
vi.mock("react-plotly.js/factory", async () => {
  const React = await import("react");
  return {
    default: () =>
      function FakePlot(props: any) {
        state.props = props;
        const element = React.useRef<HTMLDivElement | null>(null);
        React.useEffect(() => {
          props.onInitialized({}, element.current);
          return () => props.onPurge?.();
        }, [props]);
        return <div ref={element} data-testid="plot" />;
      },
  };
});

// Display-only fixture, retaining original measured-elevation/NoData coverage.
const result = {
  id: "observed-run",
  surface: {
    x_m: [
      [0, 1000],
      [0, 1000],
    ],
    y_m: [
      [0, 0],
      [2000, 2000],
    ],
    elevation_m: [
      [-2000, null],
      [-1500, -1000],
    ],
    display_step: 3,
  },
  path: [
    { lat: 0, lon: 0, x_m: 0, y_m: 0, elevation_m: -2000 },
    { lat: 0.01, lon: 0.01, x_m: 1000, y_m: 2000, elevation_m: -1000 },
  ],
} as ExplorerRoute;

beforeEach(() => {
  state.props = {};
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: () => void) {
        state.resize = callback;
      }
      observe() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    getExtension: () => ({ loseContext: state.loseContext }),
  } as any);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("renderer controls, measured geometry and device fallback", () => {
  it("keeps measured route/surface elevations and NoData unchanged when exaggerating", async () => {
    const { rerender } = render(
      <TerrainScene3D result={result} exaggeration={1} />,
    );
    await screen.findByTestId("plot");
    const data = state.props.data;
    expect(data[0].z).toEqual([
      [-2, null],
      [-1.5, -1],
    ]);
    expect(data[1].z).toEqual([-2, -1]);
    expect(state.props.layout.scene.aspectratio.z).toBe(0.5);
    rerender(<TerrainScene3D result={result} exaggeration={20} />);
    expect(state.props.data).toEqual(data);
    expect(state.props.layout.scene.aspectratio.z).toBe(10);
    expect(screen.getByText("Vertical exaggeration ×20")).toBeTruthy();
    expect(state.props.layout.uirevision).toBe("observed-run");
  });

  it("exposes orbit/zoom controls and resets the actual renderer explicitly", async () => {
    render(<TerrainScene3D result={result} exaggeration={5} />);
    await screen.findByTestId("plot");
    expect(state.props.config).toMatchObject({
      displayModeBar: true,
      scrollZoom: true,
      responsive: true,
    });
    expect(state.props.useResizeHandler).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Reset 3D camera" }));
    await waitFor(() =>
      expect(Plotly.relayout).toHaveBeenCalledWith(expect.any(HTMLElement), {
        "scene.camera": {
          eye: { x: 1.45, y: -1.45, z: 1.1 },
          up: { x: 0, y: 0, z: 1 },
        },
      }),
    );
    expect(
      screen.getByText(/Surface displayed every 3 analysis cells/),
    ).toBeTruthy();
  });

  it("resizes after pane changes and announces a runtime renderer failure", async () => {
    render(<TerrainScene3D result={result} exaggeration={5} />);
    const plot = await screen.findByTestId("plot");
    act(() => state.resize?.());
    expect(state.plotResize).toHaveBeenCalledWith(plot);
    act(() => state.props.onError(new Error("WebGL unavailable")));
    expect(screen.getByRole("alert").textContent).toContain(
      "Map view or download",
    );
  });

  it("announces the unavailable-device fallback without a broken Plotly pane", async () => {
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null);
    render(<TerrainScene3D result={result} exaggeration={5} />);
    expect((await screen.findByRole("alert")).textContent).toContain(
      "Map view or download",
    );
    expect(screen.queryByTestId("plot")).toBeNull();
    expect(
      (
        screen.getByRole("button", {
          name: "Reset 3D camera",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it("treats context-creation exceptions as unavailable graphics", () => {
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockImplementation(() => {
      throw new Error("Device unavailable");
    });
    expect(browserHasWebGL()).toBe(false);
  });

  it("releases the capability-probe context before creating the real renderer", () => {
    expect(browserHasWebGL()).toBe(true);
    expect(state.loseContext).toHaveBeenCalledTimes(1);
  });
});

import { useEffect, useMemo, useRef, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js/dist/plotly-gl3d.min.js";
import type { ExplorerRoute } from "../lib/explorer";
import { buildTerrainScene, defaultTerrainCamera } from "../lib/terrainScene";

const Plot = createPlotlyComponent(Plotly);

export function browserHasWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!context) return false;
    context.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

export default function TerrainScene3D({
  result,
  exaggeration,
}: {
  result: ExplorerRoute;
  exaggeration: number;
}) {
  const [webglSupported, setWebglSupported] = useState<boolean | null>(null);
  const [plotError, setPlotError] = useState("");
  const graphRef = useRef<HTMLElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    setWebglSupported(browserHasWebGL());
  }, []);
  useEffect(() => {
    const resize = () => {
      if (!graphRef.current) return;
      try {
        Plotly.Plots.resize(graphRef.current);
      } catch {
        /* The WebGL scene may be unmounting. */
      }
    };
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
    if (containerRef.current) observer?.observe(containerRef.current);
    return () => observer?.disconnect();
  }, []);
  const resetCamera = async () => {
    if (!graphRef.current) return;
    try {
      await Plotly.relayout(graphRef.current, {
        "scene.camera": defaultTerrainCamera(),
      });
    } catch (error) {
      setPlotError(
        error instanceof Error
          ? error.message
          : "The 3D view could not update.",
      );
    }
  };
  const figure = useMemo(
    () => buildTerrainScene(result, exaggeration),
    [result, exaggeration],
  );
  const layout = {
    ...figure.layout,
    uirevision: result.id,
    scene: {
      ...figure.layout.scene,
      camera: defaultTerrainCamera(),
      uirevision: result.id,
    },
  };
  return (
    <div
      className="terrain-scene-3d"
      style={{
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
      }}
      aria-label="Interactive three-dimensional Mars route terrain"
    >
      <div className="terrain-scene-controls">
        <span>
          Vertical exaggeration ×{Math.max(1, Math.min(100, exaggeration))}
        </span>
        <button
          type="button"
          disabled={!webglSupported || !!plotError}
          onClick={() => void resetCamera()}
        >
          Reset 3D camera
        </button>
      </div>
      <p className="small-copy">
        Drag to orbit. Use the plot toolbar to pan, zoom or reset. Elevation
        measurements remain unchanged by exaggeration.
      </p>
      {(webglSupported === false || plotError) && (
        <p role="alert">
          The 3D view is unavailable in this browser. Use the Map view or
          download the calculated route; its measurements remain available.
          {plotError && ` ${plotError}`}
        </p>
      )}
      <div
        ref={containerRef}
        className="terrain-scene-plot"
        style={{ flex: 1, minHeight: 0 }}
      >
        {webglSupported === null && (
          <p role="status">Checking 3D graphics support…</p>
        )}
        {webglSupported === true && !plotError && (
          <Plot
            data={figure.data}
            layout={layout}
            config={{
              displayModeBar: true,
              displaylogo: false,
              responsive: true,
              scrollZoom: true,
            }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            onInitialized={(_figure, graph) => {
              graphRef.current = graph;
            }}
            onPurge={() => {
              graphRef.current = null;
            }}
            onError={(error) => setPlotError(error.message)}
            onWebGlContextLost={() =>
              setPlotError("The graphics context was lost.")
            }
          />
        )}
      </div>
      <p className="small-copy">
        Surface displayed every {result.surface.display_step} analysis cells.
        Gaps mark NoData; route vertices use their measured terrain elevations.
      </p>
    </div>
  );
}

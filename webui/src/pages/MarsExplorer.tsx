import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Link } from "react-router-dom";
import {
  ArrowDownToLine,
  Globe2,
  Layers3,
  LoaderCircle,
  MapPin,
  Navigation,
  RotateCcw,
} from "lucide-react";
import MarsExplorerMap from "../components/MarsExplorerMap";
import {
  loadExplorer,
  normalizeLongitude,
  planExplorerRoute,
  pollExplorerRoute,
  validPoint,
} from "../lib/explorer";
import type {
  ExplorerJob,
  ExplorerMetadata,
  ExplorerRoute,
  MapCommand,
  MarsPoint,
  PickMode,
} from "../lib/explorer";
import "../explorer.css";

const TerrainScene3D = lazy(() => import("../components/TerrainScene3D"));
type Coordinates = {
  startLat: string;
  startLon: string;
  endLat: string;
  endLon: string;
};
const empty: Coordinates = {
  startLat: "",
  startLon: "",
  endLat: "",
  endLon: "",
};
type ActiveRequest = {
  id: string;
  signature: string;
  controller: AbortController;
  pending: boolean;
};
const number = (value: number, digits = 2) =>
  value.toLocaleString("en-US", { maximumFractionDigits: digits });
const distance = (value: number) =>
  value < 1000 ? `${number(value, 0)} m` : `${number(value / 1000)} km`;
const message = (failure: unknown) =>
  failure instanceof Error
    ? failure.message
    : "The Mars analysis service could not complete the request.";

export function pointFromInputs(
  latitude: string,
  longitude: string,
): MarsPoint | null {
  if (!latitude.trim() || !longitude.trim()) return null;
  const point = { lat: Number(latitude), lon: Number(longitude) };
  return validPoint(point) ? point : null;
}

export default function MarsExplorer() {
  const [metadata, setMetadata] = useState<ExplorerMetadata | null>(null);
  const [coordinates, setCoordinates] = useState<Coordinates>(empty);
  const [selection, setSelection] = useState<PickMode>("start");
  const [maxSlope, setMaxSlope] = useState(25);
  const [view, setView] = useState<"map" | "split" | "3d">("map");
  const [exaggeration, setExaggeration] = useState(5);
  const [command, setCommand] = useState<MapCommand | null>(null);
  const [job, setJob] = useState<ExplorerJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<ExplorerRoute | null>(null);
  const [error, setError] = useState("");
  const [sourceError, setSourceError] = useState("");
  const [notice, setNotice] = useState("");
  const [retry, setRetry] = useState(0);
  const signature = JSON.stringify([coordinates, maxSlope]);
  const currentSignature = useRef(signature);
  currentSignature.current = signature;
  const activeRequest = useRef<ActiveRequest | null>(null);
  const start = pointFromInputs(coordinates.startLat, coordinates.startLon);
  const end = pointFromInputs(coordinates.endLat, coordinates.endLon);
  const busy = submitting || job?.state === "running";
  const moveView = useCallback(
    (kind: MapCommand["kind"]) =>
      setCommand((previous) => ({ id: (previous?.id ?? 0) + 1, kind })),
    [],
  );
  const isCurrent = useCallback(
    (active: ActiveRequest) =>
      activeRequest.current === active &&
      !active.controller.signal.aborted &&
      currentSignature.current === active.signature,
    [],
  );
  const publish = useCallback(
    (next: ExplorerJob, active: ActiveRequest, expectedId?: string) => {
      if (!isCurrent(active)) return;
      if (
        next.request_id !== active.id ||
        (expectedId && next.id !== expectedId) ||
        (next.state === "ready" &&
          (!next.result ||
            next.result.id !== next.id ||
            next.result.request_id !== active.id))
      ) {
        active.pending = false;
        setError("The Mars API returned a different route request.");
        setJob(null);
        return;
      }
      setJob(next);
      if (next.state === "failed") {
        active.pending = false;
        setError(
          next.error || "No route could be calculated for these points.",
        );
      }
      if (next.state === "ready" && next.result) {
        active.pending = false;
        setResult(next.result);
        setView("split");
        moveView("route");
      }
    },
    [isCurrent, moveView],
  );

  useEffect(() => {
    const controller = new AbortController();
    setSourceError("");
    void loadExplorer(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setMetadata(value);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setSourceError(message(failure));
      });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    const previous = activeRequest.current;
    previous?.controller.abort();
    activeRequest.current = null;
    setJob(null);
    setSubmitting(false);
    setResult(null);
    setError("");
    setNotice(
      previous?.pending
        ? "Your points changed during calculation. Plan a route for the new selection."
        : "",
    );
    setView("map");
  }, [signature]);

  useEffect(
    () => () => {
      activeRequest.current?.controller.abort();
      activeRequest.current = null;
    },
    [],
  );

  useEffect(() => {
    if (job?.state !== "running") return;
    const active = activeRequest.current;
    if (!active || !isCurrent(active)) return;
    const controller = new AbortController();
    const cancelPoll = () => controller.abort();
    active.controller.signal.addEventListener("abort", cancelPoll, {
      once: true,
    });
    const timer = window.setTimeout(async () => {
      try {
        const next = await pollExplorerRoute(
          job.id,
          controller.signal,
          active.id,
        );
        if (!controller.signal.aborted) publish(next, active, job.id);
      } catch (failure) {
        if (!controller.signal.aborted && isCurrent(active)) {
          active.pending = false;
          setError(message(failure));
          setJob(null);
        }
      }
    }, 1200);
    return () => {
      window.clearTimeout(timer);
      active.controller.signal.removeEventListener("abort", cancelPoll);
      controller.abort();
    };
  }, [job, isCurrent, publish]);

  const pick = (point: MarsPoint, which: PickMode) => {
    setCoordinates((previous) => ({
      ...previous,
      [`${which}Lat`]: point.lat.toFixed(6),
      [`${which}Lon`]: normalizeLongitude(point.lon).toFixed(6),
    }));
    if (which === "start") setSelection("end");
  };

  const plan = async () => {
    if (activeRequest.current?.pending) return;
    if (
      !start ||
      !end ||
      !Number.isFinite(maxSlope) ||
      maxSlope <= 0 ||
      maxSlope > 60
    ) {
      setError(
        "Select two valid Mars points and a maximum slope from 1° to 60°.",
      );
      return;
    }
    setError("");
    setNotice("");
    setResult(null);
    setSubmitting(true);
    const active: ActiveRequest = {
      id: crypto.randomUUID(),
      signature,
      controller: new AbortController(),
      pending: true,
    };
    activeRequest.current?.controller.abort();
    activeRequest.current = active;
    try {
      const next = await planExplorerRoute(start, end, maxSlope, {
        signal: active.controller.signal,
        requestId: active.id,
      });
      publish(next, active);
    } catch (failure) {
      if (isCurrent(active)) {
        active.pending = false;
        setJob(null);
        setError(message(failure));
      }
    } finally {
      if (isCurrent(active)) setSubmitting(false);
    }
  };

  const example = () => {
    setCoordinates({
      startLat: "18.380010",
      startLon: "77.385212",
      endLat: "18.384727",
      endLon: "77.411880",
    });
    setSelection("end");
    setMaxSlope(25);
    moveView("jezero");
  };

  return (
    <div className="mars-explorer">
      <a className="mars-skip" href="#route-controls">
        Skip to route controls
      </a>
      <header className="mars-header">
        <Link className="mars-logo" to="/">
          <span aria-hidden="true" />
          MarsHab <small>/ MarsGIS</small>
        </Link>
        <nav aria-label="Workspace navigation">
          <span aria-current="page">Mars explorer</span>
          <Link to="/comparison">Site comparison</Link>
        </nav>
        <span className="mars-observed">NASA MOLA + CTX elevations</span>
      </header>
      <main className="mars-workspace">
        <section
          className={`mars-viewport mars-view-${view}`}
          aria-label="Mars map and terrain scene"
        >
          <div className="mars-toolbar">
            <div>
              <Globe2 size={18} />
              <strong>Explore Mars</strong>
              <span>Drag to pan · scroll to zoom</span>
            </div>
            <div
              className="mars-view-buttons"
              role="group"
              aria-label="Workspace view"
            >
              {(["map", "split", "3d"] as const).map((mode) => (
                <button
                  key={mode}
                  aria-pressed={view === mode}
                  disabled={mode !== "map" && !result}
                  onClick={() => setView(mode)}
                >
                  {mode === "map"
                    ? "Map"
                    : mode === "split"
                      ? "Map + 3D"
                      : "3D path"}
                </button>
              ))}
            </div>
          </div>
          <div className="mars-map-pane" aria-hidden={view === "3d"}>
            {metadata ? (
              <MarsExplorerMap
                metadata={metadata}
                start={start}
                end={end}
                selection={selection}
                result={result}
                command={command}
                onPick={pick}
              />
            ) : (
              <div className="mars-placeholder" role="status">
                {sourceError ? (
                  "Map source is unavailable."
                ) : (
                  <>
                    <LoaderCircle className="mars-spin" />
                    Loading global Mars terrain…
                  </>
                )}
              </div>
            )}
            <div className="mars-map-actions">
              <button onClick={() => moveView("global")} disabled={!metadata}>
                <Globe2 size={14} />
                Whole planet
              </button>
              <button onClick={example} disabled={!metadata}>
                Jezero example
              </button>
              {result && (
                <button onClick={() => moveView("route")}>Fit route</button>
              )}
            </div>
            <div className="mars-pick-hint">
              <MapPin size={15} />
              Click to set{" "}
              {selection === "start" ? "A · start" : "B · destination"}
            </div>
            <div className="mars-map-credit">
              MOLA global topography · ~3.7 km at equator · outlined Jezero area
              has CTX detail
            </div>
          </div>
          {result && view !== "map" && (
            <section className="mars-scene-pane" aria-label="3D path viewer">
              <div className="mars-scene-heading">
                <Layers3 size={17} />
                <strong>3D terrain & path</strong>
                <label className="mars-inline-exaggeration">
                  Vertical exaggeration
                  <input
                    aria-label="3D vertical exaggeration"
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={exaggeration}
                    onChange={(event) =>
                      setExaggeration(Number(event.target.value))
                    }
                  />
                </label>
                <span>{exaggeration}× vertical exaggeration</span>
              </div>
              <Suspense
                fallback={
                  <div className="mars-placeholder" role="status">
                    Loading 3D viewer…
                  </div>
                }
              >
                <TerrainScene3D result={result} exaggeration={exaggeration} />
              </Suspense>
            </section>
          )}
        </section>
        <aside className="mars-route-panel" id="route-controls">
          <div className="mars-panel-title">
            <Navigation size={20} />
            <div>
              <p>TWO-POINT ROUTING</p>
              <h1>Choose your traverse.</h1>
            </div>
          </div>
          <p className="mars-help">
            Select a start and destination on the map, or enter coordinates
            below. The route and 3D view use the same elevation data.
          </p>
          <div
            className="mars-select-buttons"
            role="group"
            aria-label="Point selection"
          >
            <button
              aria-pressed={selection === "start"}
              onClick={() => setSelection("start")}
            >
              <b>A</b>Choose start
            </button>
            <button
              aria-pressed={selection === "end"}
              onClick={() => setSelection("end")}
            >
              <b>B</b>Choose destination
            </button>
          </div>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void plan();
            }}
          >
            {(["start", "end"] as const).map((which) => (
              <fieldset key={which}>
                <legend>
                  <span className={`mars-point-label mars-point-${which}`}>
                    {which === "start" ? "A" : "B"}
                  </span>
                  {which === "start" ? "Start" : "Destination"}
                </legend>
                <div className="mars-coordinate-pair">
                  {(["Lat", "Lon"] as const).map((axis) => {
                    const key = `${which}${axis}` as keyof Coordinates;
                    return (
                      <label key={axis}>
                        {axis === "Lat" ? "Latitude" : "Longitude"}
                        <input
                          aria-label={`${which === "start" ? "Start" : "Destination"} ${axis === "Lat" ? "latitude" : "longitude"}`}
                          type="number"
                          step="any"
                          min={axis === "Lat" ? -90 : -180}
                          max={axis === "Lat" ? 90 : 180}
                          required
                          value={coordinates[key]}
                          placeholder={
                            axis === "Lat" ? "−90 … 90" : "−180 … 180"
                          }
                          onChange={(event) =>
                            setCoordinates((previous) => ({
                              ...previous,
                              [key]: event.target.value,
                            }))
                          }
                        />
                      </label>
                    );
                  })}
                </div>
              </fieldset>
            ))}
            <label className="mars-slope-control">
              Maximum analyzed slope <span>{maxSlope}°</span>
              <input
                type="range"
                min="1"
                max="60"
                step="1"
                value={maxSlope}
                onChange={(event) => setMaxSlope(Number(event.target.value))}
              />
            </label>
            <button
              className="mars-plan-button"
              type="submit"
              disabled={!metadata || !start || !end || busy}
            >
              {busy ? (
                <LoaderCircle size={16} className="mars-spin" />
              ) : (
                <Navigation size={16} />
              )}
              {busy ? "Calculating route…" : "Plan route"}
            </button>
          </form>
          <div className="mars-secondary-actions">
            <button onClick={example} disabled={!metadata}>
              Use Jezero example
            </button>
            <button
              onClick={() => {
                setCoordinates(empty);
                setSelection("start");
              }}
            >
              <RotateCcw size={13} />
              Clear points
            </button>
          </div>
          {busy && (
            <p role="status" className="mars-status">
              Calculating a terrain-weighted route on a bounded analysis grid.
              The map stays interactive.
            </p>
          )}
          {notice && (
            <p role="status" className="mars-status">
              {notice}
            </p>
          )}
          {(sourceError || error) && (
            <div className="mars-error" role="alert">
              <strong>Unable to complete this request</strong>
              <p>{sourceError || error}</p>
              {!metadata && (
                <button onClick={() => setRetry((value) => value + 1)}>
                  Retry connection
                </button>
              )}
            </div>
          )}

          {result && (
            <section className="mars-route-results" aria-label="Computed route">
              <h2>Route results</h2>
              <p className="mars-source-badge">{result.source.label}</p>
              <dl>
                <div>
                  <dt>Route distance</dt>
                  <dd>{distance(result.distance_m)}</dd>
                </div>
                <div>
                  <dt>Direct spherical distance</dt>
                  <dd>{distance(result.straight_line_m)}</dd>
                </div>
                <div>
                  <dt>Analysis spacing</dt>
                  <dd>
                    {distance(
                      result.source.analysis_resolution_m ??
                        result.source.native_resolution_m,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Mean / maximum slope</dt>
                  <dd>
                    {number(result.mean_slope_deg)}° /{" "}
                    {number(result.max_slope_deg)}°
                  </dd>
                </div>
              </dl>
              <p className="mars-resolution-note">
                {result.warning ||
                  "Endpoints are snapped to the elevation grid. Route optimality applies to this bounded analysis window."}
              </p>
              <label className="mars-exaggeration-control">
                Vertical exaggeration <strong>{exaggeration}×</strong>
                <input
                  aria-label="Vertical exaggeration"
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={exaggeration}
                  onChange={(event) =>
                    setExaggeration(Number(event.target.value))
                  }
                />
              </label>
              <div className="mars-exaggeration-footer">
                <span>Display only; elevations stay unchanged.</span>
                <button onClick={() => setExaggeration(1)}>
                  True scale · 1×
                </button>
              </div>
              <a
                className="mars-download-route"
                href={`/api/v1/explorer/routes/${result.id}/geojson`}
                download="mars-route.geojson"
              >
                <ArrowDownToLine size={15} />
                Download this route
              </a>
            </section>
          )}
          <details className="mars-methods">
            <summary>Coverage, resolution & methods</summary>
            <p>
              The whole-planet map uses NASA MOLA topography, an
              observed-derived grid with interpolated bins. Its 16 pixels/degree
              correspond to about3.7 km at the equator. It is suitable for
              regional terrain comparison, not rover-scale hazard assessment.
            </p>
            <p>
              Where the complete routing window fits inside the Jezero CTX
              footprint, routing uses that separate20 m source. Large windows
              are sampled more coarsely to keep calculations bounded; the result
              reports its actual analysis spacing.
            </p>
            <p>
              Costs favor lower slopes and roughness. Path lengths use the
              source's Mars sphere, not Earth's radius. The 3D surface and path
              share projected metric coordinates; exaggeration changes display
              proportions only.
            </p>
            {metadata && (
              <>
                <a href={metadata.global.url} target="_blank" rel="noreferrer">
                  MOLA source ↗
                </a>
                <a href={metadata.detail.url} target="_blank" rel="noreferrer">
                  CTX source ↗
                </a>
              </>
            )}
          </details>
        </aside>
      </main>
    </div>
  );
}

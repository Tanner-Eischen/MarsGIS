import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Check,
  Layers,
  LoaderCircle,
  MapPin,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  assetUrl,
  getPortfolioStatus,
  loadProbe,
  loadScene,
  probeCell,
  startPortfolioRun,
} from "../lib/portfolioData";
import type {
  Candidate,
  LayerId,
  PortfolioScene,
  PortfolioStatus,
} from "../lib/portfolioData";
import "../portfolio.css";

const number = (value: number, digits = 2) =>
  value.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
const km = (meters: number) => `${number(meters / 1000)} km`;
const errorText = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "The analysis service could not complete this request.";

export default function PortfolioShowcase() {
  const [scene, setScene] = useState<PortfolioScene | null>(null);
  const [status, setStatus] = useState<PortfolioStatus | null>(null);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [starting, setStarting] = useState(false);
  const [layerId, setLayerId] = useState<LayerId>("elevation");
  const [rank, setRank] = useState(1);
  const [showAll, setShowAll] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [probe, setProbe] = useState<ArrayBuffer | null>(null);
  const [probeError, setProbeError] = useState("");
  const [probeRetry, setProbeRetry] = useState(0);
  const [cell, setCell] = useState<[number, number] | null>(null);
  const [rowInput, setRowInput] = useState("");
  const [colInput, setColInput] = useState("");
  const [cellError, setCellError] = useState("");
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    const connect = async () => {
      try {
        const current = await getPortfolioStatus(controller.signal);
        setStatus(current);
        if (current.data_available)
          setScene(await loadScene(controller.signal));
        if (current.error) setError(current.error);
      } catch (failure) {
        if (!controller.signal.aborted) setError(errorText(failure));
      }
    };
    void connect();
    return () => controller.abort();
  }, [reload]);

  useEffect(() => {
    if (status?.state !== "running") return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const next = await getPortfolioStatus(controller.signal);
        if (next.state !== "running" && next.data_available)
          setScene(await loadScene(controller.signal));
        if (next.error) setError(next.error);
        setStatus(next);
      } catch (failure) {
        if (!controller.signal.aborted) {
          setError(errorText(failure));
          setStatus(null);
        }
      }
    }, 1500);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [status]);

  useEffect(() => {
    if (!scene) return;
    const controller = new AbortController();
    setProbe(null);
    setProbeError("");
    void loadProbe(scene, controller.signal)
      .then(setProbe)
      .catch((failure) => {
        if (!controller.signal.aborted) setProbeError(errorText(failure));
      });
    return () => controller.abort();
  }, [scene, probeRetry]);

  const candidate =
    scene?.candidates.find((item) => item.rank === rank) ??
    scene?.candidates[0];
  useEffect(() => {
    if (!candidate) return;
    setCell([candidate.row, candidate.col]);
    setRowInput(String(candidate.row));
    setColInput(String(candidate.col));
    setCellError("");
  }, [candidate]);
  useEffect(() => {
    setImageError(false);
  }, [layerId, scene]);

  const start = async () => {
    setStarting(true);
    setError("");
    try {
      setStatus(await startPortfolioRun());
    } catch (failure) {
      setError(errorText(failure));
      setReload((value) => value + 1);
    } finally {
      setStarting(false);
    }
  };
  const running = starting || status?.state === "running";
  const layer =
    scene?.layers.find((item) => item.id === layerId) ?? scene?.layers[0];
  const selectCell = (row: number, col: number) => {
    setCell([row, col]);
    setRowInput(String(row));
    setColInput(String(col));
    setCellError("");
  };
  let values: Record<string, number | null> | null = null;
  if (scene && probe && cell)
    values = probeCell(scene, probe, cell[0], cell[1]);

  return (
    <div className="portfolio">
      <a className="skip-link" href="#terrain">
        Skip to terrain explorer
      </a>
      <header className="portfolio-header">
        <a href="/" className="portfolio-brand">
          <span aria-hidden="true" className="planet-mark" />
          MarsHab<span> / MarsGIS</span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#terrain">Explore terrain</a>
          <a href="#candidates">Candidates & routes</a>
          <a href="#methods">Methods</a>
          <a href="#downloads">Downloads</a>
        </nav>
        <span className="data-badge">
          <span aria-hidden="true" />
          NASA observations
        </span>
      </header>
      <main className="portfolio-main">
        <section className="portfolio-intro">
          <div>
            <p className="eyebrow">
              SCIENTIFIC SOFTWARE / MARS RECONNAISSANCE ORBITER
            </p>
            <h1>
              Jezero crater.
              <br />
              <span>Terrain, compared.</span>
            </h1>
            <p className="intro-copy">
              Explore measured Mars elevations, compare terrain-only candidate
              locations, and inspect the rover routes between them.
            </p>
            <div className="intro-meta">
              <span>18.25–18.45° N</span>
              <span>77.25–77.45° E</span>
              <span>20 m analysis grid</span>
            </div>
          </div>
          <div className="run-card">
            <p className="eyebrow">ONE REPRODUCIBLE WORKFLOW</p>
            <p>
              Real DEM → terrain derivatives → suitability → routes → exports.
            </p>
            <button
              className="primary-button"
              onClick={() => void start()}
              disabled={running || !status}
            >
              {running ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <Play size={16} />
              )}
              {running ? "Analysis running…" : "Run Jezero analysis"}
            </button>
            <p className="small-copy">
              Runs the validated Python calculations on the server. Source
              checksum verified; synthetic fallback disabled.
            </p>
            <p role="status" className="run-status">
              {running
                ? "Calculating terrain and three complete routes. Existing results remain available."
                : scene
                  ? scene.execution_mode === "server_computed_analysis"
                    ? "Showing results computed by this analysis service."
                    : "Showing a verified reference run. Run the analysis to recompute it."
                  : status
                    ? "Ready to calculate the Jezero example."
                    : error
                      ? "Analysis service unavailable."
                      : "Connecting to analysis service…"}
            </p>
          </div>
        </section>

        {error && (
          <div className="notice error-notice" role="alert">
            <div>
              <strong>Could not complete the request</strong>
              <p>{error}</p>
            </div>
            <button onClick={() => setReload((value) => value + 1)}>
              <RefreshCw size={15} />
              Retry connection
            </button>
          </div>
        )}

        {!scene && !error && (
          <div className="loading-panel" role="status">
            <LoaderCircle className="spin" />
            {running
              ? "Preparing real-Mars analysis results…"
              : status
                ? "Start the Jezero analysis to generate the terrain map, candidates and routes."
                : "Loading the scientific workspace…"}
          </div>
        )}

        {scene && candidate && layer && (
          <>
            <section id="terrain" className="explorer-section">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">01 / TERRAIN EXPLORER</p>
                  <h2>One region. Five views.</h2>
                </div>
                <span className="grid-caption">
                  {scene.grid.shape[1]} × {scene.grid.shape[0]} cells · North up
                </span>
              </div>
              <div className="explorer-grid">
                <div className="terrain-panel">
                  <div
                    className="layer-buttons"
                    role="group"
                    aria-label="Terrain layer"
                  >
                    {scene.layers.map((item) => (
                      <button
                        key={item.id}
                        aria-pressed={item.id === layerId}
                        onClick={() => setLayerId(item.id)}
                      >
                        {item.id === layerId && <Layers size={14} />}
                        {item.label}
                      </button>
                    ))}
                  </div>
                  <div className="map-toolbar">
                    <span>
                      {layer.label}{" "}
                      <span className="muted">/ {layer.unit}</span>
                    </span>
                    <div role="group" aria-label="Map zoom">
                      <button
                        aria-label="Zoom out"
                        disabled={zoom === 1}
                        onClick={() =>
                          setZoom((value) => Math.max(1, value / 2))
                        }
                      >
                        −
                      </button>
                      <span aria-live="polite">{zoom}×</span>
                      <button
                        aria-label="Zoom in"
                        disabled={zoom === 4}
                        onClick={() =>
                          setZoom((value) => Math.min(4, value * 2))
                        }
                      >
                        +
                      </button>
                      <button onClick={() => setZoom(1)}>Fit region</button>
                    </div>
                  </div>
                  <TerrainView
                    scene={scene}
                    candidate={candidate}
                    layerId={layer.id}
                    zoom={zoom}
                    showAll={showAll}
                    cell={cell}
                    onSelect={setRank}
                    onProbe={selectCell}
                    onImageError={() => setImageError(true)}
                  />
                  {imageError && (
                    <p role="alert" className="map-error">
                      The terrain image could not load.{" "}
                      <button onClick={() => setReload((value) => value + 1)}>
                        Reload analysis
                      </button>
                    </p>
                  )}
                  <div className="map-legend">
                    <span>
                      {number(layer.min)} {layer.unit}
                    </span>
                    <span
                      className="color-ramp"
                      style={{
                        background: `linear-gradient(to right, ${layer.palette.map((rgb) => `rgb(${rgb.join(",")})`).join(",")})`,
                      }}
                    />
                    <span>
                      {number(layer.max)} {layer.unit}
                    </span>
                  </div>
                  <p className="map-note">
                    Local Mars metric grid. Select a marker to compare a
                    candidate; click a raster cell to inspect its measurements.
                    Transparent cells have no valid value.
                  </p>
                </div>
                <aside className="inspector-panel">
                  <p className="eyebrow">RASTER CELL INSPECTOR</p>
                  <h3>Read the measurements.</h3>
                  <p className="small-copy">
                    Values come from the exported rasters, after bilinear
                    reprojection. Cell indices start at zero.
                  </p>
                  <form
                    onSubmit={(event) => {
                      event.preventDefault();
                      const r = Number(rowInput),
                        c = Number(colInput);
                      if (
                        rowInput === "" ||
                        colInput === "" ||
                        !Number.isInteger(r) ||
                        !Number.isInteger(c) ||
                        r < 0 ||
                        c < 0 ||
                        r >= scene.grid.shape[0] ||
                        c >= scene.grid.shape[1]
                      ) {
                        setCellError(
                          "Enter a row and column inside this grid.",
                        );
                        return;
                      }
                      selectCell(r, c);
                    }}
                  >
                    <div className="cell-inputs">
                      <label>
                        Row
                        <input
                          type="number"
                          min="0"
                          max={scene.grid.shape[0] - 1}
                          step="1"
                          value={rowInput}
                          onChange={(event) => setRowInput(event.target.value)}
                          required
                        />
                      </label>
                      <label>
                        Column
                        <input
                          type="number"
                          min="0"
                          max={scene.grid.shape[1] - 1}
                          step="1"
                          value={colInput}
                          onChange={(event) => setColInput(event.target.value)}
                          required
                        />
                      </label>
                    </div>
                    <button type="submit">Inspect cell</button>
                  </form>
                  {cellError && <p role="alert">{cellError}</p>}
                  {probeError ? (
                    <div className="map-error" role="alert">
                      <p>{probeError}</p>
                      <button
                        onClick={() => setProbeRetry((value) => value + 1)}
                      >
                        Retry measurements
                      </button>
                    </div>
                  ) : !probe ? (
                    <p role="status" className="small-copy">
                      Loading and checksum-verifying cell measurements…
                    </p>
                  ) : (
                    <dl className="probe-values">
                      {scene.layers.map((item) => (
                        <div key={item.id}>
                          <dt>{item.label}</dt>
                          <dd>
                            {values?.[item.id] == null
                              ? "NoData"
                              : `${number(values[item.id]!, item.id === "suitability" ? 4 : 2)} ${item.unit}`}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  <div className="inspector-note">
                    <Check size={16} />
                    <p>
                      Each measurement buffer is checked against the analysis
                      package’s SHA-256 before use.
                    </p>
                  </div>
                </aside>
              </div>
            </section>

            <section id="candidates" className="candidates-section">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">02 / CANDIDATES & ROUTES</p>
                  <h2>Compare the terrain tradeoffs.</h2>
                </div>
                <label className="route-toggle">
                  <input
                    type="checkbox"
                    checked={showAll}
                    onChange={(event) => setShowAll(event.target.checked)}
                  />
                  Show all routes
                </label>
              </div>
              <div className="candidate-cards">
                {scene.candidates.map((item) => (
                  <button
                    className={`candidate-card ${item.rank === candidate.rank ? "selected" : ""}`}
                    key={item.rank}
                    aria-pressed={item.rank === candidate.rank}
                    onClick={() => setRank(item.rank)}
                  >
                    <div className="candidate-title">
                      <span>0{item.rank}</span>
                      <MapPin size={18} />
                      <h3>Candidate {item.rank}</h3>
                      {item.rank === candidate.rank && <Check size={18} />}
                    </div>
                    <p>
                      {number(item.lat, 5)}° N · {number(item.lon, 5)}° E
                    </p>
                    <dl>
                      <div>
                        <dt>Relative score</dt>
                        <dd>{number(item.score, 4)}</dd>
                      </div>
                      <div>
                        <dt>Slope</dt>
                        <dd>{number(item.slope_deg, 3)}°</dd>
                      </div>
                      <div>
                        <dt>Roughness</dt>
                        <dd>{number(item.roughness_m, 3)} m</dd>
                      </div>
                    </dl>
                    <span className="candidate-link">
                      {item.rank === candidate.rank
                        ? "Selected route"
                        : "Select route"}
                      <ArrowUpRight size={15} />
                    </span>
                  </button>
                ))}
              </div>
              <div className="route-detail">
                <div>
                  <p className="eyebrow">ROUTE TO CANDIDATE {candidate.rank}</p>
                  <h3>A terrain-aware path.</h3>
                  <p>
                    All routes share one proposed start. A* avoids invalid and
                    over-threshold cells, with no diagonal corner cutting or
                    smoothing.
                  </p>
                </div>
                <dl className="route-metrics">
                  <div>
                    <dt>Route length</dt>
                    <dd>{km(candidate.distance_m)}</dd>
                  </div>
                  <div>
                    <dt>Straight-line distance</dt>
                    <dd>{km(candidate.straight_line_m)}</dd>
                  </div>
                  <div>
                    <dt>Terrain penalty</dt>
                    <dd>
                      {number(candidate.terrain_cost_m_equivalent / 1000)} km
                      equivalent
                    </dd>
                  </div>
                  <div>
                    <dt>Penalty reduction vs distance-only route</dt>
                    <dd>
                      {number(
                        100 *
                          (1 -
                            candidate.terrain_cost_m_equivalent /
                              candidate.distance_only_route_terrain_cost_m_equivalent),
                        1,
                      )}
                      %
                    </dd>
                  </div>
                </dl>
              </div>
              <p className="section-note">
                Scores are relative to this region, not safety probabilities.
                Terrain penalty is an illustrative cost, not energy or travel
                time. Candidates are raster cells, not certified habitat
                footprints.
              </p>
            </section>

            <section id="methods" className="methods-section">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">03 / DATA & METHODS</p>
                  <h2>Trace every result.</h2>
                </div>
                <a
                  className="text-link"
                  href={scene.source.citation}
                  target="_blank"
                  rel="noreferrer"
                >
                  USGS source description
                  <ArrowUpRight size={16} />
                </a>
              </div>
              <div className="method-grid">
                <article>
                  <span className="method-number">01</span>
                  <h3>Observed elevation</h3>
                  <p>
                    NASA MRO CTX stereo terrain, MOLA controlled, distributed by
                    USGS. The original DEM is 20 m per pixel. No synthetic
                    terrain is substituted.
                  </p>
                </article>
                <article>
                  <span className="method-number">02</span>
                  <h3>A shared metric grid</h3>
                  <p>
                    Bilinear reprojection to a local Mars spherical azimuthal
                    equidistant grid. Elevation, slope, roughness, hillshade and
                    suitability share the same transform.
                  </p>
                </article>
                <article>
                  <span className="method-number">03</span>
                  <h3>Explicit decisions</h3>
                  <p>
                    Slope weight {number(scene.configuration.weights.slope, 2)};
                    roughness weight{" "}
                    {number(scene.configuration.weights.roughness, 2)}. Both
                    lower-is-better, min-max normalized. Routes require slope
                    ≤15° and 5×5 roughness ≤20 m.
                  </p>
                </article>
              </div>
              <details className="provenance">
                <summary>Inspect provenance and scientific limits</summary>
                <div>
                  <p>
                    <strong>Source SHA-256</strong>
                  </p>
                  <code>{scene.source.sha256}</code>
                  <p>
                    <a href={scene.source.url} target="_blank" rel="noreferrer">
                      Download original USGS DEM ↗
                    </a>{" "}
                    · {number(scene.source.bytes / 1e6, 2)} MB
                  </p>
                  <p>
                    <strong>Coordinate reference</strong>
                  </p>
                  <pre>{scene.grid.crs_wkt}</pre>
                  <p>
                    Roughness is elevation standard deviation over a 100 m
                    footprint, including regional slope. Hillshade models local
                    illumination, not distant cast shadows. The example infers
                    no ice, dust, soil strength, solar power, or mission
                    economics.
                  </p>
                  <p>
                    All distances use this small local planar grid. GeoJSON
                    contains Mars east longitude and planetocentric latitude,
                    not terrestrial WGS84. Candidate separation is at least 1.5
                    km; only the largest traversable component is searched.
                  </p>
                </div>
              </details>
            </section>
            <section id="downloads" className="downloads-section">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">04 / RESULTS YOU CAN KEEP</p>
                  <h2>Download the analysis.</h2>
                </div>
              </div>
              <div className="download-grid">
                {scene.downloads.map((item) => (
                  <a
                    key={item.id}
                    href={assetUrl(item.path, scene.asset_base)}
                    download={item.path}
                    className="download-card"
                  >
                    <ArrowDownToLine size={20} />
                    <div>
                      <strong>{item.label}</strong>
                      <span>
                        {item.path} · {number(item.bytes / 1e6, 2)} MB
                      </span>
                    </div>
                    <ArrowUpRight size={16} />
                  </a>
                ))}
              </div>
              <p className="section-note">
                Downloads belong to the displayed run. The manifest records
                source, parameters, software versions, coordinate systems and
                file checksums.
              </p>
            </section>
          </>
        )}
      </main>
      <footer className="portfolio-footer">
        <span>MarsHab / MarsGIS</span>
        <span>Measured data. Explicit assumptions. Reproducible results.</span>
      </footer>
    </div>
  );
}

function TerrainView({
  scene,
  candidate,
  layerId,
  zoom,
  showAll,
  cell,
  onSelect,
  onProbe,
  onImageError,
}: {
  scene: PortfolioScene;
  candidate: Candidate;
  layerId: LayerId;
  zoom: number;
  showAll: boolean;
  cell: [number, number] | null;
  onSelect: (rank: number) => void;
  onProbe: (row: number, col: number) => void;
  onImageError: () => void;
}) {
  const [height, width] = scene.grid.shape;
  const viewWidth = width / zoom,
    viewHeight = height / zoom;
  const left = Math.max(
    0,
    Math.min(width - viewWidth, candidate.col + 0.5 - viewWidth / 2),
  );
  const top = Math.max(
    0,
    Math.min(height - viewHeight, candidate.row + 0.5 - viewHeight / 2),
  );
  const layer = scene.layers.find((item) => item.id === layerId)!;
  return (
    <div
      className="terrain-map"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      <svg
        role="group"
        aria-label="Jezero terrain map, local grid, north up"
        viewBox={`${left} ${top} ${viewWidth} ${viewHeight}`}
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) return;
          const col = Math.floor(
            left + ((event.clientX - rect.left) / rect.width) * viewWidth,
          );
          const row = Math.floor(
            top + ((event.clientY - rect.top) / rect.height) * viewHeight,
          );
          if (row >= 0 && col >= 0 && row < height && col < width)
            onProbe(row, col);
        }}
      >
        <image
          key={layer.image}
          href={assetUrl(layer.image, scene.asset_base)}
          x="0"
          y="0"
          width={width}
          height={height}
          onError={onImageError}
        />
        {scene.routes
          .filter((route) => showAll || route.rank === candidate.rank)
          .map((route) => (
            <polyline
              data-testid={`route-${route.rank}`}
              key={route.rank}
              points={route.points.map((point) => point.join(",")).join(" ")}
              fill="none"
              stroke={route.rank === candidate.rank ? "#fff4ca" : "#8be3e1"}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        <g aria-label="Proposed common start">
          <circle
            cx={scene.start_pixel[1] + 0.5}
            cy={scene.start_pixel[0] + 0.5}
            r={5 / zoom}
            fill="#0a2428"
            stroke="white"
            strokeWidth={1.5 / zoom}
          />
          <text
            x={scene.start_pixel[1] + 9 / zoom}
            y={scene.start_pixel[0] + 4 / zoom}
            fill="white"
            fontSize={10 / zoom}
          >
            START
          </text>
        </g>
        {cell && (
          <rect
            x={cell[1]}
            y={cell[0]}
            width="1"
            height="1"
            fill="none"
            stroke="#fff"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}
        {scene.candidates.map((item) => (
          <g
            key={item.rank}
            role="button"
            tabIndex={0}
            aria-label={`Select candidate ${item.rank} on map`}
            aria-pressed={item.rank === candidate.rank}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(item.rank);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(item.rank);
              }
            }}
          >
            <circle
              cx={item.col + 0.5}
              cy={item.row + 0.5}
              r={11 / zoom}
              fill={item.rank === candidate.rank ? "#ffb46a" : "#172629"}
              stroke="#fff4ca"
              strokeWidth={1.5 / zoom}
            />
            <text
              x={item.col + 0.5}
              y={item.row + 0.5}
              dy=".35em"
              fill={item.rank === candidate.rank ? "#131d21" : "#fff4ca"}
              textAnchor="middle"
              fontSize={11 / zoom}
              fontWeight="700"
            >
              {item.rank}
            </text>
          </g>
        ))}
      </svg>
      <span className="north-label" aria-hidden="true">
        ↑ N
      </span>
    </div>
  );
}

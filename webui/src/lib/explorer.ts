export interface MarsPoint {
  lat: number;
  lon: number;
}

export interface ExplorerSource {
  id: string;
  label: string;
  url: string;
  sha256: string;
  radius_m: number;
  native_resolution_m: number;
  analysis_resolution_m?: number;
  limitations?: string | string[];
  [key: string]: unknown;
}

export interface ExplorerMetadata {
  global: ExplorerSource;
  detail: ExplorerSource & { bounds: [number, number, number, number] };
  tiles: { mola: string; ctx: string };
  initial_view: { lat: number; lon: number; zoom: number };
  detail_view: { lat: number; lon: number; zoom: number };
  max_zoom: number;
  [key: string]: unknown;
}

export interface RoutePoint extends MarsPoint {
  elevation_m: number;
  x_m: number;
  y_m: number;
}

export interface ExplorerRoute {
  id: string;
  source: ExplorerSource;
  requested: { start: MarsPoint; end: MarsPoint };
  snapped: { start: MarsPoint; end: MarsPoint };
  path: RoutePoint[];
  geojson: Record<string, unknown>;
  distance_m: number;
  straight_line_m: number;
  terrain_cost_m_equivalent: number;
  max_slope_deg: number;
  mean_slope_deg: number;
  grid: { shape: [number, number]; [key: string]: unknown };
  surface: {
    x_m: number[][];
    y_m: number[][];
    elevation_m: (number | null)[][];
    display_step: number;
    [key: string]: unknown;
  };
  elevation_profile: { distance_m: number; elevation_m: number }[];
  warning?: string;
  [key: string]: unknown;
}

export interface ExplorerJob {
  id: string;
  state: "running" | "ready" | "failed";
  poll_url?: string;
  error?: string;
  result?: ExplorerRoute;
}

export type PickMode = "start" | "end";
export interface MapCommand {
  id: number;
  kind: "global" | "jezero" | "route";
}

export function normalizeLongitude(lon: number): number {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

export function validPoint(point: MarsPoint | null): point is MarsPoint {
  return (
    !!point &&
    Number.isFinite(point.lat) &&
    Number.isFinite(point.lon) &&
    Math.abs(point.lat) <= 90 &&
    Math.abs(point.lon) <= 180
  );
}

async function jsonResponse(response: Response) {
  if (!response.headers.get("Content-Type")?.includes("application/json")) {
    throw new Error(
      `The Mars API returned ${response.status} without JSON. Check the unified service.`,
    );
  }
  const value = await response.json();
  if (!response.ok) {
    const detail =
      typeof value.detail === "string" ? value.detail : value.error;
    throw new Error(detail || `Mars API request failed (${response.status}).`);
  }
  return value;
}

export async function loadExplorer(
  signal?: AbortSignal,
): Promise<ExplorerMetadata> {
  const data = await jsonResponse(
    await fetch("/api/v1/explorer/metadata", { signal }),
  );
  if (
    !data.global ||
    !data.detail ||
    !Array.isArray(data.detail.bounds) ||
    !data.tiles?.mola ||
    !data.tiles?.ctx
  )
    throw new Error("The Mars API returned incomplete source metadata.");
  return data;
}

export async function planExplorerRoute(
  start: MarsPoint,
  end: MarsPoint,
  maxSlope: number,
): Promise<ExplorerJob> {
  if (!validPoint(start) || !validPoint(end))
    throw new Error("Select valid start and end coordinates.");
  return jsonResponse(
    await fetch("/api/v1/explorer/routes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, end, max_slope_deg: maxSlope }),
    }),
  );
}

export async function pollExplorerRoute(
  id: string,
  signal?: AbortSignal,
): Promise<ExplorerJob> {
  if (!/^[A-Za-z0-9_-]+$/.test(id))
    throw new Error("Invalid route job identifier.");
  return jsonResponse(await fetch(`/api/v1/explorer/routes/${id}`, { signal }));
}

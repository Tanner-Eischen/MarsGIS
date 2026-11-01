"""Pinned source validation, geographic tile centers, transparency and memory bounds."""

import hashlib
import io

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from marshab.core import planetary_data as data


@pytest.fixture
def sources(tmp_path, monkeypatch):
    # Tiny substitutes retain real encoding, hashing, CRS and NoData behavior.
    values = (np.arange(32).reshape(4, 8) * 600 - 8000).astype(">i2")
    values[0, 0], values[0, 4] = -8177, 21171
    values[3, 3], values[3, 7], values[2, 0] = 21171, -8177, 0
    global_path = tmp_path / "mola16.img"
    global_path.write_bytes(values.tobytes())
    monkeypatch.setattr(data, "GLOBAL_SHAPE", values.shape)
    monkeypatch.setattr(data, "GLOBAL_SHA256", hashlib.sha256(values.tobytes()).hexdigest())
    ctx_path = tmp_path / "ctx.tif"
    detail = np.full((4, 4), 1000, dtype=np.float32)
    detail[0, 0] = -9999
    with rasterio.open(
        ctx_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="+proj=longlat +R=3396190 +no_defs",
        transform=from_origin(10, 2, 0.5, 0.5),
        nodata=-9999,
    ) as dst:
        dst.write(detail, 1)
    monkeypatch.setattr(data, "SOURCE_SHA256", data._sha256(ctx_path))
    return global_path, ctx_path


@pytest.fixture
def store(sources):
    return data.PlanetaryDataStore(*sources)


def pixels(tile):
    with Image.open(io.BytesIO(tile)) as image:
        assert image.size == (256, 256) and image.mode == "RGBA"
        return np.asarray(image)


def test_readonly_memory_map_metadata_and_actual_ctx_bounds(store):
    assert isinstance(store.global_array, np.memmap)
    assert store.global_array.dtype == np.dtype(">i2")
    assert not store.global_array.flags.writeable
    assert store.global_array[2, 0] == 0
    metadata = store.metadata()
    assert metadata["global"]["radius_m"] == 3396000
    assert metadata["global"]["nodata"] is None
    assert metadata["detail"]["radius_m"] == 3396190
    assert store.ctx_bounds == (10, 0, 12, 2)
    assert metadata["detail_view"] == {"lat": 1, "lon": 11, "zoom": 9}
    metadata["global"]["shape"][0] = -1
    assert store.metadata()["global"]["shape"] == [4, 8]


def test_geographic_tile_centers_longitude_wrap_and_zero_are_valid(store, monkeypatch):
    # Isolate geographic sampling from separately tested display relief.
    monkeypatch.setattr(store, "_native_shade", lambda rows, cols: np.ones((len(rows), len(cols))))
    western = pixels(store.tile("mola", 0, 0, 0))  # [-180,0] becomes native [180,360].
    eastern = pixels(store.tile("mola", 0, 1, 0))  # [0,180].
    np.testing.assert_array_equal(western[0, 0, :3], data.PALETTE[-1])
    np.testing.assert_array_equal(eastern[0, 0, :3], data.PALETTE[0])
    np.testing.assert_array_equal(western[-1, -1, :3], data.PALETTE[0])
    np.testing.assert_array_equal(eastern[-1, -1, :3], data.PALETTE[-1])
    assert (western[..., 3] == 255).all() and (eastern[..., 3] == 255).all()
    # This pixel samples the known zero source cell: it must remain opaque.
    assert eastern[128, 0, 3] == 255
    assert eastern[128, 0, :3].tolist() != list(data.PALETTE[0])
    assert data.tile_bounds(0, 0, 0) == (-180, -90, 0, 90)
    assert data.tile_bounds(0, 1, 0) == (0, -90, 180, 90)


@pytest.mark.parametrize(
    "indices", [(-1, 0, 0), (13, 0, 0), (0, -1, 0), (0, 2, 0), (0, 0, 1), (True, 0, 0), (1.5, 0, 0)]
)
def test_invalid_tile_bounds_rejected(store, indices):
    with pytest.raises(ValueError, match="Tile indices"):
        store.tile("mola", *indices)


def test_unknown_layer_rejected(store):
    with pytest.raises(ValueError, match="Unknown"):
        store.tile("synthetic", 0, 0, 0)


def test_ctx_transparent_outside_coverage_and_on_nodata(store):
    assert not pixels(store.tile("ctx", 0, 0, 0))[..., 3].any()
    tile = pixels(store.tile("ctx", 9, 541, 251))
    assert (tile[..., 3] == 0).any()
    assert (tile[..., 3] == 255).any()


def test_lru_cache_bounds_entries_and_bytes(store, monkeypatch):
    monkeypatch.setattr(data, "MAX_CACHED_TILES", 3)
    monkeypatch.setattr(data, "MAX_CACHE_BYTES", 2048)
    for x in range(8):
        for y in range(4):
            store.tile("mola", 2, x, y)
            info = store.cache_info()
            assert info["entries"] <= 3 and info["bytes"] <= 2048
    first = store.tile("ctx", 0, 0, 0)
    assert store.tile("ctx", 0, 0, 0) is first


def test_corrupt_and_missing_pinned_sources_fail_without_download(sources, tmp_path):
    global_path, ctx_path = sources
    with pytest.raises(FileNotFoundError, match="missing"):
        data.PlanetaryDataStore(tmp_path / "absent.img", ctx_path)
    body = global_path.read_bytes()
    global_path.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.PlanetaryDataStore(global_path, ctx_path)
    global_path.write_bytes(b"too short")
    with pytest.raises(ValueError, match="byte length"):
        data.PlanetaryDataStore(global_path, ctx_path)


def test_streamed_provisioning_pins_source_and_cleans_bad_partial(sources, tmp_path, monkeypatch):
    body = sources[0].read_bytes()
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(body))
    path = tmp_path / "provisioned" / "mola.img"
    assert data.ensure_global_dem(path).read_bytes() == body
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(bytes(len(body)))
    )
    bad = tmp_path / "bad.img"
    with pytest.raises(ValueError, match="pinned"):
        data.ensure_global_dem(bad)
    assert not bad.exists() and not list(tmp_path.glob("*.partial"))


def test_environment_factory_uses_provisioned_paths(sources, monkeypatch):
    from marshab.config import Config

    monkeypatch.setenv("MARSHAB_GLOBAL_DEM_PATH", str(sources[0]))
    monkeypatch.setenv("MARSHAB_REFERENCE_DEM_PATH", str(sources[1]))
    store = data.from_config(Config.model_construct())
    assert store.ctx_path == sources[1]


def test_cli_default_matches_configured_runtime_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from marshab.config import Config, PathsConfig
    from scripts import prepare_planetary_data as cli

    cache = tmp_path / "configured-cache"
    config = Config.model_construct(paths=PathsConfig(cache_dir=cache))
    monkeypatch.setattr(cli, "get_config", lambda: config)
    for name in ("MARSHAB_GLOBAL_DEM_PATH", "MARSHAB_REFERENCE_DEM_PATH"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def provision(path):
        calls.append(path)
        return path

    def store(global_path, ctx_path):
        assert (global_path, ctx_path) == (cache / "mola16.img", cache / "jezero_ctx20m.tif")
        return SimpleNamespace(
            global_provenance={"sha256": "global"}, ctx_provenance={"sha256": "ctx"}
        )

    monkeypatch.setattr(cli, "ensure_global_dem", provision)
    monkeypatch.setattr(cli, "acquire_dem", provision)
    monkeypatch.setattr(cli, "PlanetaryDataStore", store)
    monkeypatch.setattr(cli.sys, "argv", ["prepare_planetary_data.py"])
    cli.main()
    assert calls == [cache / "mola16.img", cache / "jezero_ctx20m.tif"]
    runtime_paths = []

    def capture_constructor(self, global_path, ctx_path):
        runtime_paths.extend([global_path, ctx_path])

    monkeypatch.setattr(data.PlanetaryDataStore, "__init__", capture_constructor)
    data.from_config(config)
    assert runtime_paths == calls


def test_native_flat_relief_preserves_palette_and_scientific_elevations(store, monkeypatch):
    elevation = np.full(store.global_array.shape, 0, dtype=float)
    monkeypatch.setattr(store, "global_array", elevation)
    shade = store._native_shade(np.arange(4), np.arange(8))
    np.testing.assert_array_equal(shade, np.ones((4, 8)))
    assert store.tile("mola", 0, 0, 0) == data._png(np.zeros((256, 256)))
    assert store.global_array is elevation and (elevation == 0).all()
    assert store.metadata()["global"]["display_relief"]["visual_only"]
    assert store.metadata()["global"]["display_relief"]["z_factor"] == 3


@pytest.mark.parametrize("direction", ["east", "south"])
def test_native_tilt_relief_matches_surface_normal_reference(store, monkeypatch, direction):
    height, width = store.global_array.shape
    row, col = 1, 3
    native_slope = 0.1
    if direction == "east":
        latitude = np.pi / 2 - (row + 0.5) * np.pi / height
        spacing = data.GLOBAL_RADIUS_M * (2 * np.pi / width) * np.cos(latitude)
        elevation = np.broadcast_to(np.arange(width) * native_slope * spacing, (height, width))
        normal = np.array([-0.3, 0, 1])
    else:
        spacing = data.GLOBAL_RADIUS_M * np.pi / height
        elevation = np.broadcast_to(
            (np.arange(height) * native_slope * spacing)[:, None], (height, width)
        )
        normal = np.array([0, 0.3, 1])
    monkeypatch.setattr(store, "global_array", elevation)
    sun = np.array([-0.5, 0.5, np.sqrt(0.5)])  # Northwest, 45-degree altitude.
    expected = np.dot(normal / np.linalg.norm(normal), sun) / np.sqrt(0.5)
    np.testing.assert_allclose(
        store._native_shade(np.array([row]), np.array([col])), expected, atol=1e-14
    )
    # One-sided north/south source edges have the same gradient on a southward plane.
    if direction == "south":
        np.testing.assert_allclose(
            store._native_shade(np.array([0, height - 1]), np.array([col])), expected, atol=1e-14
        )


def test_native_relief_wraps_longitude_and_is_finite_at_polar_edges(store, monkeypatch):
    height, width = store.global_array.shape
    periodic = np.broadcast_to(
        10000 * np.cos(2 * np.pi * np.arange(width) / width), (height, width)
    )
    monkeypatch.setattr(store, "global_array", periodic)
    shade = store._native_shade(np.array([0, height - 1]), np.array([0, width - 1]))
    assert np.isfinite(shade).all()
    assert (shade >= 0.35).all() and (shade <= 1.3).all()
    # Wrapped east/west neighbors are equal at the periodic cosine maximum.
    np.testing.assert_allclose(shade[:, 0], 1, atol=1e-14)


@pytest.mark.parametrize("same_tile", [False, True])
def test_inflight_rendering_is_bounded_and_queued_requests_recheck_cache(
    store, monkeypatch, same_tile
):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    active, peak, calls = 0, 0, 0

    def render(layer, bounds):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)  # Let concurrent requests overlap while holding their arrays.
            return b"rendered tile"
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(store, "_render_tile", render)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(store.tile, "mola", 2, 0 if same_tile else index, 0)
            for index in range(8)
        ]
        assert all(future.result(timeout=5) == b"rendered tile" for future in futures)
    assert peak <= 2 and active == 0
    # At most the first two misses can render the same tile; queued requests reuse it.
    assert calls <= 2 if same_tile else calls == 8

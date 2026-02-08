import pytest

from marshab.core.raster_contracts import (
    FALLBACK_REASON_FIELD,
    IS_FALLBACK_FIELD,
    RASTER_RESPONSE_METADATA_FIELDS,
    REAL_DEM_UNAVAILABLE_ERROR_CODE,
    build_raster_response_metadata,
    build_real_dem_unavailable_payload,
    epsg4326_x_cols,
    epsg4326_y_rows,
    normalize_lon_for_client,
    normalize_lon_for_raster,
)


@pytest.mark.parametrize(("z", "x_cols", "y_rows"), [(0, 2, 1), (1, 4, 2), (3, 16, 8)])
def test_epsg4326_matrix_dimensions(z: int, x_cols: int, y_rows: int):
    assert epsg4326_x_cols(z) == x_cols
    assert epsg4326_y_rows(z) == y_rows


def test_longitude_normalization_contract():
    assert normalize_lon_for_raster(-180.0) == pytest.approx(180.0)
    assert normalize_lon_for_raster(370.0) == pytest.approx(10.0)
    assert normalize_lon_for_client(350.0) == pytest.approx(-10.0)
    assert normalize_lon_for_client(-185.0) == pytest.approx(175.0)


def test_raster_metadata_fields_and_builder():
    metadata = build_raster_response_metadata(
        dataset_requested="hirise",
        dataset_used="mola",
        is_fallback=True,
        fallback_reason="hirise_unavailable",
    )
    assert tuple(metadata.keys()) == RASTER_RESPONSE_METADATA_FIELDS
    assert metadata[IS_FALLBACK_FIELD] is True
    assert metadata[FALLBACK_REASON_FIELD] == "hirise_unavailable"


def test_real_dem_unavailable_payload_builder():
    payload = build_real_dem_unavailable_payload(
        detail="Real DEM unavailable for tile_basemap.",
        dataset_requested="mola",
        dataset_used="mola",
        is_fallback=False,
        fallback_reason=None,
    )
    assert payload["error"] == REAL_DEM_UNAVAILABLE_ERROR_CODE
    assert payload["dataset_requested"] == "mola"
    assert payload["dataset_used"] == "mola"
    assert payload["is_fallback"] is False

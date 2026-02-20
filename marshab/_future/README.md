# Future / Unimplemented Features

Components in this directory are preserved for future implementation. They are **not** loaded or mounted by the application.

## Contents

- **fusion_routes.py** – API routes for multi-resolution data fusion. Requires `multi_resolution_fusion.py`.
- **multi_resolution_fusion.py** – Core fusion logic (weighted average, hierarchical, adaptive blending).
- **visibility.py** – VisibilityAnalyzer for viewshed / line-of-sight analysis (Comms overlay "Coming soon").

## Enabling

1. **Fusion**: In `marshab/web/api.py`, add:
   ```python
   from marshab._future import fusion_routes
   app.include_router(fusion_routes.router, prefix="/api/v1", tags=["fusion"])
   ```

2. **Visibility**: Create a route that imports `marshab._future.visibility.VisibilityAnalyzer` and exposes a viewshed endpoint.

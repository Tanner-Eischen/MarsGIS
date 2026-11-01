FROM marshab-release:20261031-pythonpatched
USER root
RUN rm -rf /app/marshab /app/scripts /app/webui/dist
COPY marshab/ /app/marshab/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/pyproject.toml
COPY webui/dist/ /app/webui/dist/
COPY source-manifest.json /app/mh03-source-manifest.json
USER marshab

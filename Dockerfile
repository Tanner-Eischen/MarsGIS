FROM condaforge/mambaforge:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment definition
COPY environment.yml .

# Create conda environment
RUN mamba env create -f environment.yml && \
    mamba clean --all -y

# Activate environment in shell
SHELL ["conda", "run", "-n", "marshab", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Install marshab package in development mode
RUN pip install -e .

# Set entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "marshab"]
CMD ["python", "-m", "marshab"]


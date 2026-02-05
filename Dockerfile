# All-in-one Docker image for MadGraphMLProducer
# Contains MadGraph5, Pythia8, and all processing tools

FROM scailfin/madgraph5-amc-nlo:mg5_amc3.5.4

# Install Python packages for post-processing
RUN python3 -m pip install --no-cache-dir \
    numpy \
    h5py \
    pyhepmc \
    pylhe \
    pyjet \
    pydantic \
    jinja2 \
    pyyaml \
    click \
    tqdm \
    awkward \
    vector

# Create working directories
WORKDIR /app
RUN mkdir -p /app/output /app/work

# Copy the application code
COPY src/ /app/src/
COPY cards/ /app/cards/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY run_docker.py /app/run_docker.py
COPY pyproject.toml /app/

# Install the package
RUN pip install -e /app

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default entrypoint
ENTRYPOINT ["python3", "/app/run_docker.py"]
CMD ["--help"]

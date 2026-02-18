# All-in-one Docker image for MadGraphMLProducer
# Contains MadGraph5, Pythia8, and all processing tools

FROM scailfin/madgraph5-amc-nlo:mg5_amc3.5.1

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

# Download and install RPVMSSM_UFO model for R-parity violating SUSY
# Model from FeynRules: https://feynrules.irmp.ucl.ac.be/wiki/RPVMSSM
# The UFO model file is named af1_ufo.tgz - rename extracted dir to RPVMSSM_UFO
# Find MadGraph5 models directory (varies by image version)
RUN MG5_DIR=$(find /usr -name "MG5_aMC_v*" -type d 2>/dev/null | head -1) && \
    if [ -z "$MG5_DIR" ]; then MG5_DIR=$(find /home -name "MG5_aMC*" -type d 2>/dev/null | head -1); fi && \
    if [ -z "$MG5_DIR" ]; then MG5_DIR=$(find / -name "models" -path "*/MG5*" -type d 2>/dev/null | head -1 | sed 's|/models||'); fi && \
    echo "Found MadGraph5 at: $MG5_DIR" && \
    cd "$MG5_DIR/models" && \
    curl -kL -o af1_ufo.tgz "https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/RPVMSSM/af1_ufo.tgz" && \
    tar -xzf af1_ufo.tgz && \
    rm af1_ufo.tgz && \
    EXTRACTED_DIR=$(ls -d */ | grep -i -E "(rpv|ufo|mssm)" | head -1 | tr -d '/') && \
    if [ -n "$EXTRACTED_DIR" ] && [ "$EXTRACTED_DIR" != "RPVMSSM_UFO" ]; then \
        echo "Renaming $EXTRACTED_DIR to RPVMSSM_UFO"; \
        mv "$EXTRACTED_DIR" RPVMSSM_UFO; \
    fi && \
    ls -la && \
    echo "RPVMSSM_UFO model installed in $MG5_DIR/models" && \
    echo "Converting model to Python 3..." && \
    echo "convert model $MG5_DIR/models/RPVMSSM_UFO" | mg5_aMC && \
    echo "Model conversion complete"

# Create working directories with open permissions (for --user flag)
WORKDIR /app
RUN mkdir -p /app/output /app/work && chmod 777 /app/work /app/output

# Copy the application code
COPY src/ /app/src/
COPY cards/ /app/cards/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY run_docker.py /app/run_docker.py

# Set Python path (no pip install needed - we run directly from source)
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default entrypoint
ENTRYPOINT ["python3", "/app/run_docker.py"]
CMD ["--help"]

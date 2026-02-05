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
RUN cd /root/MG5_aMC/models && \
    curl -L -o RPVMSSM_UFO.tar.gz "https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/RPVMSSM/RPVMSSM_UFO.tar.gz" && \
    tar -xzf RPVMSSM_UFO.tar.gz && \
    rm RPVMSSM_UFO.tar.gz && \
    echo "RPVMSSM_UFO model installed"

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

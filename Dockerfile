# Base image: just plain Ubuntu
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system tools (wget, bash, bzip2 needed for conda)
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# Download and install Miniconda manually
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# -------- key part: parametrize env name and file ----------
ARG ENV_NAME=mlops-dev-project-two
ARG ENV_FILE=/app/environment.yml

# Make ENV_NAME available at runtime (entrypoint uses it)
ENV ENV_NAME=${ENV_NAME}

# -----------------------------------------------------------

# Create conda environment
COPY environment.yml /app/environment.yml

#RUN conda env create --name ${ENV_NAME} -f ${ENV_FILE}
#RUN conda env create --name "${ENV_NAME}" -f "${ENV_FILE}" \
#    && conda clean -afy

#RUN conda install -n base -y conda-libmamba-solver && \
#    conda config --system --set solver libmamba && \
#    conda config --system --set channel_priority strict && \
#    conda env create --name "${ENV_NAME}" -f "${ENV_FILE}" \
#        --override-channels -c conda-forge && \
#    conda clean -afy 

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create --name "${ENV_NAME}" -f "${ENV_FILE}" && \
    conda clean -afy

# Set conda environment for all future steps
#SHELL ["conda", "run", "-n", "${ENV_NAME}", "/bin/bash", "-c"]
SHELL ["/bin/bash", "-lc"]

# Set working directory for your project
WORKDIR /app

# Copy project files into the image
COPY . .

# Ensure /app is always importable (common/*) and sitecustomize.py auto-loads
# App paths + Python import root
ENV APP_ROOT=/app \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/model \
    TMP_DIR=/tmp \
    PYTHONPATH="/app:${PYTHONPATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Enable conda activate in interactive shells
RUN conda init bash

# Start in bash
#CMD ["bash"]

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy and permissions change can also be done with one line instead. 
# COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Use the entrypoint and login to wandb on launch
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash", "-l"]
# Start with the specified Ubuntu version
FROM ubuntu:22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set the locale to avoid potential localization issues
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python and pip, along with the necessary packages to ensure
RUN apt-get update && apt-get install -y \
        python3-pip \
        python3-dev \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-setuptools \
        ca-certificates \
        curl \
        libtool \
        vim \
        htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to its latest version
RUN python3 -m pip install --upgrade pip

# Install Python dependencies using pip
RUN pip install \
        fastapi \
        uvicorn \
        pandas \
        numpy \
        jupyterlab \
        catboost \
        shap \
        seaborn \
        matplotlib \
        scikit-learn \
        pyarrow \
        ipywidgets \
        gunicorn

# Set the working directory to /service
WORKDIR /service

# Copy the current directory contents into the container at /service
COPY . /service/

# Make ports 8888, 5000, and 8000 available to the world outside this container
EXPOSE 8888 5000 8000

# Default command to run on container start
CMD ["/bin/bash"]

# Use the official Miniconda3 image
FROM continuumio/miniconda3

# Set build argument for CPUINFO handling
ARG NO_CPUINFO=False
ENV NO_CPUINFO=${NO_CPUINFO}

# Set the working directory inside the container
WORKDIR /home/app

# Install system dependencies and required tools
RUN apt-get update -y && apt-get install -y \
    nano \
    unzip \
    curl \
    g++ \
    make \
    cmake \
    python3-dev

# Install the Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Copy the requirements.txt file and install dependencies using pip
COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the app using Gunicorn with Uvicorn workers
CMD gunicorn app:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker #

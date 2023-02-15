FROM continuumio/miniconda3

COPY . /biotmpy

WORKDIR /biotmpy

# Create the environment:
RUN conda env create -f conda_environment_lin.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "biotmpygpu", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

Expose 5000

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "biotmpygpu", "python", "web_app/app.py"]

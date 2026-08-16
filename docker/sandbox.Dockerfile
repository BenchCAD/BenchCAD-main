# Execution environment for model-written CadQuery code.
#
# Pinned to the same stack the harness scores with, so a program that runs here
# runs identically when re-executed for scoring. linux/amd64 specifically:
# nlopt 2.7.1 — the last release whose metadata allows the numpy<2 that
# cadquery 2.3.0 needs — publishes no aarch64 wheel, so an arm64 image cannot
# resolve. On Apple silicon this runs emulated, which is slower but correct.
FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        cadquery==2.3.0 cadquery-ocp==7.9.3.0 \
        numpy==1.26.4 scipy==1.17.1 trimesh==4.12.2 \
        multimethod==2.0.2 typish==1.9.3 nptyping==2.5.0 ezdxf==1.4.3 \
        nlopt==2.7.1 casadi==3.7.2 pillow vtk

# Model code runs unprivileged, with no home it can persist to.
RUN useradd -m -u 1000 runner
USER runner
WORKDIR /work

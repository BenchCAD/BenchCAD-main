# Native arm64 execution environment for model-written CadQuery code.
#
# The amd64 image runs under QEMU on Apple silicon, and the emulation cost is
# not incidental: VTK rendering plus CadQuery inside an emulated container was
# heavy enough that three concurrent records took the host down. Building
# natively removes that layer entirely.
#
# The awkward pin is nlopt. cadquery 2.3.0 needs numpy<2, and every nlopt
# release that declares numpy>=1.14 (<=2.7.1) publishes no linux aarch64 wheel.
# 2.11.0 does, so it is installed with --no-deps: its metadata asks for numpy>=2
# but it runs against 1.26 fine, which is the same accommodation the repo's own
# environment already makes.
FROM python:3.12-slim

# libosmesa6 is the one that matters: the renderer draws offscreen, and with no
# display and no GPU in the container VTK falls back to OSMesa for software
# OpenGL. Without it vtkOSOpenGLRenderWindow aborts the process (SIGSEGV, rc 139)
# rather than failing cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglx-mesa0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        cadquery==2.3.0 cadquery-ocp==7.9.3.0 \
        numpy==1.26.4 scipy==1.17.1 trimesh==4.12.2 \
        multimethod==2.0.2 typish==1.9.3 nptyping==2.5.0 ezdxf==1.4.3 \
        casadi==3.7.2 pillow vtk \
 && pip install --no-cache-dir --no-deps nlopt==2.11.0

RUN useradd -m -u 1000 runner
USER runner
WORKDIR /work

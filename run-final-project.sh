#!/bin/bash

xhost +

docker run -it --rm --name final-project-image \
-w /workspace \
-v $(pwd):/workspace \
--net=host \
-v /tmp/.x11-unix:/tmp/.x11-unix:rw \
-e DISPLAY=unix${DISPLAY} \
final-project-image
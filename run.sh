docker run -it --rm \
-p 5556:5556 -p 5557:5557  \
--mount type=bind,source="$(pwd)"/app,target=/app \
--mount type=volume,source=experiments,target=/experiments \
--entrypoint=bash \
 netlyt/behavior
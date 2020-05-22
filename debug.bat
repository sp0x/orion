@echo off
REM this will just start bash, use this script for debugging
REM for linux use --mount type=bind,source="$(pwd)"/src,target=/app
docker run -it --rm ^
-p 5560:5560 -p 5556:5556 -p 5557:5557 -p 8282:80 -m 6g  ^
-v d:/dev/asp.net/Netlyt/orion/app:/app ^
-v d:/dev/asp.net/Netlyt/orion/experiments:/experiments ^
--entrypoint=bash ^
--env-file ../orion.env ^
--network=netlyt_default ^
 orion:latest


rem --entrypoint=bash ^
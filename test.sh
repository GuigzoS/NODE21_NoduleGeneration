#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

. ./build.sh
echo "0"
mkdir $SCRIPTPATH/results
chmod 777 $SCRIPTPATH/results
docker run --rm --memory=11g --runtime nvidia -v $SCRIPTPATH/test:/input/ -v $SCRIPTPATH/results:/output/ nodulegenerator
echo "1"
docker volume create nodulegeneration-output
echo "2"
echo $SCRIPTPATH
docker run --rm \
        --runtime nvidia \
        --memory=8g \
        -v $SCRIPTPATH/test/:/input/ \
        -v nodulegeneration-output:/output/ \
        nodulegenerator
echo "3"
docker run --rm \
        -v nodulegeneration-output:/output/ \
        python:3.7-slim cat /output/results.json | python3 -m json.tool
echo "4"
docker run --rm \
        -v nodulegeneration-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.7-slim python3 -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"
echo "5"
if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm nodulegeneration-output

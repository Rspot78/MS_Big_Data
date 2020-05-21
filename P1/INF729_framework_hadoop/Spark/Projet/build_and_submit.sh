#!/bin/bash

# This script needs to be located where the build.sbt file is.
# It takes as optional parameter the path of the spark directory. If no path is specified
# it will look for the spark directory in $HOME.
#
# Example:  ./build_and_submit.sh WordCount /Users/flo/Documents/packages/spark-2.3.4-bin-hadoop2.7
#
# Paramters:
#  - $1 : job to execute
#  - $2 : [optional] path to spark directory
set -o pipefail

echo -e "\n --- building .jar --- \n"

sbt assembly || { echo 'Build failed' ; exit 1; }

echo -e "\n --- spark-submit --- \n"

path_to_spark="$HOME/spark-2.3.4-bin-hadoop2.7"

if [ -n "$2" ]; then path_to_spark=$2; fi

$path_to_spark/bin/spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 10g --class paristech.$1 target/scala-2.11/*.jar

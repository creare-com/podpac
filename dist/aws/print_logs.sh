#!/bin/sh
#
# Print cloudwatch logs to console for a user specific log group
# Defaults to showing the last 10 minutes of logs
# 
# Usage:
# 
# $ bash print_logs.sh [log-group-name] [amount of time]
# 
# [amount of time] must be a parameter than can be passed to the `date --date ` function
#   i.e. "10 minutes", "1 hour", "1 day", "1 week"
# 
# Example:
# 
# $ bash print_logs.sh /aws/lambda/podpac-s3            # show the last 10 minutes of logs
# $ bash print_logs.sh /aws/lambda/podpac-s3 "1 hour"   # show the last hour of logs
#

function dumpstreams() {
  aws $AWSARGS logs describe-log-streams --order-by LastEventTime --log-group-name "$LOGGROUP" --output text | while read -a st; do 
      [ "${st[4]}" -lt "$starttime" ] && continue
      stname="${st[1]}"
      echo ${stname##*:}
    done | while read stream; do
      aws $AWSARGS logs get-log-events --start-from-head --start-time $starttime --log-group-name "$LOGGROUP" --log-stream-name $stream --output text
    done
}

AWSARGS="--region us-east-1"
LOGGROUP="$1"
DEFAULT_DT='10 minutes'
DT=${2:-${DEFAULT_DT}}
TAIL=
starttime=$(date --date "-${DT}" +%s)000
nexttime=$(date +%s)000
dumpstreams
if [ -n "$TAIL" ]; then
  while true; do
    starttime=$nexttime
    nexttime=$(date +%s)000
    sleep 1
    dumpstreams
  done
fi

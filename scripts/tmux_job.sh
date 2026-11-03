#!/bin/bash
# bash tmux_job.sh <name> [VAR=val ...] sbatch [sbatch opts] <script.sh>
# bash tmux_job.sh --attach <name> <jobid>
# Viewer only: runs tail/squeue, never work.
set -euo pipefail
LOGDIR=/scratch/users/anirban/tamaghnam/HOI/logs
make_session() {
    local name="$1" jobid="$2" session="${1}_${2}" out err
    out=$(scontrol show job "$jobid" 2>/dev/null | sed -n 's/^ *StdOut=//p')
    err=$(scontrol show job "$jobid" 2>/dev/null | sed -n 's/^ *StdErr=//p')
    if [[ -z "$out" ]]; then out=$(ls "$LOGDIR"/*_"$jobid".out 2>/dev/null | head -1); err="${out%.out}.err"; fi
    tmux has-session -t "$session" 2>/dev/null && { echo "exists: $session"; return; }
    tmux new-session -d -s "$session" -n log
    tmux set-option -t "$session" history-limit 200000
    tmux send-keys -t "$session:log" "tail -n +1 -F '$out'" Enter
    tmux new-window -t "$session" -n err
    tmux send-keys -t "$session:err" "tail -n +1 -F '$err'" Enter
    tmux new-window -t "$session" -n status
    tmux send-keys -t "$session:status" "while squeue -h -j $jobid | grep -q .; do clear; date; squeue -j $jobid; sleep 60; done; sleep 30; tmux send-keys -t ${session}:log C-c; tmux send-keys -t ${session}:err C-c; clear; sacct -j $jobid -X --format=JobID,JobName%24,State,Elapsed,NodeList" Enter
    tmux select-window -t "$session:log"
    echo "job $jobid -> tmux a -t $session"
}
if [[ "${1:-}" == "--attach" ]]; then make_session "$2" "$3"; exit 0; fi
name="$1"; shift; env_vars=()
while [[ $# -gt 0 && "$1" == *=* ]]; do env_vars+=("$1"); shift; done
[[ "${1:-}" == "sbatch" ]] && shift
jobid=$(env "${env_vars[@]}" sbatch --parsable "$@"); jobid="${jobid%%;*}"
echo "submitted $jobid"; make_session "$name" "$jobid"

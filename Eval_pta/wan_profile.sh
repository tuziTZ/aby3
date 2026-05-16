# compile and synchronize
python build.py
scp ./bin/frontend aby31:~/aby3/bin &
scp ./bin/frontend aby32:~/aby3/bin &
wait;

# simulate WAN network
bandwidth=100
latency=50
./Eval/network/network_set.sh $bandwidth $latency;
wait;

# start the deployment profiling
./Eval/deploy_profile.sh
wait;

# start the task profiling.
./Eval/pta_profiler.sh
wait;

./Eval/network/network_clean.sh;
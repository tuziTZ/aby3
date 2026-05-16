bandwidth=$1; latency=$2;

echo "bandwidth = "${bandwidth}
echo "latency = "${latency}

latency=$((latency / 2))
# bandwidth=$((bandwidth / 2))

echo "bandwidth = "${bandwidth}
echo "latency = "${latency}

# setup network!
tc qdisc add dev eth0 root handle 1: tbf rate ${bandwidth}mbit burst ${bandwidth}mbit latency ${latency}ms;
tc qdisc add dev eth0 parent 1:1 handle 10: netem delay ${latency}ms;

ssh aby31 "tc qdisc add dev eth0 root handle 1: tbf rate "${bandwidth}"mbit burst "${bandwidth}"mbit latency "${latency}"ms; tc qdisc add dev eth0 parent 1:1 handle 10: netem delay "${latency}"ms;"

ssh aby32 "tc qdisc add dev eth0 root handle 1: tbf rate "${bandwidth}"mbit burst "${bandwidth}"mbit latency "${latency}"ms; tc qdisc add dev eth0 parent 1:1 handle 10: netem delay "${latency}"ms;"
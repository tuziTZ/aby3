# # cleanup the settings on the eth
tc qdisc del dev eth0 root; tc qdisc del dev eth0 ingress;
ssh aby31 "tc qdisc del dev eth0 root; tc qdisc del dev eth0 ingress";
ssh aby32 "tc qdisc del dev eth0 root; tc qdisc del dev eth0 ingress";
ENV=ARG1
AGENT=ARG2


set term pngcairo size 1280,1280 linewidth 3 fontscale 1.5;
set output "figures/".ENV.".".AGENT.".png";
set grid;
set key box bottom right;

set xlabel "Steps";
set ylabel "Average reward";
set title ((ENV eq "bandit") ? "Multi-armed bandit" : (ENV eq "newcomb") ? "Newcomb's problem" : (ENV eq "pdbandit") ? "Policy-dependent bandit" : ENV).", ".AGENT." agent";
set yrange [0:];
set multiplot layout 2,1;
plot "outputs/".ENV."/".AGENT."_0.01.txt" u 1:2 w l title "ε-greedy, ε=1%", \
     "outputs/".ENV."/".AGENT."_0.10.txt" u 1:2 w l title "ε-greedy, ε=10%", \
     "outputs/".ENV."/".AGENT."_0.00.txt" u 1:2 w l title "greedy";

unset key;
set ylabel "Optimal action rate";
unset title;
set yrange [0:1];
plot "outputs/".ENV."/".AGENT."_0.01.txt" u 1:3 w l, \
     "outputs/".ENV."/".AGENT."_0.10.txt" u 1:3 w l, \
     "outputs/".ENV."/".AGENT."_0.00.txt" u 1:3 w l;
unset multiplot;


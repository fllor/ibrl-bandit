ENV=ARG1
AGENT=ARG2

set style fill transparent solid 0.5

set term pngcairo size 1024,512 linewidth 3 fontscale 1.5;
set output "figures/".ENV.".".AGENT.".png";
set grid;
set key box bottom center opaque;

set xlabel "Steps";
set ylabel "Average reward";
set title ((ENV eq "bandit")                 ? "Multi-armed bandit" : \
           (ENV eq "newcomb")                ? "Newcomb's problem" : \
           (ENV eq "damascus")               ? "Death in Damascus" : \
           (ENV eq "asymmetric-damascus")    ? "Asymmetric Death in Damascus" : \
           (ENV eq "coordination")           ? "Coordination game" : \
           (ENV eq "pdbandit")               ? "Policy-dependent bandit" : \
           ENV).", ". \
          ((AGENT eq "classical")            ? "Q-learning agent" : \
           (AGENT eq "bayesian")             ? "Bayesian agent" : \
           (AGENT eq "experimental1")        ? "Experimental agent 1" : \
           (AGENT eq "experimental2")        ? "Experimental agent 2" : \
           AGENT);
set yrange [0:];
if(AGENT eq "experimental2") {
     plot "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:2 w l ls 1 title "Optimal policy", \
          "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:($3+$4):($3-$4) w filledcurves ls 2 notitle, \
          "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:3 w l ls 2 title "ε-greedy policy";
} else {
     plot "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:2 w l ls 1 title "Optimal policy", \
          "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:($3+$4):($3-$4) w filledcurves ls 2 notitle, \
          "outputs/".ENV.".".AGENT.".epsilon.txt" u 1:3 w l ls 2 title "ε-greedy policy", \
          "outputs/".ENV.".".AGENT.".softmax.txt" u 1:($3+$4):($3-$4) w filledcurves ls 3 notitle, \
          "outputs/".ENV.".".AGENT.".softmax.txt" u 1:3 w l ls 3 title "Softmax policy";
}

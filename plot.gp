ENV=ARG1
AGENT=ARG2

set style fill transparent solid 0.5

set term pngcairo size 1280,1280 linewidth 3 fontscale 1.5;
set output "figures/".ENV.".".AGENT.".png";
set grid;
set key box bottom right;

set xlabel "Steps";
set ylabel "Average reward";
set title ((ENV eq "bandit") ? "Multi-armed bandit" : (ENV eq "newcomb") ? "Newcomb's problem" : ENV).", ". \
          ((AGENT eq "classical") ? "Q-learning" : "Infrabayesian")." agent";
set yrange [0:];
set multiplot layout 2,1;
plot "outputs/".ENV."/".AGENT.".txt" u 1:($2+$3):($2-$3) w filledcurves ls 1 notitle, \
     "outputs/".ENV."/".AGENT.".txt" u 1:2 w l ls 1 notitle;

unset key;
set ylabel "Optimal action rate";
unset title;
set yrange [0:1];
plot "outputs/".ENV."/".AGENT.".txt" u 1:($5+$6):($5-$6) w filledcurves ls 1, \
     "outputs/".ENV."/".AGENT.".txt" u 1:5 w l ls 1;
unset multiplot;


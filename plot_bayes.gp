set term pngcairo size 1280,1280 linewidth 3 fontscale 1.5;
set output "bayes.png";
set grid;
set key box bottom right;

set xlabel "Steps";
set ylabel "Average reward";
set title "Bayesian agent";
set yrange [0:1.6];
set multiplot layout 2,1;
plot "outputs/bayesian_0.01.txt" u 1:2 w l title "ε-greedy, 0.01", \
     "outputs/bayesian_0.10.txt" u 1:2 w l title "ε-greedy, 0.1", \
     "outputs/bayesian_0.00.txt" u 1:2 w l title "greedy";

unset key;
set ylabel "Optimal action rate";
unset title;
set yrange [0:1];
plot "outputs/bayesian_0.01.txt" u 1:3 w l, \
     "outputs/bayesian_0.10.txt" u 1:3 w l, \
     "outputs/bayesian_0.00.txt" u 1:3 w l;
unset multiplot;


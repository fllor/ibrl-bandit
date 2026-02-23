OPTIONS=--arms 10 --steps 1000 --runs 2000
OUTPUTS_CLASSICAL=outputs/classical_0.01.txt outputs/classical_0.10.txt outputs/classical_0.00.txt
OUTPUTS_BAYESIAN=outputs/bayesian_0.01.txt outputs/bayesian_0.10.txt outputs/bayesian_0.00.txt

all: classical.png bayes.png
.PHONY: all

outputs:
	mkdir -p outputs

$(OUTPUTS_CLASSICAL): outputs/classical_%.txt: main.py | outputs
	python3 -m main classical $(OPTIONS) --epsilon $* > $@
$(OUTPUTS_BAYESIAN): outputs/bayesian_%.txt: main.py | outputs
	python3 -m main bayes $(OPTIONS) --epsilon $* > $@

# Figure 2.2 from Barto&Sutton
classical.png: plot_classical.gp $(OUTPUTS_CLASSICAL)
	gnuplot $<
bayes.png: plot_bayes.gp $(OUTPUTS_BAYESIAN)
	gnuplot $<

.PHONY: clean
clean:
	rm -rf outputs classical.png bayes.png


OPTIONS_BANDIT=--arms 10 --steps 1001 --runs 2000
OUTPUTS_BANDIT_CLASSICAL=outputs/bandit/classical_0.01.txt outputs/bandit/classical_0.10.txt outputs/bandit/classical_0.00.txt
OUTPUTS_BANDIT_BAYESIAN=outputs/bandit/bayesian_0.01.txt outputs/bandit/bayesian_0.10.txt outputs/bandit/bayesian_0.00.txt
OPTIONS_NEWCOMB=--arms 2 --steps 201 --runs 2000
OUTPUTS_NEWCOMB_CLASSICAL=outputs/newcomb/classical_0.01.txt outputs/newcomb/classical_0.10.txt outputs/newcomb/classical_0.00.txt
OUTPUTS_NEWCOMB_BAYESIAN=outputs/newcomb/bayesian_0.01.txt outputs/newcomb/bayesian_0.10.txt outputs/newcomb/bayesian_0.00.txt
IMAGES=bandit.classical.png bandit.bayesian.png newcomb.classical.png newcomb.bayesian.png

all: $(IMAGES)
.PHONY: all

outputs/bandit outputs/newcomb: %:
	mkdir -p $@

$(OUTPUTS_BANDIT_CLASSICAL): outputs/bandit/classical_%.txt: main.py | outputs/bandit
	python3 -m main bandit classical $(OPTIONS_BANDIT) --epsilon $* > $@
$(OUTPUTS_BANDIT_BAYESIAN): outputs/bandit/bayesian_%.txt: main.py | outputs/bandit
	python3 -m main bandit bayes $(OPTIONS_BANDIT) --epsilon $* > $@
$(OUTPUTS_NEWCOMB_CLASSICAL): outputs/newcomb/classical_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb classical $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_NEWCOMB_BAYESIAN): outputs/newcomb/bayesian_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb bayes $(OPTIONS_NEWCOMB) --epsilon $* > $@

# Figure 2.2 from Barto&Sutton
bandit.classical.png: plot.gp $(OUTPUTS_BANDIT_CLASSICAL)
	gnuplot -c $< bandit classical
bandit.bayesian.png: plot.gp $(OUTPUTS_BANDIT_BAYESIAN)
	gnuplot -c $< bandit bayesian
newcomb.classical.png: plot.gp $(OUTPUTS_NEWCOMB_CLASSICAL)
	gnuplot -c $< newcomb classical
 newcomb.bayesian.png: plot.gp $(OUTPUTS_NEWCOMB_BAYESIAN)
	gnuplot -c $< newcomb bayesian

.PHONY: clean
clean:
	rm -rf outputs $(IMAGES)


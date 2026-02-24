EPSILON=0.01 0.10 0.00
OPTIONS_BANDIT=--arms 10 --steps 1001 --runs 2000
OUTPUTS_BANDIT_CLASSICAL=$(EPSILON:%=outputs/bandit/classical_%.txt)
OUTPUTS_BANDIT_BAYESIAN=$(EPSILON:%=outputs/bandit/bayesian_%.txt)
OUTPUTS_BANDIT_INFRABAYESIAN=$(EPSILON:%=outputs/bandit/infrabayesian_%.txt)
OPTIONS_NEWCOMB=--arms 2 --steps 1001 --runs 2000
OUTPUTS_NEWCOMB_CLASSICAL=$(EPSILON:%=outputs/newcomb/classical_%.txt)
OUTPUTS_NEWCOMB_BAYESIAN=$(EPSILON:%=outputs/newcomb/bayesian_%.txt)
OUTPUTS_NEWCOMB_INFRABAYESIAN=$(EPSILON:%=outputs/newcomb/infrabayesian_%.txt)
OUTPUTS_PDBANDIT_CLASSICAL=$(EPSILON:%=outputs/pdbandit/classical_%.txt)
OUTPUTS_PDBANDIT_BAYESIAN=$(EPSILON:%=outputs/pdbandit/bayesian_%.txt)
OUTPUTS_PDBANDIT_INFRABAYESIAN=$(EPSILON:%=outputs/pdbandit/infrabayesian_%.txt)
IMAGES=$(foreach AGENT,classical bayesian infrabayesian,$(foreach ENV,bandit newcomb pdbandit,$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs/bandit outputs/newcomb outputs/pdbandit: %:
	mkdir -p $@

$(OUTPUTS_BANDIT_CLASSICAL): outputs/bandit/classical_%.txt: main.py | outputs/bandit
	python3 -m main bandit classical $(OPTIONS_BANDIT) --epsilon $* > $@
$(OUTPUTS_BANDIT_BAYESIAN): outputs/bandit/bayesian_%.txt: main.py | outputs/bandit
	python3 -m main bandit bayesian $(OPTIONS_BANDIT) --epsilon $* > $@
$(OUTPUTS_BANDIT_INFRABAYESIAN): outputs/bandit/infrabayesian_%.txt: main.py | outputs/bandit
	python3 -m main bandit infrabayesian $(OPTIONS_BANDIT) --epsilon $* --optimism 4 > $@
$(OUTPUTS_NEWCOMB_CLASSICAL): outputs/newcomb/classical_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb classical $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_NEWCOMB_BAYESIAN): outputs/newcomb/bayesian_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb bayesian $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_NEWCOMB_INFRABAYESIAN): outputs/newcomb/infrabayesian_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb infrabayesian $(OPTIONS_NEWCOMB) --epsilon $* --optimism 101 > $@
$(OUTPUTS_PDBANDIT_CLASSICAL): outputs/pdbandit/classical_%.txt: main.py | outputs/pdbandit
	python3 -m main pdbandit classical $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_PDBANDIT_BAYESIAN): outputs/pdbandit/bayesian_%.txt: main.py | outputs/pdbandit
	python3 -m main pdbandit bayesian $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_PDBANDIT_INFRABAYESIAN): outputs/pdbandit/infrabayesian_%.txt: main.py | outputs/pdbandit
	python3 -m main pdbandit infrabayesian $(OPTIONS_NEWCOMB) --epsilon $* --optimism 4 > $@

# Figure 2.2 from Barto&Sutton
bandit.classical.png: plot.gp $(OUTPUTS_BANDIT_CLASSICAL)
	gnuplot -c $< bandit classical
bandit.bayesian.png: plot.gp $(OUTPUTS_BANDIT_BAYESIAN)
	gnuplot -c $< bandit bayesian
bandit.infrabayesian.png: plot.gp $(OUTPUTS_BANDIT_INFRABAYESIAN)
	gnuplot -c $< bandit infrabayesian
newcomb.classical.png: plot.gp $(OUTPUTS_NEWCOMB_CLASSICAL)
	gnuplot -c $< newcomb classical
newcomb.bayesian.png: plot.gp $(OUTPUTS_NEWCOMB_BAYESIAN)
	gnuplot -c $< newcomb bayesian
newcomb.infrabayesian.png: plot.gp $(OUTPUTS_NEWCOMB_INFRABAYESIAN)
	gnuplot -c $< newcomb infrabayesian
pdbandit.classical.png: plot.gp $(OUTPUTS_PDBANDIT_CLASSICAL)
	gnuplot -c $< pdbandit classical
pdbandit.bayesian.png: plot.gp $(OUTPUTS_PDBANDIT_BAYESIAN)
	gnuplot -c $< pdbandit bayesian
pdbandit.infrabayesian.png: plot.gp $(OUTPUTS_PDBANDIT_INFRABAYESIAN)
	gnuplot -c $< pdbandit infrabayesian

.PHONY: clean
clean:
	rm -rf outputs $(IMAGES)


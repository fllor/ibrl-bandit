EPSILON=0.01 0.10 0.00
OPTIONS_BANDIT=--arms 10 --steps 1001 --runs 2000
OUTPUTS_BANDIT_CLASSICAL=$(EPSILON:%=outputs/bandit/classical_%.txt)
OPTIONS_NEWCOMB=--arms 2 --steps 1001 --runs 2000
OUTPUTS_NEWCOMB_CLASSICAL=$(EPSILON:%=outputs/newcomb/classical_%.txt)
IMAGES=$(foreach AGENT,classical,$(foreach ENV,bandit newcomb,figures/$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs/bandit outputs/newcomb outputs/pdbandit figures: %:
	mkdir -p $@

$(OUTPUTS_BANDIT_CLASSICAL): outputs/bandit/classical_%.txt: main.py | outputs/bandit
	python3 -m main bandit q $(OPTIONS_BANDIT) --epsilon $* > $@
$(OUTPUTS_NEWCOMB_CLASSICAL): outputs/newcomb/classical_%.txt: main.py | outputs/newcomb
	python3 -m main newcomb q $(OPTIONS_NEWCOMB) --epsilon $* > $@
$(OUTPUTS_PDBANDIT_CLASSICAL): outputs/pdbandit/classical_%.txt: main.py | outputs/pdbandit
	python3 -m main pdbandit q $(OPTIONS_NEWCOMB) --epsilon $* > $@

# Figure 2.2 from Barto&Sutton
figures/bandit.classical.png: plot.gp $(OUTPUTS_BANDIT_CLASSICAL) | figures
	gnuplot -c $< bandit classical
figures/newcomb.classical.png: plot.gp $(OUTPUTS_NEWCOMB_CLASSICAL) | figures
	gnuplot -c $< newcomb classical

.PHONY: clean
clean:
	rm -rf outputs figures


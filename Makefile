OPTIONS_BANDIT=--arms 10 --steps 1001 --runs 2000
OPTIONS_NEWCOMB=--arms 2 --steps 1001 --runs 2000
IMAGES=$(foreach AGENT,classical,$(foreach ENV,bandit newcomb,figures/$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs/bandit outputs/newcomb figures: %:
	mkdir -p $@

outputs/bandit/classical.txt: main.py | outputs/bandit
	python3 -m main bandit q $(OPTIONS_BANDIT) > $@
outputs/newcomb/classical.txt: main.py | outputs/newcomb
	python3 -m main newcomb q $(OPTIONS_NEWCOMB) > $@

figures/bandit.classical.png: plot.gp outputs/bandit/classical.txt | figures
	gnuplot -c $< bandit classical
figures/newcomb.classical.png: plot.gp outputs/newcomb/classical.txt | figures
	gnuplot -c $< newcomb classical

.PHONY: clean
clean:
	rm -rf outputs figures


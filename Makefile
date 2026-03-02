ENVS=bandit newcomb damascus
OPTIONS_BASE=--arms 2 --steps 1001 --runs 2000
OPTIONS_bandit  =$(OPTIONS_BASE) --arms 10
OPTIONS_newcomb =$(OPTIONS_BASE)
OPTIONS_damascus=$(OPTIONS_BASE)
IMAGES=$(foreach AGENT,classical,$(foreach ENV,$(ENVS),figures/$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs figures: %:
	mkdir -p $@

$(ENVS:%=outputs/%.classical.txt): outputs/%.classical.txt: main.py | outputs
	python3 -m main $* q $(OPTIONS_$*) > $@

$(ENVS:%=figures/%.classical.png): figures/%.classical.png: plot.gp outputs/%.classical.txt | figures
	gnuplot -c $< $* classical

.PHONY: clean
clean:
	rm -rf outputs figures


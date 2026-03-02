ENVS=bandit newcomb damascus
OPTIONS_BASE=--steps 1001 --runs 2000
OPTIONS_bandit  =$(OPTIONS_BASE) --arms 10
OPTIONS_newcomb =$(OPTIONS_BASE) --arms 2
OPTIONS_damascus=$(OPTIONS_BASE) --arms 2
IMAGES=$(foreach AGENT,classical,$(foreach ENV,$(ENVS),figures/$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs figures: %:
	mkdir -p $@

$(ENVS:%=outputs/%.classical.epsilon.txt): outputs/%.classical.epsilon.txt: main.py | outputs
	python3 -m main $* q $(OPTIONS_$*) --policy epsilon-greedy > $@
$(ENVS:%=outputs/%.classical.softmax.txt): outputs/%.classical.softmax.txt: main.py | outputs
	python3 -m main $* q $(OPTIONS_$*) --policy softmax > $@

$(ENVS:%=figures/%.classical.png): figures/%.classical.png: plot.gp outputs/%.classical.epsilon.txt outputs/%.classical.softmax.txt | figures
	gnuplot -c $< $* classical

.PHONY: clean
clean:
	rm -rf outputs figures


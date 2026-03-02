ENVS=bandit newcomb damascus asymmetric-damascus coordination
OPTIONS_BASE=--steps 1001 --runs 2000
OPTIONS_bandit= --arms 10
IMAGES=$(foreach AGENT,classical,$(foreach ENV,$(ENVS),figures/$(ENV).$(AGENT).png))

all: $(IMAGES)
.PHONY: all

outputs figures: %:
	mkdir -p $@

$(ENVS:%=outputs/%.classical.epsilon.txt): outputs/%.classical.epsilon.txt: main.py | outputs
	python3 -m main $* q $(OPTIONS_BASE) $(OPTIONS_$*) --policy epsilon-greedy > $@
$(ENVS:%=outputs/%.classical.softmax.txt): outputs/%.classical.softmax.txt: main.py | outputs
	python3 -m main $* q $(OPTIONS_BASE) $(OPTIONS_$*) --policy softmax > $@

$(ENVS:%=figures/%.classical.png): figures/%.classical.png: plot.gp outputs/%.classical.epsilon.txt outputs/%.classical.softmax.txt | figures
	gnuplot -c $< $* classical

.PHONY: clean
clean:
	rm -rf outputs figures


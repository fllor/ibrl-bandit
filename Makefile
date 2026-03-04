ENVS=bandit newcomb damascus asymmetric-damascus coordination
AGENTS=classical experimental1 experimental2
POLICIES=epsilon softmax
OPTIONS_BASE=--steps 1001 --runs 2000
OPTIONS_bandit=--arms 10
ALL=$(foreach env,$(ENVS),$(foreach agent,$(AGENTS),figures/$(env).$(agent).png))
ALL:=$(filter-out figures/bandit.experimental2.png,$(ALL))

all: $(ALL)
.PHONY: all

outputs figures: %:
	mkdir -p $@

# Perform simulation. One target per (environment,agent,policy)
define CREATE_RUN_TARGET # env agent policy
outputs/$1.$2.$3.txt: main.py | outputs
	python3 -m main $1 $2 --policy $3 $$(OPTIONS_BASE) $$(OPTIONS_$1) $$(OPTIONS_$2) > $$@
endef
$(foreach env,$(ENVS),$(foreach agent,$(AGENTS),$(foreach policy,$(POLICIES),$(eval $(call CREATE_RUN_TARGET,$(env),$(agent),$(policy))))))

# Create plot. One target per (environment,agent)
define CREATE_PLOT_TARGET # env agent
figures/$1.$2.png: plot.gp outputs/$1.$2.epsilon.txt outputs/$1.$2.softmax.txt | figures
	gnuplot -c $$< $1 $2
endef
$(foreach env,$(ENVS),$(foreach agent,$(AGENTS),$(eval $(call CREATE_PLOT_TARGET,$(env),$(agent)))))

# Shortcuts to create all plots involving a specific environment or agent
.PHONY: $(ENVS) $(AGENTS)
$(filter-out experimental2,$(AGENTS)): %: $(foreach env,$(ENVS),figures/$(env).%.png)
experimental2: %: $(foreach env,$(filter-out bandit,$(ENVS)),figures/$(env).%.png)
$(filter-out bandit,$(ENVS)): %: $(foreach agent,$(AGENTS),figures/%.$(agent).png)
bandit: %: $(foreach agent,$(filter-out experimental2,$(AGENTS)),figures/%.$(agent).png)

.PHONY: clean
clean:
	rm -rf outputs figures


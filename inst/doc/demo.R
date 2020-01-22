## ----setup, include = FALSE---------------------------------------------------
library("parallelMap")
library("ParamHelpers")
library("mlr")
library("mlrCPO")

library("ecr")
library("mosmafs")

library("magrittr")
library("ggplot2")


set.seed(8008135)

options(width = 80)

cores <- parallel::detectCores()
if (Sys.getenv("FASTVIGNETTE") != "true") {
  cores <- min(cores, 2)
}
if (.Platform$OS.type == "windows") {
  parallelStartSocket(cores, show.info = FALSE)
} else {
  parallelStartMulticore(cores, show.info = FALSE, mc.set.seed = FALSE)
}

print.list <- function(x) {
  if (all(vlapply(x, is.atomic))) {
    x <- sapply(x, function(x) if (is.numeric(x)) round(x, 3) else x)
    catf("list(%s)",
      collapse(sprintf("%s = %s", names(x),
        vcapply(x, deparse, width.cutoff = 500, nlines = 1)), ", "))
  } else {
    NextMethod(x)
  }
}

# mock runtime output
getStatistics <- function(log) {
  stats <- ecr::getStatistics(log)
  if ("runtime.mean" %in% colnames(stats)) {
    if ("fidelity.sum" %in% colnames(stats)) {
      fid <- stats$fidelity.sum / stats$size
    } else {
      fid <- rep(5, nrow(stats))
    }
    stats$runtime.mean <- rnorm(nrow(stats), 0.034, 0.002) * (fid + 2)
    stats$runtime.sum <- stats$runtime.mean * stats$size
  }
  stats
}

getPopulations <- function(log) {
  pop <- ecr::getPopulations(log)
  for (i in seq_along(pop)) {
    pop[[i]]$population <- lapply(pop[[i]]$population, function(ind) {
      fid <- attr(ind, "fidelity")
      if (is.null(fid)) {
        fid <- 5
      }
      attr(ind, "runtime")[1] <- rnorm(1, 0.034, 0.002) * (fid + 2)
      ind
    })
  }
  pop
}

knitr::opts_chunk$set(
  cache = FALSE,
  collapse = TRUE,
  comment = "#>"
)

## ---- eval = FALSE------------------------------------------------------------
#  devtools::install_github("jakobbossek/ecr2")
#  library("ecr")
#  library("magrittr")
#  library("ggplot2")
#  library("ParamHelpers")
#  library("mlr")
#  library("mlrCPO")
#  library("mosmafs")

## -----------------------------------------------------------------------------
mutFlipflop <- makeMutator(
  mutator = function(ind, ...) {
    1 - ind
  }, supported = "binary")

mutFlipflop(c(1, 0, 1))

## -----------------------------------------------------------------------------
mutCycle <- makeMutator(
  mutator = function(ind, values) {
    index <- mapply(match, ind, values)
    index <- index %% viapply(values, length) + 1
    mapply(`[[`, values, index)
  }, supported = "custom")

mutCycle(c("a", "x", "z"), values = list(letters))

## -----------------------------------------------------------------------------
mutPlus <- makeMutator(
  mutator = function(ind, summand = 1, ...) {
    ind + summand
  }, supported = "custom")

mutPlus(c(1, 2, 3))

## -----------------------------------------------------------------------------
mutReverse <- makeMutator(
  mutator = function(ind, ...) {
    rev(ind)
  }, supported = "custom")

mutReverse(c(1, 2, 3))

## -----------------------------------------------------------------------------
task.whole <- create.hypersphere.data(3, 2000) %>%
  create.classif.task(id = "sphere") %>%
  task.add.permuted.cols(10)
rows.whole <- sample(2000)
task <- subsetTask(task.whole, rows.whole[1:500])
task.hout <- subsetTask(task.whole, rows.whole[501:2000])

## -----------------------------------------------------------------------------
ps <- pSS(
  bin1: logical,
  bin2: discrete[no, yes],
  disc1: discrete[letters]^3,
  num1: numeric[0, 10])

## -----------------------------------------------------------------------------
combo <- combine.operators(ps,
  bin1 = mutFlipflop,
  bin2 = mutFlipflop,
  disc1 = mutCycle,
  num1 = mutPlus)

## -----------------------------------------------------------------------------
combo2 <- combine.operators(ps,
  bin1 = mutFlipflop,
  bin2 = mutCycle,
  disc1 = mutCycle,
  num1 = mutPlus,
  .binary.discrete.as.logical = FALSE)

## -----------------------------------------------------------------------------
combo(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

combo2(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

## -----------------------------------------------------------------------------
combo.group <- combine.operators(ps,
  .params.group1 = c("bin1", "bin2"),
  group1 = mutFlipflop,
  discrete = mutCycle,
  num1 = mutPlus)

## -----------------------------------------------------------------------------
combo.group(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

## -----------------------------------------------------------------------------
combo.rev.indiv <- combine.operators(ps,
  bin1 = mutReverse,
  bin2 = mutReverse,
  discrete = mutCycle,
  num1 = mutPlus)

combo.rev.group <- combine.operators(ps,
  .params.group1 = c("bin1", "bin2"),
  group1 = mutReverse,
  discrete = mutCycle,
  num1 = mutPlus)

## -----------------------------------------------------------------------------
combo.rev.indiv(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

## -----------------------------------------------------------------------------
combo.rev.group(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

## -----------------------------------------------------------------------------
combo.strategy <- combine.operators(ps,
  logical = mutFlipflop,
  discrete = mutCycle,
  numeric = mutPlus,
  .strategy.numeric = function(ind) {
    if (ind$bin2 == "yes") {
      return(list(summand = 1))
    } else {
      return(list(summand = -1))
    }
  })

## -----------------------------------------------------------------------------
combo.strategy(list(bin1 = TRUE, bin2 = "yes", disc1 = c("a", "x", "z"), num1 = 3))

combo.strategy(list(bin1 = TRUE, bin2 = "no", disc1 = c("a", "x", "z"), num1 = 3))

## -----------------------------------------------------------------------------

value <- sampleValue(ps, discrete.names  = TRUE)

print(value)

combo(value)

## -----------------------------------------------------------------------------
lrn <- makeLearner("classif.rpart", maxsurrogate = 0)

## -----------------------------------------------------------------------------
ps.simple <- pSS(
  maxdepth: integer[1, 30],
  minsplit: integer[2, 30],
  cp: numeric[0.001, 0.999])

## -----------------------------------------------------------------------------
fitness.fun.simple <- makeObjective(lrn, task, ps.simple, cv5,
  holdout.data = task.hout)

## -----------------------------------------------------------------------------
ps.objective <- getParamSet(fitness.fun.simple)

## -----------------------------------------------------------------------------
mutator.simple <- combine.operators(ps.objective,
  numeric = ecr::setup(mutGauss, sdev = 0.1),
  integer = ecr::setup(mutGaussInt, sdev = 3),
  selector.selection = mutBitflipCHW)

crossover.simple <- combine.operators(ps.objective,
  numeric = recPCrossover,
  integer = recPCrossover,
  selector.selection = recPCrossover)

## -----------------------------------------------------------------------------
initials.simple <- sampleValues(ps.objective , 32, discrete.names = TRUE)

## ---- include = FALSE---------------------------------------------------------
set.seed(1)

## -----------------------------------------------------------------------------
run.simple <- slickEcr(
  fitness.fun = fitness.fun.simple,
  lambda = 16,
  population = initials.simple,
  mutator = mutator.simple,
  recombinator = crossover.simple,
  generations = 10)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts <- function(run) {
  fronts <- fitnesses(run, function(x) paretoEdges(x, c(1, 1)))
  ggplot(data = fronts, aes(x = perf, y = propfeat, color = ordered(gen))) +
    geom_line() +
    geom_point(data = fronts[fronts$point, ], shape = "x", size = 5) +
    xlim(0, 1) +
    ylim(0, 1) +
    coord_fixed()
}

plot_fronts(run.simple)

## -----------------------------------------------------------------------------
populations <- getPopulations(run.simple$log)
pop.gen1 <- populations[[1]]

individual.1 <- pop.gen1$population[[1]]
attr(individual.1, "runtime")

individual.2 <- pop.gen1$population[[2]]
attr(individual.2, "runtime")

getStatistics(run.simple$log.newinds)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
ggplot(collectResult(run.simple), aes(x = evals)) +
  geom_line(aes(y = hout.perf.mean, color = "Holdout")) +
  geom_line(aes(y = eval.perf.mean, color = "Training")) +
  ylim(0, 1)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
ggplot(collectResult(run.simple), aes(x = evals)) +
  geom_line(aes(y = true.hout.domHV, color = "Holdout")) +
  geom_line(aes(y = eval.domHV, color = "Training")) +
  ylim(0, 1)

## -----------------------------------------------------------------------------
colnames(collectResult(run.simple))

## -----------------------------------------------------------------------------
filters <- c("praznik_JMI", "auc", "anova.test", "variance", "DUMMY")
fima <- makeFilterMat(task, filters = filters)

## -----------------------------------------------------------------------------
ps.strat <- pSS(
  maxdepth: integer[1, 30],
  minsplit: integer[2, 30],
  cp: numeric[0.001, 0.999],
  filterweights: numeric[0.001, 0.999]^length(filters))

## -----------------------------------------------------------------------------
fitness.fun.strat <- makeObjective(lrn, task, ps.strat, cv5)
ps.strat.obj <- getParamSet(fitness.fun.strat)

## -----------------------------------------------------------------------------
mutator.strat <- combine.operators(ps.strat.obj,
  numeric = ecr::setup(mutGauss, sdev = 0.1),
  integer = ecr::setup(mutGaussInt, sdev = 3),
  selector.selection = mutUniformMetaResetSHW,
  .strategy.selector.selection = makeFilterStrategy(
    reset.dists = fima, weight.param.name = "filterweights"))

## -----------------------------------------------------------------------------
crossover.strat <- combine.operators(ps.strat.obj,
  numeric = recPCrossover,
  integer = recPCrossover,
  selector.selection = recPCrossover)

## -----------------------------------------------------------------------------
initials.strat <- sampleValues(ps.strat.obj, 32, discrete.names = TRUE) %>%
  initSelector(
    soften.op = ecr::setup(mutUniformMetaResetSHW, p = 1),
    soften.op.strategy = makeFilterStrategy(fima, "filterweights"))

## ---- include = FALSE---------------------------------------------------------
set.seed(2)

## -----------------------------------------------------------------------------
run.strat <- slickEcr(
  fitness.fun = fitness.fun.strat,
  lambda = 16,
  population = initials.strat,
  mutator = mutator.strat,
  recombinator = crossover.strat,
  generations = 10)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts(run.strat)

## ---- include = FALSE---------------------------------------------------------
set.seed(2)

## -----------------------------------------------------------------------------
run.strat.20 <- continueEcr(run.strat, 10, lambda = 8)

## -----------------------------------------------------------------------------
getStatistics(run.strat.20$log.newinds)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts(run.strat.20)


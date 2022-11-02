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
  parallelLibrary("mlr")
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

slickEcr <- function(fidelity = NULL, ...) {
  if (!is.null(fidelity)) {
    fidelity[[2]] = fidelity[[2]] * 0 + seq_along(fidelity[[2]])
    if (length(fidelity) > 2) {
      fidelity[[3]] = fidelity[[3]] * 0 + 1
    }
  }
  mosmafs::slickEcr(fidelity = fidelity, ...)
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
task.whole <- create.hypersphere.data(3, 2000) %>%
  create.classif.task(id = "sphere") %>%
  task.add.permuted.cols(10)
rows.whole <- sample(2000)
task <- subsetTask(task.whole, rows.whole[1:500])
task.hout <- subsetTask(task.whole, rows.whole[501:2000])

lrn <- makeLearner("classif.rpart", maxsurrogate = 0)

ps.simple <- pSS(
  maxdepth: integer[1, 30],
  minsplit: integer[2, 30],
  cp: numeric[0.001, 0.999])

fitness.fun.simple <- makeObjective(lrn, task, ps.simple, cv5,
  holdout.data = task.hout)

ps.objective <- getParamSet(fitness.fun.simple)

mutator.simple <- combine.operators(ps.objective,
  numeric = ecr::setup(mutGauss, sdev = 0.1),
  integer = ecr::setup(mutGaussInt, sdev = 3),
  selector.selection = mutBitflipCHW)

crossover.simple <- combine.operators(ps.objective,
  numeric = recPCrossover,
  integer = recPCrossover,
  selector.selection = recPCrossover)

initials <- sampleValues(ps.objective, 32, discrete.names = TRUE)

## -----------------------------------------------------------------------------
nRes <- function(n) {
  makeResampleDesc("Subsample", split = 0.9, iters = n)
}

## -----------------------------------------------------------------------------
fitness.fun <- makeObjective(lrn, task, ps.simple, nRes, holdout.data = task.hout)

formals(fitness.fun)

## -----------------------------------------------------------------------------
fidelity <- data.frame(
    c(1, 6, 11),
    c(1, 3, 5))
print(fidelity)

## ---- include = FALSE---------------------------------------------------------
set.seed(3)

## -----------------------------------------------------------------------------
run.gen.mufi <- slickEcr(
    fitness.fun = fitness.fun,
    lambda = 16,
    population = initials,
    mutator = mutator.simple,
    recombinator = crossover.simple,
    generations = 15,
    fidelity = fidelity)

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

plot_fronts(run.gen.mufi)

## -----------------------------------------------------------------------------
populations <- getPopulations(run.gen.mufi$log)

ind1.gen1 <- populations[[1]]$population[[1]]
attr(ind1.gen1, "fidelity")
attr(ind1.gen1, "runtime")

ind1.gen7 <- populations[[7]]$population[[1]]
attr(ind1.gen7, "fidelity")
attr(ind1.gen7, "runtime")

ind1.gen15 <- populations[[15]]$population[[1]]
attr(ind1.gen15, "fidelity")
attr(ind1.gen15, "runtime")

getStatistics(run.gen.mufi$log.newinds)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
cres <- collectResult(run.gen.mufi)
ggplot(cres, aes(x = cum.fid, y = eval.domHV, color = "Training")) +
  geom_line() + geom_point(size = 1) +
  geom_point(data = cres[cres$fid.reeval, ], shape = "x", size = 5) +
  geom_line(aes(x = cum.fid, y = true.hout.domHV, color = "Holdout")) +
  ylim(0, 1)

## -----------------------------------------------------------------------------
fidelity <- data.frame(1, 1, 10)
print(fidelity)

## ---- include = FALSE---------------------------------------------------------
set.seed(36)

## -----------------------------------------------------------------------------
run.dom.mufi <- slickEcr(
    fitness.fun = fitness.fun,
    lambda = 16,
    population = initials,
    mutator = mutator.simple,
    recombinator = crossover.simple,
    generations = 15,
    fidelity = fidelity)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts(run.dom.mufi)

## -----------------------------------------------------------------------------
getStatistics(run.dom.mufi$log.newinds)

## -----------------------------------------------------------------------------
popAggregate(run.dom.mufi$log, "fidelity")[[2]]

## ---- fig.width = 6, fig.height = 5-------------------------------------------
gen.i <- 3
logobj <- run.dom.mufi$log.newinds
stat <- getStatistics(logobj)
pop <- popAggregate(logobj, c("fidelity", "fitness"), data.frame = TRUE)
ngen.info <- pop[[which(stat$gen == gen.i + 1)]]
front <- fitnesses(run.dom.mufi, function(x) paretoEdges(x, c(1, 1))) %>%
    subset(gen == gen.i)
new.front <- fitnesses(run.dom.mufi, function(x) paretoEdges(x, c(1, 1))) %>%
    subset(gen == gen.i + 1)


ggplot() +
  geom_line(data = front, aes(x = perf, y = propfeat, linetype = "Gen3 Front")) +
  geom_line(data = new.front, aes(x = perf, y = propfeat, linetype = "Gen4 Front")) +
  geom_point(data = ngen.info,
    aes(x = fitness.perf, y = fitness.propfeat,
      shape = as.factor(fidelity),
      color = as.factor(fidelity),
      size = 3)) +
  scale_size_identity() +
  guides(colour = guide_legend(override.aes = list(size=3))) +
  xlim(0, 1) +
  ylim(0, 1) +
  coord_fixed()

## -----------------------------------------------------------------------------
fidelity <- data.frame(
    c(1, 3, 6, 11),
    c(1, 2, 3, 5),
    c(5, 4, 8, 10))
print(fidelity)

## ---- include = FALSE---------------------------------------------------------
set.seed(5)

## -----------------------------------------------------------------------------
run.all.mufi <- slickEcr(
    fitness.fun = fitness.fun,
    lambda = 16,
    population = initials,
    mutator = mutator.simple,
    recombinator = crossover.simple,
    generations = 15,
    fidelity = fidelity)

## -----------------------------------------------------------------------------
getStatistics(run.all.mufi$log.newinds)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts(run.all.mufi)

## -----------------------------------------------------------------------------
fidelity.new <- data.frame(
    c(1, 10, 17, 19),
    c(11, 20, 23, 25))
print(fidelity.new)

## ---- include = FALSE---------------------------------------------------------
set.seed(2)

## -----------------------------------------------------------------------------
run.all.mufi.20 <- continueEcr(run.all.mufi, 5,
  lambda = 8, fidelity = fidelity.new)

## -----------------------------------------------------------------------------
getStatistics(run.all.mufi.20$log.newinds)

## ---- fig.width = 6, fig.height = 5-------------------------------------------
plot_fronts(run.all.mufi.20)

## ---- include = FALSE---------------------------------------------------------
parallelStop()

library(rjags)
library(R2WinBUGS) # to use write.model()

# JAGS MCMC の結果を格納している mcmc.list を bugs オブジェクトに変更する関数
source("http://hosho.ees.hokudai.ac.jp/~kubo/ce/r/mcmc.list2bugs.R")

model.bugs <- function()
{
	Tau.noninformative <- 1.0E-4
	for (i in 1:N) {
		Y[i] ~ dpois(lambda[i])
		log(lambda[i]) <- beta1 + beta2 * (X[i] - Mean.X)
	}
	beta1 ~ dnorm(0, Tau.noninformative)
	beta2 ~ dnorm(0, Tau.noninformative)
}
file.model <- "model.bug.txt"
write.model(model.bugs, file.model)

load("d.RData")
list.data <- list(Y = d$y, X = d$x, Mean.X = mean(d$x), N = nrow(d))
inits <- list(beta1 = 0, beta2 = 0)
n.burnin <- 1000
n.chain <- 3
n.thin <- 2
n.iter <- n.thin * 1000

model <- jags.model(
	file = file.model, data = list.data,
	inits = inits, n.chain = n.chain
)
update(model, n.burnin) # burn in
post.mcmc.list <- coda.samples(
	model = model,
	variable.names = names(inits),
	n.iter = n.iter,
	thin = n.thin
)
post.bugs <- mcmc.list2bugs(post.mcmc.list)

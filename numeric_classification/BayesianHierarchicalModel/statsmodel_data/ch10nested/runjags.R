library(rjags)
library(R2WinBUGS) # to use write.model()

# JAGS MCMC の結果を格納している mcmc.list を bugs オブジェクトに変更する関数
source("http://hosho.ees.hokudai.ac.jp/~kubo/ce/r/mcmc.list2bugs.R")

model.bugs <- function()
{
	for (i in 1:N.sample) {
		Y[i] ~ dpois(lambda[i])
		log(lambda[i]) <- beta1 + beta2 * F[i] + r[i] + rp[Pot[i]]
	}
	beta1 ~ dnorm(0, 1.0E-4)
	beta2 ~ dnorm(0, 1.0E-4)
	for (i in 1:N.sample) {
		r[i] ~ dnorm(0, tau[1])
	}
	for (j in 1:N.pot) {
		rp[j] ~ dnorm(0, tau[2])
	}
	for (k in 1:N.tau) {
		tau[k] <- 1.0 / (s[k] * s[k])
		s[k] ~ dunif(0, 1.0E+4)
	}
}
file.model <- "model.bug.txt"
write.model(model.bugs, file.model)

d <- read.csv("d1.csv")
list.data <- list(
	Y = d$y, F = d$f, Pot = d$pot,
	N.sample = nrow(d), N.tau = 2,
	N.pot = length(unique(d$pot))
)
inits <- list(
	beta1 = 0,
	beta2 = 0,
	s = rep(1, list.data$N.tau),
	r = rnorm(list.data$N.sample, 0.0, 0.01),
	rp = rnorm(list.data$N.pot, 0.0, 0.01)
)
n.burnin <- 3000
n.chain <- 3
n.thin <- 10
n.iter <- n.thin * 3000

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

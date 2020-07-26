library(rjags)
library(R2WinBUGS) # to use write.model()

# JAGS MCMC の結果を格納している mcmc.list を bugs オブジェクトに変更する関数
source("http://hosho.ees.hokudai.ac.jp/~kubo/ce/r/mcmc.list2bugs.R")

model.bugs <- function()
{
	for (i in 1:N.data) {
		Y[i] ~ dbin(q[i], 8)
		logit(q[i]) <- a + r[i]
	}
	a ~ dnorm(0.0, 1.0E-4)
	for (i in 1:N.data) {
		r[i] ~ dnorm(0.0, tau)
	}
	tau <- 1 / (s * s)
	s ~ dunif(0.0, 1.0E+4)
}
file.model <- "model.bug.txt"
write.model(model.bugs, file.model)

d <- read.csv("data7a.csv")
list.data <- list(Y = d$y, N.data = nrow(d))
inits <- list(a = 0.0, s = 1.0, r = rnorm(nrow(d), 0.0, 0.01))
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

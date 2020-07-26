library(rjags)
library(R2WinBUGS) # to use write.model()

# JAGS MCMC の結果を格納している mcmc.list を bugs オブジェクトに変更する関数
source("http://hosho.ees.hokudai.ac.jp/~kubo/ce/r/mcmc.list2bugs.R")

model.bugs <- function()
{
	for (i in 1:N.data) {
		Y[i] ~ dbin(q, 8)
	}
	q ~ dunif(0.0, 1.0)
}
file.model <- "model.bug.txt"
write.model(model.bugs, file.model)

load("data.RData")
list.data <- list(Y = data, N.data = length(data))
inits <- list(q = 0.5)
n.burnin <- 1000
n.chain <- 3
n.thin <- 1
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


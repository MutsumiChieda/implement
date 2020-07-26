# R --vanilla!
d <- read.csv("data7a.csv")
load("post.mcmc.RData")

# plot(post.bugs)
pdf("plotbugs.pdf", width = 6, height = 6)
plot(post.bugs)
dev.off()

pdf("posta.pdf", width = 6, height = 3)
plot(post.list[,1,], smooth = F) # "beta1"
dev.off()
pdf("postb.pdf", width = 6, height = 3)
plot(post.list[,2,], smooth = F) # "beta2"
dev.off()

cnm <- colnames(post.mcmc)
m <- post.mcmc[,grep("rp", cnm)]
keys <- colnames(m)
list.density <- lapply(
	keys, function(key) {
		xy <- density(m[,key])
		list(x = xy$x, y = xy$y)
	}
)
names(list.density) <- keys
ymax <- max(sapply(list.density, function(xy) xy$y))
dymax <- 0.05
par(mar = c(2.5, 0.1, 0.1, 0.1), mgp = c(1.5, 0.5, 0))
plot(
	numeric(0), numeric(0),
	type = "n",
	xlim = quantile(m, probs = c(0.01, 0.99)),
	ylim = c(0, ymax * (1 + dymax * 2)),
	axes = FALSE,
	yaxs = "i",
	xlab = "", ylab = ""
)
abline(h = 0)
axis(1)
for (i in 1:10) {
	xy <- list.density[[i]]
	col <- c("#0000ff80", "#ff000080")[(i > 5) + 1]
	lines(xy$x, xy$y, lwd = 3, col = col)
	text(mean(xy$x), max(xy$y) + ymax * dymax, LETTERS[i], col = col)
}
dev.off()




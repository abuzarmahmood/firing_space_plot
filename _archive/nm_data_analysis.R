install.packages('car')

anova_dat = read.csv2('nm_data.csv',header=T, stringsAsFactors = FALSE)
anova_dat <- anova_dat[-1]
anova_dat2 <- anova_dat
anova_dat2[,c(1,2,5)] <- as.data.frame(sapply(anova_dat2[,c(1,2,5)], as.numeric))
anova_dat2[,c(3,4)] <- as.data.frame(sapply(anova_dat2[,c(3,4)], as.factor))

res.aov2 <- aov(rho_2 ~ laser * shuffle, data = anova_dat2)
summary(res.aov2)

library(car)
my_anova <- aov(rho_2 ~ laser * shuffle, data = anova_dat2)
Anova(my_anova, type = "III")
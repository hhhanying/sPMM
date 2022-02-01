library(data.table)
library(purrr)
library(ggplot2)
library(ggpubr)
library(MOFA2)

dt <- fread("ftp://ftp.ebi.ac.uk/pub/databases/mofa/microbiome/data.txt.gz")
dat = read.csv("./data/cleaned_data.csv")
dat_copy = copy(dt)

# bacteria
## estimate prior for Lambda
x = dat[2:181]

means = sapply(x, mean)
vars = sapply(x, var)
lambdas = 1/vars

beta_ = mean(lambdas)/var(lambdas)
alpha_ = mean(lambdas)*beta_
mu_ = mean(means)
sigma2_ = var(means)*(alpha_-1)/beta_

lambdas %>% ggdensity()
lambda_sim = rgamma(1000,alpha_, beta_)
lambda_sim%>%ggdensity()

means %>% ggdensity()
mean_sim = sapply(1:1000, function(x) rnorm(1,mu_-1,sqrt(sigma2_/lambda_sim[x])))
mean_sim%>%ggdensity()

dt[view == "Bacteria"] %>% ggdensity( x="value", fill="gray70")
x_sim = sapply(1:1000, function(x) rnorm(1,mean_sim[x],sqrt(1/lambda_sim[x])))
x_sim%>%ggdensity()

mu_ = mu_ - 1
c(alpha_,beta_,mu_,sigma2_) # 3.2438328 11.1157220 -1.0000000  0.9138574

# Fungi
x = dat[182:199]

means = sapply(x, mean)
vars = sapply(x, var)
lambdas = 1/vars

beta_ = mean(lambdas)/var(lambdas)
alpha_ = mean(lambdas)*beta_
mu_ = mean(means)
sigma2_ = var(means)*(alpha_-1)/beta_

alpha_;beta_;mu_;sigma2_

lambdas %>% ggdensity()
lambda_sim = rgamma(1000,alpha_, beta_)
lambda_sim%>%ggdensity()

means %>% ggdensity()
mean_sim = sapply(1:1000, function(x) rnorm(1,mu_,sqrt(sigma2_/2/lambda_sim[x])))
mean_sim%>%ggdensity()

dt[view == "Fungi"] %>% ggdensity( x="value", fill="gray70")
x_sim = sapply(1:1000, function(x) rnorm(1,mean_sim[x],sqrt(1/lambda_sim[x])))
x_sim%>%ggdensity()

sigma2_ = sigma2_/2
c(alpha_,beta_,mu_,sigma2_) # 3.416767e+00 1.105094e+01 1.446800e-16 6.660274e-01


# Viruses
x = dat[200:241]

means = sapply(x, mean)
vars = sapply(x, var)
lambdas = 1/vars

beta_ = mean(lambdas)/var(lambdas)
beta_ = beta_
alpha_ = mean(lambdas)*beta_

lambdas %>% ggdensity()
lambda_sim = rgamma(1000,alpha_, beta_)
lambda_sim%>%ggdensity()


mu_ = mean(means)
sigma2_ = var(means)/12
means %>% ggdensity()
mean_sim = sapply(1:1000, function(x) rnorm(1,mu_,sqrt(sigma2_)))
mean_sim%>%ggdensity()

dt[view == "Viruses"] %>% ggdensity( x="value", fill="gray70")
x_sim = sapply(1:1000, function(x) rnorm(1,mean_sim[x],sqrt(1/lambda_sim[x])))
x_sim%>%ggdensity()


c(alpha_,beta_,mu_,sigma2_) # 9.551973e-01 4.841954e-01 1.409721e-16 4.300383e-02



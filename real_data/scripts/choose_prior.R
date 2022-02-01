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
alpha_ = mean(lambdas)*beta_

lambdas %>% ggdensity()
lambda_sim = rgamma(1000,alpha_, beta_)
lambda_sim%>%ggdensity()


mu_ = mean(means)
sigma2_ = var(means)/6

means %>% ggdensity()
mean_sim = sapply(1:1000, function(x) rnorm(1,mu_,sqrt(sigma2_)))
mean_sim%>%ggdensity()

dt[view == "Viruses"] %>% ggdensity( x="value", fill="gray70")
x_sim = sapply(1:1000, function(x) rnorm(1,mean_sim[x],sqrt(1/lambda_sim[x])))
x_sim%>%ggdensity()


alpha_;beta_;mu_;sigma2_



> means = sapply(x, mean)
> vars = sapply(x, var)
> lambdas = 1/vars
> beta_ = mean(lambdas)/var(lambdas)
> alpha_ = mean(lambdas)*beta_
> mu_ = mean(means)
> sigma2_ = var(means)*(alpha_-1)/beta_
> alpha_;beta_;mu_;sigma2_




> mean_sim%>%ggdensity()
> mean_sim = sapply(1:1000, function(x) rnorm(1,mu_,sqrt(sigma2_/2/lambda_sim[x])))
> means %>% ggdensity()
> mean_sim = sapply(1:1000, function(x) rnorm(1,mu_,sqrt(sigma2_/2/lambda_sim[x])))
> mean_sim%>%ggdensity()
> dt[view == "Fungi"] %>% ggdensity( x="value", fill="gray70")
> x_sim = sapply(1:1000, function(x) rnorm(1,mean_sim[x],sqrt(1/lambda_sim[x])))
> x_sim%>%ggdensity()
> sigma2_ = sigma2_/2
# bacteria
## estimate prior for Lambda
mean_Bac = sapply(dat[2:181], mean)
var_Bac = sapply(dat[2:181], var)
lambda_Bac = 1/var_Bac
beta_Bac = mean(lambda_Bac)/var(lambda_Bac)
alpha_Bac = mean(lambda_Bac)*beta_Bac
alpha_Bac; beta_Bac

dat_copy = copy(dt)
# bacteria
dt[view == "Bacteria"] %>% ggdensity( x="value", fill="gray70")
mean(dt[view == "Bacteria"]$value)

alpha_Lambda =  1
beta_Lambda = 2
mu_Mu = 1
sigma2_Mu = 0.01

Lambda = c()
Mu = c()
x = c()
for(i in 1:1000){
lambda = rgamma(1, alpha_Lambda, beta_Lambda)
Lambda = c(Lambda, lambda)
mu = rnorm(1,mu_Mu, sigma2_Mu/lambda)
Mu = c(Mu, mu)
tem = rnorm(1, mu, 1/sqrt(lambda))
x = c(x,tem)
}
hist(Lambda)
hist(Mu)
help(hist)
x = data.frame(x=x)
x %>% ggdensity(x="x")

# Fungi
dt[view == "Fungi"] %>% ggdensity( x="value", fill="gray70")
mean(dt[view == "Fungi"]$value)
sd(dt[view == "Fungi"]$value)
alpha_Lambda =  3
beta_Lambda = 3
mu_Mu = 1
sigma2_Mu = 2

Lambda = c()
Mu = c()
x = c()
for(i in 1:1000){
  lambda = rgamma(1, alpha_Lambda, beta_Lambda)
  Lambda = c(Lambda, lambda)
  mu = rnorm(1,mu_Mu, sigma2_Mu/lambda)
  Mu = c(Mu, mu)
  tem = rnorm(1, mu, 1/sqrt(lambda))
  x = c(x,tem)
}
#hist(Lambda)
#hist(Mu)
x = data.frame(x=x)
x %>% ggdensity(x="x",xlim=c(-10,10))

# Viruses
dt[view == "Viruses"] %>% ggdensity( x="value", fill="gray70")
dt[view == "Viruses"] %>% ggdensity( x="value", fill="gray70",xlim=c(-3,3))
mean(dt[view == "Viruses"]$value)
sd(dt[view == "Viruses"]$value)
alpha_Lambda =  3
beta_Lambda = 0.3
mu_Mu = -0.5
sigma2_Mu = 0.000001

Lambda = c()
Mu = c()
x = c()
for(i in 1:1000){
  lambda = rgamma(1, alpha_Lambda, beta_Lambda)
  Lambda = c(Lambda, lambda)
  mu = rnorm(1,mu_Mu, sigma2_Mu/lambda)
  Mu = c(Mu, mu)
  tem = rnorm(1, mu, 1/sqrt(lambda))
  x = c(x,tem)
}
#hist(Lambda)
#hist(Mu)
x = data.frame(x=x)
x %>% ggdensity(x="x",xlim=c(-2,10))


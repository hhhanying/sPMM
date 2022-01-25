library(data.table)
library(purrr)
library(ggplot2)
library(ggpubr)
library(MOFA2)

dt <- fread("ftp://ftp.ebi.ac.uk/pub/databases/mofa/microbiome/data.txt.gz")

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


Working with count compositional data
================
SK
2023-07-31

# Introduction

Dirichlet multinomial ($DM$) distribution is a compound probability
distribution, where a probability vector $\theta$ is drawn from a
Dirichlet distribution with parameter vector $\alpha$ and an observation
data (counts; $y$) drawn from a multinomial distribution with
probability vector $\theta$ and total number of trials $n$. The
Dirichlet prior - the vector $\alpha\in\mathbb{R}^{+}$, can be seen as
**pseudo**count:

- $y \sim \mathrm{Multinomial}(\theta, n)$
- $\theta \sim \mathrm{Dirichlet}(\alpha)$
- $\alpha = \ldots$

Crash course on Dirichlet and Multinomial distributions:
<https://ixkael.github.io/dirichlet-multinomial-conjugate-priors/>

# Lets simulate data from DM distribution

Here I assume we have samples obtained from a number of patients before
and after treatment. Each sample measures mRNA abundances (**counts**)
of $K$ genes (categories). Hence, sample $i$ is represented by
$K$-vector $\vec{y}_{i}=\{y_{i1},y_{i2},\ldots,y_{ik}\}$.

For patient $i$ we have two such samples:

- $\vec{y}_i^{b}$: collected before treatment
- $\vec{y}_i^{a}$: collected after treatment

The sum of the counts in each sample is fixed to $n_i=10^4$, i.e.,
$n_i=\sum_{j=1}^k y_{ij}=10^4$. We can use a STAN model (code shown
below) to simulate data from $DM$ distribution:

- $y \sim \mathrm{Multinomial}(\theta, n)$
- $\theta \sim \mathrm{Dirichlet}(\alpha)$
- $\alpha = \xi\cdot \mathrm{softmax}(\gamma + \beta x)$

where $x$ is a design variable, with $x=1$ for samples before and $x=-1$
for samples after treatment. $\gamma$ is intercept, $\beta$ is the
effect of the treatment, and $\xi\in\mathbb{R}^{+}$ is precision
(inverse of overdispersion).

``` stan
data {
  int<lower=0> K; // categories (genes)
  int<lower=0> N; // samples
  vector[K] gamma; // intercept
  real<lower=0> gamma_sigma; 
  vector[K] delta; // effect
  real<lower=0> delta_sigma;
  real xi; // precision
  int n; // total counts (tries)
}

generated quantities {
  int y_a [N,K]; // sample after
  int y_b [N,K]; // sample before
  vector [K] mu [2]; // intermediate alphas from dirichlet dist.
  vector [K] alpha [N]; 
  vector [K] beta [N];
  
  for(i in 1:N) {
   for(j in 1:K) {
     alpha[i][j] = normal_rng(gamma[j], gamma_sigma);
     beta[i][j] = normal_rng(delta[j], delta_sigma);
   } 
   mu[1] = dirichlet_rng(xi*softmax(alpha[i] + beta[i])); // before treatment sim. from DM.
   mu[2] = dirichlet_rng(xi*softmax(alpha[i] - beta[i])); // after treatment sim. from DM.
   y_a[i,] = multinomial_rng(mu[1], n); // simulate from multinomial dist. -> after sample
   y_b[i,] = multinomial_rng(mu[2], n); // simulate from multinomial dist. -> before sample 
  }
}
```

To simulate data we only need to provide the inputs (see ‘data’ block in
the above STAN code):

``` r
set.seed(seed = 123456)
K <- 7
N_train <- 10
N_test <- 10
N <- N_train + N_test
gamma <- rnorm(n = K, mean = 0, sd = 1)
delta <- c(rnorm(n = K-5, mean = 0, sd = 0.05), 
           rnorm(n = 5, mean = 0, sd = 0.35))

xi <- 500
n <- 10^4
gamma_sigma = 0.1
delta_sigma = 0.2
```

… and simulate 20 samples with:

We then extract the simulated observations: two matrices $\vec{y}^{a}$
and $\vec{y}^{b}$, with rows as samples and columns as genes. We treat
the first 10 samples (rows in each matrix) as observed/training samples
($D_{\text{train}}$), and we will treat the remaining 10 samples as
unobserved/testing sample ($D_{\text{test}}$).

We can visualize the data using ggplot. Each point is an observation
(mRNA count; y-axis) for a given gene (x-axis) from a given sample.
Notice the differences in mRNA counts between the two conditions for
genes 12, 5, etc. Train data are shown as filled points and test data as
hollow points.

<img src="Tutorial_files/figure-gfm/unnamed-chunk-13-1.png" style="display: block; margin: auto;" />

## Modeling

We know that the data comes from $DM$ distribution. What would happen if
we assumed that the data comes from:

1)  multinomial distribution (model $M$) and
2)  dirichlet multinomial distribution (model $DM$)?

For this we will use two models that I have developed for differential
immunoglobulin gene usage (DGU) analysis between pairs of samples. The
models have *similar* structure as the model used for data simulation.

The main difference between the simulation model and the DGU models is
that the DGU models are hierarchical. They assume that samples within a
condition (here condition = “before” vs “after” treatment) are not
completely independent from each other, and use partial pooling to
enable sharing of information (about parameters) between samples within
a condition.

Model $DM$ and $M$ are identical, except for the following differences:

### Model $DM$

- $\vec{y_i} \sim \mathrm{Multinomial}(\vec{\theta_i}, n_i)$
- $\vec{\theta_i} \sim \mathrm{Dirichlet}(\xi \cdot \mathrm{softmax}(\vec{\mu_i}))$
- $\vec{\mu_i} = \vec{\gamma}_i+\vec{\beta}_{i}x$

### Model $M$

- $\vec{y_i} \sim \mathrm{Multinomial}(\vec{\theta_i}, n_i)$
- $\vec{\theta_i} = \mathrm{softmax}(\vec{\mu_i})$
- $\vec{\mu_i} = \vec{\gamma}_i+\vec{\beta}_{i}x$

### STAN code for model $DM$

``` stan
functions {
  real dirichlet_multinomial_complete_lpmf(int[] y, vector alpha) {
    real sum_alpha = sum(alpha);
    return lgamma(sum_alpha) - lgamma(sum(y) + sum_alpha)
           + lgamma(sum(y)+1) - sum(lgamma(to_vector(y)+1))
           + sum(lgamma(to_vector(y) + alpha)) - sum(lgamma(alpha));
  }
}

data {
  int <lower = 0> N_sample; // number of samples (repertoires)
  int <lower = 0> N_gene; // number of genes
  int Y_1 [N_gene, N_sample]; // number of successes (cells) in samples x gene
  int Y_2 [N_gene, N_sample]; // number of successes (cells) in samples x gene
  int N [N_sample, 2]; // number of total tries (repertoire size)
  // test data
  int <lower = 0> N_sample_test; // number of samples (repertoires)
  int Y_1_test [N_gene, N_sample_test]; // number of successes (cells) in samples x gene
  int Y_2_test [N_gene, N_sample_test]; // number of successes (cells) in samples x gene
  int N_test [N_sample_test, 2]; // number of total tries (repertoire size)
}

transformed data {
  real N_real [N_sample, 2];
  N_real = N;
}

parameters {
  vector [N_gene] alpha_mu_gene;
  real <lower=0> beta_sigma_gene;
  real <lower=0> alpha_sigma_gene;
  real <lower=0> beta_sigma_pop;
  vector [N_gene] beta_z [N_sample];
  vector [N_gene] alpha_z [N_sample];
  vector [N_gene] beta_z_gene;
  real <lower=0> xi;
}

transformed parameters {
  vector [N_gene] alpha [N_sample];
  vector [N_gene] beta [N_sample];
  vector [N_gene] beta_mu_gene;
  
  beta_mu_gene = 0+beta_sigma_pop*beta_z_gene;
  for(i in 1:N_sample) {
    beta[i] = beta_mu_gene + beta_sigma_gene * beta_z[i];
    alpha[i] = alpha_mu_gene + alpha_sigma_gene * alpha_z[i];
  }
}

model {
  target += exponential_lpdf(xi | 0.05);
  target += cauchy_lpdf(beta_sigma_pop | 0, 1);
  target += cauchy_lpdf(alpha_sigma_gene | 0, 1);
  target += cauchy_lpdf(beta_sigma_gene | 0, 1);
  for(i in 1:N_sample) {
    target += normal_lpdf(alpha_z[i] | 0, 1);
    target += normal_lpdf(beta_z[i] | 0, 1);
  }
  target += normal_lpdf(beta_z_gene | 0, 1);
  target += normal_lpdf(alpha_mu_gene | 0, 5);
  
  // likelihood
  for(i in 1:N_sample) {
    target += dirichlet_multinomial_complete_lpmf(Y_1[,i]|xi * softmax(alpha[i]-beta[i]));
    target += dirichlet_multinomial_complete_lpmf(Y_2[,i]|xi * softmax(alpha[i]+beta[i]));
  }
}

generated quantities {
  int Y_hat_1 [N_gene, N_sample];
  int Y_hat_2 [N_gene, N_sample];
  int Y_hat_group_1 [N_gene, N_sample];
  int Y_hat_group_2 [N_gene, N_sample];
  real log_lik [N_sample, 2];
  real log_lik_train [N_sample, 2];
  real log_lik_test [N_sample_test, 2];
  
  vector [N_gene] p [2];
  vector [N_gene] mu [2];
  real a [N_gene];
  real b [N_gene];

  for(i in 1:N_sample) {
    p[1] = dirichlet_rng(xi * softmax(alpha[i]-beta[i]));
    Y_hat_1[,i] = multinomial_rng(p[1], N[i,1]);
    p[2] = dirichlet_rng(xi * softmax(alpha[i]+beta[i]));
    Y_hat_2[,i] = multinomial_rng(p[2], N[i,2]);
    
    log_lik[i,1] = dirichlet_multinomial_complete_lpmf(Y_1[,i]|xi*softmax(alpha[i]-beta[i]));
    log_lik[i,2] = dirichlet_multinomial_complete_lpmf(Y_2[,i]|xi*softmax(alpha[i]+beta[i]));
  }
  
  // PPC: condition-specific
  a = normal_rng(alpha_mu_gene, alpha_sigma_gene);
  b = normal_rng(beta_mu_gene, beta_sigma_gene);
  mu[1] = xi * softmax(to_vector(a)-to_vector(b));
  mu[2] = xi * softmax(to_vector(a)+to_vector(b));
  // PPC: test
  for(i in 1:N_sample_test) {
    log_lik_test[i,1] = dirichlet_multinomial_complete_lpmf(Y_1_test[,i]|mu[1]);
    log_lik_test[i,2] = dirichlet_multinomial_complete_lpmf(Y_2_test[,i]|mu[2]);
  }
  // PPC: train
  for(i in 1:N_sample) {
    log_lik_train[i,1] = dirichlet_multinomial_complete_lpmf(Y_1[,i]|mu[1]);
    log_lik_train[i,2] = dirichlet_multinomial_complete_lpmf(Y_2[,i]|mu[2]);
    Y_hat_group_1[,i] = multinomial_rng(dirichlet_rng(mu[1]), N[i,1]);
    Y_hat_group_2[,i] = multinomial_rng(dirichlet_rng(mu[2]), N[i,2]);
  }
}
```

### STAN code for model $M$

``` stan
data {
  int <lower = 0> N_sample; // number of samples (repertoires)
  int <lower = 0> N_gene; // number of genes
  int Y_1 [N_gene, N_sample]; // number of successes (cells) in samples x gene
  int Y_2 [N_gene, N_sample]; // number of successes (cells) in samples x gene
  int N [N_sample, 2]; // number of total tries (repertoire size)
  // test data
  int <lower = 0> N_sample_test; // number of samples (repertoires)
  int Y_1_test [N_gene, N_sample_test]; // number of successes (cells) in samples x gene
  int Y_2_test [N_gene, N_sample_test]; // number of successes (cells) in samples x gene
  int N_test [N_sample_test, 2]; // number of total tries (repertoire size)
}

transformed data {
  real N_real [N_sample, 2];
  N_real = N;
}

parameters {
  vector [N_gene] alpha_mu_gene;
  real <lower=0> beta_sigma_gene;
  real <lower=0> alpha_sigma_gene;
  real <lower=0> beta_sigma_pop;
  vector [N_gene] beta_z [N_sample];
  vector [N_gene] alpha_z [N_sample];
  vector [N_gene] beta_z_gene;
}

transformed parameters {
  vector [N_gene] alpha [N_sample];
  vector [N_gene] beta [N_sample];
  vector [N_gene] beta_mu_gene;
  
  beta_mu_gene = 0+beta_sigma_pop*beta_z_gene;
  for(i in 1:N_sample) {
    beta[i]=beta_mu_gene+beta_sigma_gene*beta_z[i];
    alpha[i]=alpha_mu_gene+alpha_sigma_gene*alpha_z[i];
  }
}

model {
  target += cauchy_lpdf(beta_sigma_pop | 0, 1);
  target += cauchy_lpdf(alpha_sigma_gene | 0, 1);
  target += cauchy_lpdf(beta_sigma_gene | 0, 1);
  for(i in 1:N_sample) {
    target += normal_lpdf(alpha_z[i] | 0, 1);
    target += normal_lpdf(beta_z[i] | 0, 1);
  }
  target += normal_lpdf(beta_z_gene | 0, 1);
  target += normal_lpdf(alpha_mu_gene | 0, 5);
  
  // likelihood
  for(i in 1:N_sample) {
    target += multinomial_lpmf(Y_1[,i] | softmax(alpha[i]-beta[i]));
    target += multinomial_lpmf(Y_2[,i] | softmax(alpha[i]+beta[i]));
  }
}

generated quantities {
  int Y_hat_1 [N_gene, N_sample];
  int Y_hat_2 [N_gene, N_sample];
  int Y_hat_group_1 [N_gene, N_sample];
  int Y_hat_group_2 [N_gene, N_sample];
  real log_lik [N_sample, 2];
  real log_lik_test [N_sample_test, 2];
  real log_lik_train [N_sample, 2];
  
  vector [N_gene] p [2];
  vector [N_gene] mu [2];
  real a [N_gene];
  real b [N_gene];
  
  // PPC
  for(i in 1:N_sample) {
    p[1] = softmax(alpha[i] - beta[i]);
    p[2] = softmax(alpha[i] + beta[i]);
    
    Y_hat_1[,i] = multinomial_rng(p[1], sum(Y_1[,i]));
    Y_hat_2[,i] = multinomial_rng(p[2], sum(Y_2[,i]));
    log_lik[i,1] = multinomial_lpmf(Y_1[,i] | p[1]);
    log_lik[i,2] = multinomial_lpmf(Y_2[,i] | p[2]);
  }
  
  // PPC - test (condition-specific)
  a = normal_rng(alpha_mu_gene, alpha_sigma_gene);
  b = normal_rng(beta_mu_gene, beta_sigma_gene);
  mu[1]=softmax(to_vector(a)-to_vector(b));
  mu[2]=softmax(to_vector(a)+to_vector(b));
  for(i in 1:N_sample_test) {
    log_lik_test[i,1] = multinomial_lpmf(Y_1_test[,i]|mu[1]);
    log_lik_test[i,2] = multinomial_lpmf(Y_2_test[,i]|mu[2]);
  }
  for(i in 1:N_sample) {
    log_lik_train[i,1] = multinomial_lpmf(Y_1[,i]|mu[1]);
    log_lik_train[i,2] = multinomial_lpmf(Y_2[,i]|mu[2]);
    Y_hat_group_1[,i] = multinomial_rng(mu[1], N[i,1]);
    Y_hat_group_2[,i] = multinomial_rng(mu[2], N[i,2]);
  }
}
```

### Model fitting

We will fit models $DM$ and $M$ using $D_{\text{train}}$.

``` r
fit_dm <- rstan::sampling(object = model_dm,
                          data = u,
                          chains = 3,
                          cores = 3,
                          iter = 3500,
                          warmup = 1500,
                          control = list(adapt_delta = 0.95, 
                                         max_treedepth = 13),
                          algorithm = "NUTS")
```

``` r
fit_m <- rstan::sampling(object = model_m,
                         data = u,
                         chains = 3,
                         cores = 3,
                         iter = 3500,
                         warmup = 1500,
                         control = list(adapt_delta = 0.95, 
                                        max_treedepth = 13),
                         algorithm = "NUTS")
```

## Posterior predictive checks for each sample

We will use the parameters at the lowest level of the models, which are
sample specific parameters, to generate new data.

For model $DM$ predictions:

- a = dirichlet_rng(GLM)
- y_hat = multinomial_rng(a, n)

For model $M$ predictions:

- y_hat = multinomial_rng(GLM, n)

These predictions (means and 95% HDIs) will be compared against the
observed data. Take a look at the predictions with narrow 95% HDIs by
model $M$!

<img src="Tutorial_files/figure-gfm/unnamed-chunk-23-1.png" style="display: block; margin: auto;" />

Nearly all observations fall within the 95% HDIs of the counts predicted
by either model.

    ##         
    ##           DM   M
    ##   in_hdi 140 140

## PPC from condition-specific parameter sets

To check how well our models can predict the unobserved/test data
($D_{\text{test}}$), we will generate (simulate) new data from each
model, and check whether the unobserved data falls within the 95% HDI of
the posterior of the simulated data points.

In general, we can simulate data in two ways using $M$ and $DM$:

1.  we can use the posteriors of the sample-specific parameters (1st
    layer of parameters in the models) and draw random samples from
    multinomial or dirichlet-multinomial distribution

### Model $DM$

- $\vec{y_i} \sim \mathrm{Multinomial}(\vec{\theta_i}, n_i)$
- $\vec{\theta_i} \sim \mathrm{Dirichlet}(\xi \cdot \mathrm{softmax}(\vec{\mu_i}))$
- $\vec{\mu_i} = \vec{\gamma}_i+\vec{\beta}_{i}x$

### Model $M$

- $\vec{y_i} \sim \mathrm{Multinomial}(\vec{\theta_i}, n_i)$
- $\vec{\theta_i} = \mathrm{softmax}(\vec{\mu_i})$
- $\vec{\mu_i} = \vec{\gamma}_i+\vec{\beta}_{i}x$

However, in this case we cannot simulate new data using the
sample-specific parameters, i.e. for each observed sample we have a set
of parameters ththese parameters correspond to each of the observed
samples. For this, we have to use the condition-specific parameters
found at the 2nd layer of model structure (see STAN code; generated
quantities).

Now, the model $M$ predictions (means and 95% HDIs) are associated with
large degree of uncertainty, compensating for the too certain parameters
from the sample-specific layer. However, many low count observations are
outside the 95% HDIs of the predictions made by model $M$. Model $DM$
does a better prediction overall using its condition-specific
parameters.

<img src="Tutorial_files/figure-gfm/unnamed-chunk-27-1.png" style="display: block; margin: auto;" />

### How many $D_{\text{train}}$ observations are outside the 95% HDIs of the predictions?

Model $DM$ does more meaningful predictions with respect to both
$D_{\text{test}}$ and $D_{\text{train}}$.

    ##          
    ##            DM   M
    ##   in_hdi  137 138
    ##   out_hdi   3   2

### How many $D_{\text{test}}$ observations are outside the 95% HDIs of the predictions?

    ##          
    ##            DM   M
    ##   in_hdi  135 133
    ##   out_hdi   5   7

## Model comparison

### Widely Applicable Information Criterion (WAIC)

We compute the average log probability ($\text{lppd}$) for observation
$y_i$ given posterior draw $s$ of the model parameters ($\Theta$):
$\text{lppd}(y,\Theta)=\sum\limits_{i=1}^n\text{log}\dfrac{1}{S}\sum\limits_{s=1}^S \text{Pr}(y_i|\Theta_s)$.
Moreover, we compute the effective number of parameters,
$p_{\text{eff}}=\sum\limits_{i=1}^n V(y_i)$, where $V(y_i)$ is the
variance of the log probability of observation $y_i$. Finally,
$\text{WAIC} = -2(\text{lppd}-p_{\text{eff}})$, and we compute
$\text{WAIC}$ for $D_{\text{train}}$ and $D_{\text{test}}$ for each
model.

#### Model $DM$ WAIC

#### $D_{\text{train}}$

    ## lppd: -851.0969

    ## p_eff: 256.1491

    ## WAIC: 2214.492

    ## SE of lppd = lppd * sqrt(N) =: 11.61456

#### $D_{\text{test}}$

As expected, we see slightly worse (larger) for model $DM$ for
$D_{\text{test}}$ compared to $D_{\text{train}}$:

    ## lppd: -868.1147

    ## p_eff: 294.6691

    ## WAIC: 2325.568

    ## SE of lppd = lppd * sqrt(N) =: 11.19148

#### Model $M$ WAIC

Notice that $\text{lppd}(D_{\text{train}})$ of model $M$ is much smaller
than $\text{lppd}(D_{\text{train}})$ of model $DM$. This picture is
consistent with the accurate predictions of model $M$ with narrow 95%
HDIs.

#### $D_{\text{train}}$

    ## lppd: -10440.21

    ## p_eff: 2180669

    ## WAIC: 4382218

    ## SE of lppd = lppd * sqrt(N) =: 631.0579

#### $D_{\text{test}}$

However, now have a look at the incredible high (bad) values for
$\text{lppd}(D_{\text{test}})$ of model $M$ given $D_{\text{test}}$. We
see especially high $p_{\text{eff}}$, which makes sense given the highly
variable PPCs.

This indicates overfitting, i.e. model $M$ can predict accurately
$D_{\text{train}}$ based on the sample-specific parameters, however, the
model does not generalize well to new data.

    ## lppd: -11477.73

    ## p_eff: 2608419

    ## WAIC: 5239793

    ## SE of lppd = lppd * sqrt(N) =: 650.9595

# Now the other way around: simulate data from $M$ distribution

``` stan
data {
  int<lower=0> K; // categories (genes)
  int<lower=0> N; // samples
  vector[K] gamma; // intercept
  real<lower=0> gamma_sigma; 
  vector[K] delta; // effect
  real<lower=0> delta_sigma;
  int n; // total counts (tries)
}

generated quantities {
  int y_a [N,K]; // sample after
  int y_b [N,K]; // sample before
  vector [K] mu [2]; // intermediate alphas from dirichlet dist.
  vector [K] alpha [N]; 
  vector [K] beta [N];
  
  for(i in 1:N) {
   for(j in 1:K) {
     alpha[i][j] = normal_rng(gamma[j], gamma_sigma);
     beta[i][j] = normal_rng(delta[j], delta_sigma);
   } 
   mu[1] = softmax(alpha[i] + beta[i]); // before treatment sim. from DM.
   mu[2] = softmax(alpha[i] - beta[i]); // after treatment sim. from DM.
   y_a[i,] = multinomial_rng(mu[1], n); // simulate from multinomial dist. -> after sample
   y_b[i,] = multinomial_rng(mu[2], n); // simulate from multinomial dist. -> before sample 
  }
}
```

To simulate data we only need to provide the inputs (see ‘data’ block in
the above STAN code):

``` r
set.seed(seed = 12345)
K <- 7
N_train <- 10
N_test <- 10
N <- N_train + N_test
gamma <- rnorm(n = K, mean = 0, sd = 1)
delta <- c(rnorm(n = K-5, mean = 0, sd = 0.05), 
           rnorm(n = 5, mean = 0, sd = 0.25))
n <- 10^4

gamma_sigma = 0.001 # approx. complete pooling within gene -> very little noise
delta_sigma = 0.001 # approx. complete pooling within gene
```

… and simulate 20 samples with:

We then extract the simulated observations: two matrices $\vec{y}^{a}$
and $\vec{y}^{b}$, with rows as samples and columns as genes. We treat
the first 10 samples (rows in each matrix) as observed/training samples
($D_{\text{train}}$), and we will treat the remaining 10 samples as
unobserved/testing sample ($D_{\text{test}}$).

We can visualize the data using ggplot. Each point is the mRNA count
(y-axis) for a given gene (x-axis) of a sample (point). Notice the
differences in mRNA counts between the two conditions for genes 12, 5,
7, etc. Filled points are train and hollow points are test data.

<img src="Tutorial_files/figure-gfm/unnamed-chunk-45-1.png" style="display: block; margin: auto;" />

## Modeling

Same models, $DM$ and $M$, as before.

``` r
fit_dm <- rstan::sampling(object = model_dm,
                          data = u,
                          chains = 3,
                          cores = 3,
                          iter = 3500,
                          warmup = 1500,
                          control = list(adapt_delta = 0.99, 
                                         max_treedepth = 14),
                          algorithm = "NUTS")
```

``` r
fit_m <- rstan::sampling(object = model_m,
                         data = u,
                         chains = 3,
                         cores = 3,
                         iter = 3500,
                         warmup = 1500,
                         control = list(adapt_delta = 0.95, max_treedepth = 14),
                         algorithm = "NUTS")
```

## Posterior predictive checks for each sample

<img src="Tutorial_files/figure-gfm/unnamed-chunk-53-1.png" style="display: block; margin: auto;" />

    ##          
    ##            DM   M
    ##   in_hdi  140 139
    ##   out_hdi   0   1

## PPC from condition-specific parameters

<img src="Tutorial_files/figure-gfm/unnamed-chunk-56-1.png" style="display: block; margin: auto;" />

### How many $D_{\text{train}}$ observations are outside the 95% HDIs of the predictions?

    ##          
    ##            DM   M
    ##   in_hdi  140 139
    ##   out_hdi   0   1

### How many $D_{\text{test}}$ observations are outside the 95% HDIs of the predictions?

    ##          
    ##            DM   M
    ##   in_hdi  140 139
    ##   out_hdi   0   1

## Model comparison with the Widely Applicable Information Criterion (WAIC)

### Model $DM$ WAIC

#### $D_{\text{train}}$

Notice that the loo package generated similar estimates:

    ## lppd: -668.5872

    ## p_eff: 4.748375

    ## WAIC: 1346.671

    ## SE of lppd = lppd * sqrt(N) =: 0.7474915

#### $D_{\text{test}}$

    ## lppd: -670.2633

    ## p_eff: 4.989143

    ## WAIC: 1350.505

    ## SE of lppd = lppd * sqrt(N) =: 0.7749125

### Model $M$ WAIC

#### $D_{\text{train}}$

    ## lppd: -575.2776

    ## p_eff: 35.67954

    ## WAIC: 1221.914

    ## SE of lppd = lppd * sqrt(N) =: 5.147254

#### $D_{\text{test}}$

    ## lppd: -593.3237

    ## p_eff: 46.08621

    ## WAIC: 1278.82

    ## SE of lppd = lppd * sqrt(N) =: 6.739216

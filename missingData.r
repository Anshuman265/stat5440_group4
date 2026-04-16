# STAT 5440 - Bayesian Missing  and Censor Data
# =============================================================================
needed <- c("NHANES", "mice", "brms", "rstan", "dplyr", "ggplot2", 
            "bayesplot", "naniar", "posterior", "patchwork")

# Check which ones are NOT installed
missing <- needed[!(needed %in% installed.packages()[,"Package"])]

# Install the ones that are missing
if(length(missing)) install.packages(missing)

# SETUP
library(NHANES)
library(mice)
library(brms)
library(rstan)
library(dplyr)
library(ggplot2)
library(bayesplot)
library(naniar)
library(posterior)
library(patchwork)

set.seed(5440)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
dir.create("figures", showWarnings = FALSE)


####SIMPLE ANALYISIS TOY EXAMPLE:

dat1= NHANES  %>% 
  filter(Age >= 20)  %>% 
  select(BPSysAve, Age, BMI, Weight, Gender, TotChol, PhysActive, SmokeNow)  %>% 
  slice_sample(n = 2000) %>% 
  na.omit()


##Create missingness in BMI

# MCAR 30% of the rows
dat1$BMI_mcar<- dat1$BMI
dat1$BMI_mcar[sample(1:nrow(dat1), size = 0.3*nrow(dat1))] <- NA

#MAR (30%) (missingness depends on weight and age)
dat1$BMI_mar <- dat1$BMI
dat1$BMI_mar[runif(nrow(dat1)) < plogis(0.05*(dat1$Weight - 200) + 0.05*(dat1$Age - 50))] <- NA

# NMAR (missingness depends on the unobserved BMI itself)
dat1$BMI_mnar= dat1$BMI
dat1$BMI_mnar[runif(nrow(dat1)) < plogis(0.1*(dat1$BMI - 30))] <- NA



#Histogram of the four BMI variables, overlay
p_bmi2 <- ggplot(dat1, aes(x = BMI)) +
  geom_histogram(aes(y = ..density..), fill = "grey80",color="grey", alpha=0.1, bins = 50) +
  geom_density(color = "black", size = 1.5) +
  geom_density(aes(x = BMI_mcar), color = "darkblue", size = 1, alpha=0.7) +
  geom_density(aes(x = BMI_mar), color = "darkgreen", size = 1) +
  geom_density(aes(x = BMI_mnar), color = "maroon", size = 1) +
  labs(x = "BMI", y = "Density",
       title = "Histogram of BMI with missingness patterns") +
  theme_minimal() + 
  theme(legend.position = "none")


#We are interested in understanding to predict Blood Pressure (BPSysAve) as a function of Age, BMI, excercise, smoking, etc.
#We want to know how the missingness in BMI affects our inference about the relationship between BMI and Blood Pressure.

#lets assume that missing is MAR
dat1$BMI=dat1$BMI_mar
dat1= dat1 %>% 
  select(BPSysAve, Age, BMI, Weight, Gender, TotChol, PhysActive, SmokeNow)   

#(1) If we simple ignore missing (complete-case)
# Drop any row with any NA in the variables we model.

dat_cc <- na.omit(dat1)
n_drop <- nrow(dat1) - nrow(dat_cc)


fit_cc <- brm(
  BPSysAve ~.,
  data    = dat_cc,
  chains  = 2, 
  iter = 200, 
  seed = 5440, 
  refresh = 0
)

cat(" Complete-case posterior summary")
print(summary(fit_cc))

# Posterior predictive check for the CC model
p_ppc_cc <- pp_check(fit_cc, ndraws = 100) +
  ggtitle("PPC: complete-case model")
ggsave("figures/fig_ppc_cc.png", p_ppc_cc, width = 7, height = 4, dpi = 200)


## (2) MULTIPLE IMPUTATION  (mice + brm_multiple)
imp <- mice(dat1, m = 10, seed = 5440, printFlag = FALSE)

fit_mi <- brm_multiple(
  BPSysAve ~.,
  data    = imp,
  chains  = 2, iter = 200, seed = 5440, refresh = 0
)

cat(" MI posterior summary")
print(summary(fit_mi))

# MI diagnostics

print(round(fit_mi$rhats, 3))

p_trace <- mcmc_trace(fit_mi, pars = "b_BMI") +
  ggtitle("Trace of b_BMI across chains ")
ggsave("figures/fig_trace_b_bmi.png", p_trace,
       width = 8, height = 3.5, dpi = 200)

p_ppc_mi <- pp_check(fit_mi, ndraws = 100) +
  ggtitle("PPC: MI model")
ggsave("figures/fig_ppc_mi.png", p_ppc_mi, width = 7, height = 4, dpi = 200)

png("figures/fig3_obs_vs_imputed.png", width = 1400, height = 500, res = 200)
print(densityplot(imp, ~ BMI + Weight))
dev.off()

png("figures/fig4_stripplot.png", width = 1600, height = 700, res = 200)
print(stripplot(imp, BMI ~ .imp, pch = 20, cex = 0.8))
dev.off()


## (3) JOINT BAYESIAN (Stan)
# Stan model imputes BMI only

dat_stan <- dat |> filter(!is.na(Weight))
ii_obs   <- which(!is.na(dat_stan$BMI))
ii_mis   <- which(is.na(dat_stan$BMI))

stan_data <- list(
  N       = nrow(dat_stan),
  N_obs   = length(ii_obs),
  N_mis   = length(ii_mis),
  ii_obs  = ii_obs,
  ii_mis  = ii_mis,
  bmi_obs = dat_stan$BMI[ii_obs],
  age     = dat_stan$Age,
  weight  = dat_stan$Weight,
  y       = dat_stan$BPSysAve
)

stan_code <- "
data {
  int<lower=0> N;
  int<lower=0> N_obs;
  int<lower=0> N_mis;
  array[N_obs] int<lower=1,upper=N> ii_obs;
  array[N_mis] int<lower=1,upper=N> ii_mis;
  vector[N_obs] bmi_obs;
  vector[N] age;
  vector[N] weight;
  vector[N] y;
}
parameters {
  vector[N_mis] bmi_mis;                            // <-- imputed values
  real mu_bmi;  real<lower=0> sigma_bmi;            // imputation model
  real alpha;   real b_age;  real b_bmi;  real b_wt; // analysis model
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] bmi;
  bmi[ii_obs] = bmi_obs;
  bmi[ii_mis] = bmi_mis;
}
model {
  bmi ~ normal(mu_bmi, sigma_bmi);                                     // imputation
  y   ~ normal(alpha + b_age*age + b_bmi*bmi + b_wt*weight, sigma);    // analysis
  mu_bmi ~ normal(27, 10);  sigma_bmi ~ normal(0, 5);
  alpha  ~ normal(120, 20); b_age ~ normal(0, 1);
  b_bmi  ~ normal(0, 2);    b_wt  ~ normal(0, 2);
  sigma  ~ normal(0, 10);
}
"

fit_joint <- stan(model_code = stan_code, data = stan_data,
                  iter = 2000, chains = 2, seed = 5440, refresh = 0)

cat("Joint Bayesian (Stan) posterior summary")
print(fit_joint, pars = c("alpha","b_age","b_bmi","b_wt",
                          "sigma","mu_bmi","sigma_bmi"))



# COMPARE ALL THREE 

extract_ci <- function(draws_vec, label, coef) {
  data.frame(
    Method      = label,
    Coefficient = coef,
    Mean        = mean(draws_vec),
    Lower       = quantile(draws_vec, 0.025),
    Upper       = quantile(draws_vec, 0.975),
    row.names   = NULL
  )
}

draws_cc    <- as_draws_df(fit_cc)
draws_mi_df <- as_draws_df(fit_mi)
draws_jt    <- as_draws_df(fit_joint)
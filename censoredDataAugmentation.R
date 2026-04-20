# =============================================================================
# STAT 5440 — Applied Bayesian Modeling (Spring 2026)
# Group 4 Final Project: Missing & Censored Data
# =============================================================================
# FILE:    censored_data_augmentation.R
# PURPOSE: Demonstrates data augmentation for censored survival data using
#          Stan directly. Latent event times for censored patients are treated
#          as parameters and sampled with a truncation constraint T[lower,].
#          Results are compared against the marginalization approach (brms)
#          to show both methods converge to the same posterior.
# DATA:    survival::lung — 228 patients with advanced lung cancer
# OUTPUT:  Console comparison of both approaches
#          augmentation_traceplots.png  — trace plots for key parameters
#          augmentation_latent.png      — posterior draws of imputed event times
#          method_comparison.png        — beta_sex posterior: augmentation vs marginalization
# =============================================================================

# ── Dependencies ──────────────────────────────────────────────────────────────
library(rstan)      # Direct Stan interface
library(survival)   # Provides the lung dataset
library(brms)       # For the marginalization fit (loaded for comparison)
library(posterior)  # as_draws_df, summarise_draws
library(bayesplot)  # mcmc_trace, mcmc_areas
library(ggplot2)    # Plotting and ggsave
library(dplyr)      # Data manipulation

# Stan options — use all available cores, cache compiled model
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# ── Data Preparation ──────────────────────────────────────────────────────────

data("lung")

# Drop rows with missing age or sex
lung <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]

# Recode sex: 1 = male (reference), 2 = female
# We store sex as a 0/1 numeric for Stan (0 = male, 1 = female)
lung$sex_num  <- as.numeric(lung$sex == 2)

# Censoring indicator: status == 1 means censored, status == 2 means died
lung$censored <- ifelse(lung$status == 1, "right", "none")

# Split into observed (died) and censored (still alive at study end)
idx_obs  <- which(lung$status == 2)   # observed event times
idx_cens <- which(lung$status == 1)   # censoring times (lower bounds only)

cat("Total patients:", nrow(lung), "\n")
cat("Observed deaths:", length(idx_obs), "\n")
cat("Censored (alive at study end):", length(idx_cens),
    paste0("(", round(100 * length(idx_cens) / nrow(lung), 1), "%)"), "\n\n")

# ── Stan Data List ─────────────────────────────────────────────────────────────
#
# Stan needs the observed and censored patients passed separately because they
# contribute different terms to the likelihood:
#   - Observed:  full Weibull density f(t_i | theta)
#   - Censored:  latent t_cens[i] ~ Weibull T[c_i, ] (truncated below by c_i)

# ── Stan Data List ────────────────────────────────────────────────────────────
# Center age so the intercept is interpretable at mean age
# and initialization is numerically stable
mean_age <- mean(lung$age)
age_centered_obs  <- lung$age[idx_obs]  - mean_age
age_centered_cens <- lung$age[idx_cens] - mean_age

stan_data <- list(
  N_obs        = length(idx_obs),
  N_cens       = length(idx_cens),
  t_obs        = lung$time[idx_obs],
  t_cens_lower = lung$time[idx_cens],
  X_obs        = cbind(age_centered_obs,  lung$sex_num[idx_obs]),
  X_cens       = cbind(age_centered_cens, lung$sex_num[idx_cens])
)

# ── Stan Model Code ───────────────────────────────────────────────────────────
# TRUE data augmentation: t_cens declared in parameters block,
# sampled with T[lower,] truncation enforcing t_cens[i] > c_i
stan_code <- "
data {
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  vector<lower=0>[N_obs]  t_obs;
  vector<lower=0>[N_cens] t_cens_lower;
  matrix[N_obs,  2] X_obs;
  matrix[N_cens, 2] X_cens;
}
parameters {
  real          alpha;
  vector[2]     beta;
  real<lower=0> shape;
  
  // FIX 1: Strict lower bound using the censoring threshold
  vector<lower=t_cens_lower>[N_cens] t_cens; 
}
model {
  alpha ~ normal(5.9, 1);
  beta  ~ normal(0, 0.5);
  shape ~ gamma(2, 2);

  // Observed patients: full Weibull density
  t_obs ~ weibull(shape, exp(alpha + X_obs * beta));

  // Censored patients: Evaluate the base density! 
  // FIX 2: No T[...] truncation needed. The bounds in the parameters 
  // block handle the domain restriction natively.
  for (i in 1:N_cens) {
    t_cens[i] ~ weibull(shape, exp(alpha + X_cens[i] * beta));
  }
}
generated quantities {
  real med_male   = exp(alpha)           * pow(log(2), 1.0 / shape);
  real med_female = exp(alpha + beta[2]) * pow(log(2), 1.0 / shape);
}
"

# ── Initial values ─────────────────────────────────────────────────────────── 
# t_cens initialized at 2x the censoring threshold — safely above T[lower,]
init_fn <- function() {
  list(
    alpha  = 5.9,
    beta   = c(0.0, 0.0),
    shape  = 1.2,
    t_cens = stan_data$t_cens_lower * 2.0
  )
}

# ── Fit ───────────────────────────────────────────────────────────────────────
fit_augmentation <- stan(
  model_code = stan_code,
  data       = stan_data,
  chains     = 4,
  iter       = 2000,
  warmup     = 1000,
  seed       = 5440,
  refresh    = 500,
  init       = init_fn
)
# ── Diagnostics ───────────────────────────────────────────────────────────────

cat("\n===== DIAGNOSTICS: DATA AUGMENTATION =====\n")

# Extract diagnostics — focus on the structural parameters, not the 63 latent times
params_of_interest <- c("alpha", "beta[1]", "beta[2]", "shape",
                        "med_male", "med_female")

print(fit_augmentation, pars = params_of_interest,
      probs = c(0.025, 0.5, 0.975))

# Check R-hat and divergences
sampler_params <- get_sampler_params(fit_augmentation, inc_warmup = FALSE)
divergences    <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
cat("\nTotal divergences:", divergences, "\n")

rhats_aug <- summary(fit_augmentation)$summary[params_of_interest, "Rhat"]
cat("Max Rhat (structural params):", round(max(rhats_aug, na.rm = TRUE), 4), "\n")

# ── Trace Plots ───────────────────────────────────────────────────────────────
#
# Trace plots for the four structural parameters.
# We deliberately exclude the 63 latent t_cens parameters from the trace plot
# — they are high-dimensional and their mixing is harder to assess visually,
# but convergence of alpha, beta, shape implies the latent times are also mixing.

draws_array <- as.array(fit_augmentation,
                        pars = c("alpha", "beta[1]", "beta[2]", "shape"))

p_trace_aug <- mcmc_trace(draws_array) +
  ggtitle("Trace Plots — Weibull Data Augmentation",
          subtitle = "4 chains, 1000 post-warmup draws each") +
  theme_minimal()

ggsave("augmentation_traceplots.png", p_trace_aug,
       width = 10, height = 6, dpi = 150)
cat("\nSaved: augmentation_traceplots.png\n")

# ── Posterior of Imputed Event Times ─────────────────────────────────────────
#
# This plot is unique to data augmentation — marginalization never produces
# posterior draws of the censored patients' event times.
# We show the posterior median and 95% CrI for each of the 63 latent times,
# ordered by their censoring threshold c_i.

draws_df  <- as.data.frame(fit_augmentation)
cens_cols <- grep("^t_cens\\[", names(draws_df), value = TRUE)

latent_summary <- data.frame(
  patient_id    = seq_along(idx_cens),
  cens_lower    = stan_data$t_cens_lower,
  post_median   = sapply(cens_cols, function(col) median(draws_df[[col]])),
  post_lower    = sapply(cens_cols, function(col) quantile(draws_df[[col]], 0.025)),
  post_upper    = sapply(cens_cols, function(col) quantile(draws_df[[col]], 0.975))
) %>% arrange(cens_lower)

p_latent <- ggplot(latent_summary,
                   aes(x = reorder(patient_id, cens_lower))) +
  geom_errorbar(aes(ymin = post_lower, ymax = post_upper),
                color = "steelblue", alpha = 0.5, width = 0) +
  geom_point(aes(y = post_median), color = "steelblue", size = 1) +
  geom_point(aes(y = cens_lower), color = "tomato", size = 1, shape = 4) +
  labs(
    title    = "Posterior Distributions of Latent Event Times",
    subtitle = "Blue = posterior median + 95% CrI  |  Red cross = censoring threshold c_i",
    x        = "Censored patients (ordered by censoring time)",
    y        = "Survival time (days)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

ggsave("augmentation_latent.png", p_latent,
       width = 10, height = 5, dpi = 150)
cat("Saved: augmentation_latent.png\n")

# ── Method Comparison: Augmentation vs Marginalization ────────────────────────
#
# The key result: both methods should give virtually identical posteriors
# for the structural parameters (alpha, beta, shape).
# We compare beta[2] (the sex coefficient) across both approaches.
#
# NOTE: this block requires fit_weibull from censored_data_analysis.R.
# Run that script first, or load the saved fit object if cached.

if (exists("fit_weibull")) {
  
  draws_marg <- as_draws_df(fit_weibull)
  beta_sex_marg <- draws_marg$b_sex2
  
  beta_sex_aug  <- draws_df[["beta[2]"]]
  
  compare_df <- data.frame(
    beta_sex = c(beta_sex_marg, beta_sex_aug),
    Method   = c(rep("Marginalization (brms)", length(beta_sex_marg)),
                 rep("Data Augmentation (Stan)", length(beta_sex_aug)))
  )
  
  p_compare <- ggplot(compare_df, aes(x = beta_sex, fill = Method)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("steelblue", "tomato")) +
    labs(
      title    = "Posterior of beta_sex: Augmentation vs Marginalization",
      subtitle = "Both methods should produce identical posteriors",
      x        = "beta_sex (female vs male, log-time scale)",
      y        = "Density"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  ggsave("method_comparison.png", p_compare,
         width = 8, height = 5, dpi = 150)
  cat("Saved: method_comparison.png\n")
  
  # Numeric comparison
  cat("\n===== METHOD COMPARISON: beta_sex =====\n")
  cat("Marginalization — Mean:", round(mean(beta_sex_marg), 3),
      " 95% CrI:", round(quantile(beta_sex_marg, 0.025), 3),
      "-", round(quantile(beta_sex_marg, 0.975), 3), "\n")
  cat("Augmentation   — Mean:", round(mean(beta_sex_aug), 3),
      " 95% CrI:", round(quantile(beta_sex_aug, 0.025), 3),
      "-", round(quantile(beta_sex_aug, 0.975), 3), "\n")
  cat("\nIf these are close, both approaches are working correctly.\n")
  
} else {
  cat("\nNote: fit_weibull not found in environment.\n")
  cat("Run censored_data_analysis.R first to generate the comparison plot.\n")
}

cat("\n===== SUMMARY =====\n")
cat("Data augmentation introduces", length(idx_cens),
    "latent parameters (one per censored patient).\n")
cat("Marginalization uses 0 latent parameters.\n")
cat("Both methods target the same posterior for alpha, beta, shape.\n")
cat("Only augmentation provides posterior draws for imputed event times.\n")
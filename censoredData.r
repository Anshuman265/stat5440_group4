# =============================================================================
# STAT 5440 — Applied Bayesian Modeling (Spring 2026)
# Group 4 Final Project: Missing & Censored Data
# =============================================================================
# FILE:    censored_data_analysis.R
# PURPOSE: Bayesian survival analysis on the NCCTG lung cancer dataset using
#          Weibull and log-normal AFT models with right-censored outcomes.
#          Covers: model fitting, MCMC diagnostics, posterior summaries,
#          posterior predictive checks, and sensitivity to parametric family.
# DATA:    survival::lung — 228 patients with advanced lung cancer
#          Variables used: time (days), status (1=censored, 2=died), age, sex
# OUTPUT:  Console summaries for slides 43–46
#          traceplots.png, rhat.png, neff.png, ppc.png — figures for slides
# =============================================================================

# ── Dependencies ──────────────────────────────────────────────────────────────
library(brms)       # Bayesian regression via Stan; handles censored likelihoods
library(survival)   # Provides the lung dataset
library(posterior)  # Posterior draw manipulation (as_draws_df, as_draws_array)
library(bayesplot)  # MCMC diagnostic and PPC plots
library(ggplot2)    # Plot saving via ggsave

# ── Data Preparation ──────────────────────────────────────────────────────────

data("lung")

# Drop the 3 rows with missing age or sex — these are not censored,
# they are genuinely unrecorded and excluded per the analysis plan.
lung <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]

# Recode the censoring indicator for brms:
#   status == 1 → patient was alive at study end (right-censored)
#   status == 2 → patient died (observed event)
# brms cens() expects the strings "right" and "none".
lung$censored <- ifelse(lung$status == 1, "right", "none")

# Convert sex to a factor so brms creates the correct dummy variable.
# Level 1 = male (reference), Level 2 = female.
# The coefficient b_sex2 will represent the female vs. male log-time difference.
lung$sex <- factor(lung$sex)

# ── Model 1: Weibull AFT ──────────────────────────────────────────────────────
#
# Model: log(T_i) = alpha + beta_age * age_i + beta_sex * sex_i + sigma * eps_i
#        eps_i ~ Gumbel  =>  T_i ~ Weibull(shape, scale)
#
# The cens(censored) wrapper applies the correct likelihood contributions:
#   - Observed deaths:   full Weibull density f(t_i | theta)
#   - Censored patients: survival function S(c_i | theta) = P(T > c_i)
#
# Priors (on the log-time scale):
#   Intercept ~ N(log(365), 1)   — baseline median near 1 year
#   beta_age, beta_sex ~ N(0, 0.5^2) — weak shrinkage, direction-agnostic
#   shape ~ Gamma(2, 2)          — weakly informative, keeps shape > 0

fit_weibull <- brm(
  formula = time | cens(censored) ~ age + sex,
  family  = weibull(),
  data    = lung,
  prior   = c(
    prior(normal(5.9, 1),  class = "Intercept"),
    prior(normal(0, 0.5),  class = "b"),
    prior(gamma(2, 2),     class = "shape")
  ),
  chains  = 4,       # 4 independent chains for convergence assessment
  iter    = 2000,    # 2000 total iterations per chain
  warmup  = 1000,    # First 1000 discarded (adaptation/warmup phase)
  seed    = 5440,    # Reproducibility seed
  refresh = 500      # Print progress every 500 iterations
)

# ── Model 2: Log-normal AFT ───────────────────────────────────────────────────
#
# Identical formula and priors as Model 1, but the error distribution
# is normal on the log scale rather than Gumbel.
# This gives a non-monotone hazard (hazard peaks then decreases),
# which better captures long-term survivors.
# Used exclusively for the sensitivity analysis on slide 46.
#
# Note: log-normal has no shape parameter, so the Gamma prior is omitted.

fit_lognormal <- brm(
  formula = time | cens(censored) ~ age + sex,
  family  = lognormal(),
  data    = lung,
  prior   = c(
    prior(normal(5.9, 1),  class = "Intercept"),
    prior(normal(0, 0.5),  class = "b")
  ),
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  seed    = 5440,
  refresh = 500
)

# ── MCMC Diagnostics (Slide 43) ───────────────────────────────────────────────
#
# Three standard convergence checks for the Weibull model:
#   R-hat:      potential scale reduction factor; < 1.01 indicates convergence
#   Neff ratio: effective draws / total draws; > 0.1 is acceptable, > 0.4 is good
#   Divergences: non-zero count signals problematic posterior geometry

cat("\n===== DIAGNOSTICS: WEIBULL =====\n")

# Use brms:: explicitly to avoid namespace conflict with bayesplot::rhat
rhats <- brms::rhat(fit_weibull)
cat("Max Rhat:", round(max(rhats, na.rm = TRUE), 4), "\n")
cat("Any Rhat > 1.01:", any(rhats > 1.01, na.rm = TRUE), "\n")

neffs <- brms::neff_ratio(fit_weibull)
cat("Min Neff ratio:", round(min(neffs, na.rm = TRUE), 3), "\n")

np <- nuts_params(fit_weibull)
cat("Divergences:", sum(np$Value[np$Parameter == "divergent__"]), "\n")

# ── Posterior Summaries (Slide 44) ────────────────────────────────────────────
#
# Extracts the full brms posterior summary (mean, SD, 95% CrI, R-hat, ESS)
# and then computes posterior median survival times for males and females
# at the sample mean age.
#
# Median survival formula for Weibull AFT:
#   median(T) = scale * (log 2)^(1/shape)
#   where scale = exp(alpha + beta_age * age + beta_sex * sex)
# This transforms the log-time regression back to the original days scale.

cat("\n===== POSTERIOR SUMMARY: WEIBULL =====\n")
print(summary(fit_weibull))

# Compute at the sample mean age for a representative comparison
mean_age <- mean(lung$age)

# Extract all 4000 post-warmup draws as a data frame
draws_w <- as_draws_df(fit_weibull)

# Posterior draws of median survival: male (sex = 1, reference level)
med_male_w <- exp(
  draws_w$b_Intercept +
  draws_w$b_age * mean_age
) * (log(2))^(1 / draws_w$shape)

# Posterior draws of median survival: female (sex = 2, adds b_sex2)
med_female_w <- exp(
  draws_w$b_Intercept +
  draws_w$b_age * mean_age +
  draws_w$b_sex2
) * (log(2))^(1 / draws_w$shape)

cat("\nMedian survival MALE   (Weibull):", round(median(med_male_w), 1),
    " 95% CrI:", round(quantile(med_male_w, 0.025), 1),
    "-", round(quantile(med_male_w, 0.975), 1), "\n")

cat("Median survival FEMALE (Weibull):", round(median(med_female_w), 1),
    " 95% CrI:", round(quantile(med_female_w, 0.025), 1),
    "-", round(quantile(med_female_w, 0.975), 1), "\n")

# ── Posterior Predictive Check — Fraction Censored (Slide 45) ─────────────────
#
# Standard PPC is complicated by censoring: for censored patients, the observed
# value is only a lower bound on the true event time, so we cannot compare
# y_rep directly to y for those rows.
#
# Procedure:
#   1. Draw 4000 replicated survival times from the posterior predictive.
#   2. Apply the same administrative censoring rule: if a simulated survival
#      time exceeds the patient's actual follow-up window, it would have been
#      censored in the real study too.
#   3. Compare the fraction censored in simulated data vs. observed 27.6%.
#
# A well-calibrated model should produce a simulated censoring fraction
# whose 95% interval contains the observed fraction.

cat("\n===== PPC: FRACTION CENSORED =====\n")

# posterior_predict returns a matrix: rows = posterior draws, cols = patients
pp <- posterior_predict(fit_weibull)

# Each patient's follow-up time serves as their censoring threshold
followup <- lung$time

obs_cens_frac <- mean(lung$censored == "right")
cat("Observed fraction censored:", round(obs_cens_frac, 3), "\n")

# For each posterior draw, count how many simulated times exceed follow-up
sim_cens_frac <- apply(pp, 1, function(yrep) mean(yrep > followup))
cat("Posterior 95% interval for fraction censored:",
    round(quantile(sim_cens_frac, 0.025), 3), "-",
    round(quantile(sim_cens_frac, 0.975), 3), "\n")

# ── Sensitivity Analysis: Log-normal vs Weibull (Slide 46) ────────────────────
#
# The parametric family is an untestable modeling assumption — both Weibull
# and log-normal fit the observed data, but they make different extrapolations.
# We compare median survival and b_sex across both families.
# If the key conclusion (female sex -> longer survival, CrI excludes 0)
# holds under both, the result is robust to this modeling choice.
#
# Log-normal median formula:
#   median(T) = exp(mu)   where mu = alpha + beta_age * age + beta_sex * sex
# (No shape parameter needed — the median of a log-normal is just exp of the mean)

cat("\n===== POSTERIOR SUMMARY: LOG-NORMAL =====\n")
print(summary(fit_lognormal))

draws_ln <- as_draws_df(fit_lognormal)

med_male_ln <- exp(
  draws_ln$b_Intercept +
  draws_ln$b_age * mean_age
)

med_female_ln <- exp(
  draws_ln$b_Intercept +
  draws_ln$b_age * mean_age +
  draws_ln$b_sex2
)

cat("\nMedian survival MALE   (Log-normal):", round(median(med_male_ln), 1),
    " 95% CrI:", round(quantile(med_male_ln, 0.025), 1),
    "-", round(quantile(med_male_ln, 0.975), 1), "\n")

cat("Median survival FEMALE (Log-normal):", round(median(med_female_ln), 1),
    " 95% CrI:", round(quantile(med_female_ln, 0.025), 1),
    "-", round(quantile(med_female_ln, 0.975), 1), "\n")

# Side-by-side b_sex comparison across both families
cat("\nbeta_sex Weibull  — Mean:", round(mean(draws_w$b_sex2), 3),
    " 95% CrI:", round(quantile(draws_w$b_sex2, 0.025), 3),
    "-", round(quantile(draws_w$b_sex2, 0.975), 3), "\n")

cat("beta_sex Log-norm — Mean:", round(mean(draws_ln$b_sex2), 3),
    " 95% CrI:", round(quantile(draws_ln$b_sex2, 0.025), 3),
    "-", round(quantile(draws_ln$b_sex2, 0.975), 3), "\n")

# ── Diagnostic & PPC Figures ──────────────────────────────────────────────────
#
# Generates four PNG files for use in the LaTeX presentation.
# All files are saved to the working directory.
#
#   traceplots.png  → Slide 43: visual convergence check for key parameters
#   rhat.png        → Slide 43: dot plot of all R-hat values
#   neff.png        → Slide 43: dot plot of all Neff ratios
#   ppc.png         → Slide 45: posterior predictive density overlay

# 1. Trace plots — four chains should overlap like a fuzzy caterpillar
p_trace <- mcmc_trace(
  as_draws_array(fit_weibull),
  pars = c("b_Intercept", "b_age", "b_sex2", "shape")
) + ggtitle("Trace Plots — Weibull Model")

ggsave("traceplots.png", p_trace, width = 10, height = 6, dpi = 150)

# 2. R-hat plot — all values should cluster tightly below 1.01
p_rhat <- mcmc_rhat(brms::rhat(fit_weibull)) +
  ggtitle("R-hat Values — Weibull Model")

ggsave("rhat.png", p_rhat, width = 6, height = 4, dpi = 150)

# 3. Effective sample size ratio plot — values above 0.1 are acceptable
p_neff <- mcmc_neff(brms::neff_ratio(fit_weibull)) +
  ggtitle("Effective Sample Size Ratios — Weibull Model")

ggsave("neff.png", p_neff, width = 6, height = 4, dpi = 150)

# 4. PPC density overlay — simulated densities (light blue) should
#    closely match the observed density (dark line) in the bulk of the
#    distribution; divergence in the right tail reflects Weibull's
#    known underprediction of long survivors
p_ppc <- pp_check(fit_weibull, ndraws = 50) +
  ggtitle("Posterior Predictive Check — Weibull")

ggsave("ppc.png", p_ppc, width = 7, height = 4, dpi = 150)

cat("\nAll figures saved to working directory.\n")
cat("Files: traceplots.png, rhat.png, neff.png, ppc.png\n")
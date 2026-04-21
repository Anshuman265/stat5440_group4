# packages
library(brms)
library(survival)
library(dplyr)
library(tidyr)
library(ggplot2)
library(posterior)
library(bayesplot)
#-----------------------------
# 1) Prepare the lung dataset
#-----------------------------
data(lung, package = "survival")

lung2 <- lung %>%
  transmute(
    time = time,
    status = status,
    age = age,
    sex = factor(sex, levels = c(1, 2), labels = c("Male", "Female")),
    event = ifelse(status == 2, 1, 0),                       # 1 = observed event, 0 = censored
    censored = ifelse(status == 1, "right", "none")         # for brms::cens()
  ) %>%
  filter(complete.cases(.))

#-----------------------------
# 2) Priors
#-----------------------------
priors <- c(
  prior(normal(5.9, 1), class = "Intercept"),
  prior(normal(0, 0.5), class = "b"),
  prior(gamma(2, 2), class = "shape")
)

#-----------------------------
# 3) Correct model:
#    proper right-censoring
#-----------------------------
fit_marg <- brm(
  time | cens(censored) ~ age + sex,
  family = weibull(),
  data = lung2,
  prior = priors,
  chains = 4, iter = 2000, seed = 5440,
  cores = 4
)

#-----------------------------
# 4) Incorrect model:
#    treat censored time as if event happened there
#-----------------------------
fit_fixed <- brm(
  time ~ age + sex,
  family = weibull(),
  data = lung2,
  prior = priors,
  chains = 4, iter = 2000, seed = 5440,
  cores = 4
)

#=========================================================
# GRAPH 1: Compare posterior coefficients between models
#=========================================================

draws_marg <- as_draws_df(fit_marg) %>%
  transmute(
    method = "Marginalization",
    age = b_age,
    sexFemale = b_sexFemale
  )

draws_fixed <- as_draws_df(fit_fixed) %>%
  transmute(
    method = "Fixed",
    age = b_age,
    sexFemale = b_sexFemale
  )

coef_plot_data <- bind_rows(draws_marg, draws_fixed) %>%
  pivot_longer(cols = c(age, sexFemale), names_to = "term", values_to = "value") %>%
  group_by(method, term) %>%
  summarise(
    mean = mean(value),
    lower = quantile(value, 0.025),
    upper = quantile(value, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    term = recode(term,
                  age = "Age",
                  sexFemale = "Sex")
  )

p_coef <- ggplot(coef_plot_data,
                 aes(x = term, y = mean, ymin = lower, ymax = upper, color = method)) +
  geom_pointrange(position = position_dodge(width = 0.45), linewidth = 0.7) +
  geom_hline(yintercept = 0, linetype = 2, color = "gray40") +
  labs(
    title = "Posterior regression coefficients: Censoring vs Fixed",
    x = NULL,
    y = "Posterior Estimates with 95% CI",
    color = NULL
  ) +
  theme_bw(base_size = 12)

print(p_coef)

m_ppc <- pp_check(fit_marg, ndraws = 100) +
  ggtitle("Posterior Predictive Check — Marginalization")
f_ppc <- pp_check(fit_fixed, ndraws = 100) +
  ggtitle("Posterior Predictive Check — Fixed")
print(m_ppc)
print(f_ppc)

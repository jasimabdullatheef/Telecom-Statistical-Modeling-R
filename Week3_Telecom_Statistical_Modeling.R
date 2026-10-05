# ============================================================
# YuvaIntern - Week 3
# Statistical Analysis and Predictive Modeling using R
# Telecom Customer Churn Analysis
# ============================================================

# 1. Load packages
library(dplyr)
library(caret)
library(car)

# 2. Set working directory
# Update this path if your project folder is located elsewhere.
setwd("C:/Users/USER/OneDrive/Desktop/YuvaIntern_Week3_Telecom_Statistical_Modeling")

# 3. Load dataset
data <- read.csv("telecom_customer_churn.csv", stringsAsFactors = FALSE)

# Initial inspection
dim(data)
str(data)
summary(data)

# ============================================================
# 4. Data Cleaning
# ============================================================

# Convert blank character values to NA
data[] <- lapply(data, function(x) {
  if (is.character(x)) {
    x <- trimws(x)
    x[x == ""] <- NA
  }
  x
})

# Missing-value check
colSums(is.na(data))

# Business-meaningful missing values
data$Multiple.Lines[is.na(data$Multiple.Lines) & data$Phone.Service == "No"] <-
  "No Phone Service"

internet_cols <- c(
  "Internet.Type", "Online.Security", "Online.Backup",
  "Device.Protection.Plan", "Premium.Tech.Support",
  "Streaming.TV", "Streaming.Movies", "Streaming.Music",
  "Unlimited.Data"
)

for (col in internet_cols) {
  data[[col]][is.na(data[[col]]) & data$Internet.Service == "No"] <-
    "No Internet Service"
}

# Churn fields are not applicable to customers who did not churn
data$Churn.Category[is.na(data$Churn.Category) &
                      data$Customer.Status != "Churned"] <- "Not Applicable"

data$Churn.Reason[is.na(data$Churn.Reason) &
                    data$Customer.Status != "Churned"] <- "Not Applicable"

# Remaining numerical missing values: median imputation
numeric_cols <- names(data)[sapply(data, is.numeric)]

for (col in numeric_cols) {
  if (anyNA(data[[col]])) {
    data[[col]][is.na(data[[col]])] <-
      median(data[[col]], na.rm = TRUE)
  }
}

# Remaining character missing values
character_cols <- names(data)[sapply(data, is.character)]

for (col in character_cols) {
  if (anyNA(data[[col]])) {
    data[[col]][is.na(data[[col]])] <- "Unknown"
  }
}

# Verify missing values
colSums(is.na(data))

# Duplicate check
sum(duplicated(data))

# ============================================================
# 5. Correct Invalid Monthly Charge Values
# ============================================================

# Count invalid negative values
sum(data$Monthly.Charge < 0)

# Replace invalid negative values with NA
data$Monthly.Charge[data$Monthly.Charge < 0] <- NA

# Replace with median valid Monthly Charge
data$Monthly.Charge[is.na(data$Monthly.Charge)] <-
  median(data$Monthly.Charge, na.rm = TRUE)

# Verify
sum(data$Monthly.Charge < 0)

# ============================================================
# 6. Create Binary Churn Target
# ============================================================

# Keep only customers with Churned or Stayed status
model_data <- data %>%
  filter(Customer.Status %in% c("Churned", "Stayed"))

# Binary target: 1 = Churned, 0 = Stayed
model_data$Churn <- ifelse(model_data$Customer.Status == "Churned", 1, 0)

# Remove ID and unused target-related fields
model_data <- model_data %>%
  select(
    Customer.Status,
    Age,
    Number.of.Dependents,
    Number.of.Referrals,
    Tenure.in.Months,
    Monthly.Charge,
    Contract,
    Internet.Service,
    Paperless.Billing,
    Payment.Method,
    Churn
  )

# Convert categorical variables to factors
model_data$Customer.Status <- factor(model_data$Customer.Status)
model_data$Contract <- factor(model_data$Contract)
model_data$Internet.Service <- factor(model_data$Internet.Service)
model_data$Paperless.Billing <- factor(model_data$Paperless.Billing)
model_data$Payment.Method <- factor(model_data$Payment.Method)

# Inspect modeling data
str(model_data)
dim(model_data)

# Churn distribution
table(model_data$Churn)

# ============================================================
# 7. Descriptive Statistics
# ============================================================

summary(model_data[, c(
  "Age",
  "Number.of.Dependents",
  "Number.of.Referrals",
  "Tenure.in.Months",
  "Monthly.Charge"
)])

summary(model_data[, c(
  "Total.Charges",
  "Total.Revenue"
)])

# ============================================================
# 8. Hypothesis Testing
# ============================================================

# H0: Mean Monthly Charge is equal between Churned and Stayed.
# H1: Mean Monthly Charge differs between Churned and Stayed.

t.test(
  Monthly.Charge ~ Customer.Status,
  data = model_data
)

# H0: Mean Tenure is equal between Churned and Stayed.
# H1: Mean Tenure differs between Churned and Stayed.

t.test(
  Tenure.in.Months ~ Customer.Status,
  data = model_data
)

# ============================================================
# 9. Correlation Analysis
# ============================================================

correlation_data <- data[, c(
  "Age",
  "Number.of.Dependents",
  "Number.of.Referrals",
  "Tenure.in.Months",
  "Avg.Monthly.GB.Download",
  "Monthly.Charge",
  "Total.Charges",
  "Total.Revenue"
)]

correlation_matrix <- cor(
  correlation_data,
  use = "complete.obs"
)

round(correlation_matrix, 2)

# ============================================================
# 10. Normality Test
# ============================================================

set.seed(123)

monthly_charge_sample <- sample(
  model_data$Monthly.Charge,
  size = min(5000, nrow(model_data))
)

shapiro.test(monthly_charge_sample)

# ============================================================
# 11. Train/Test Split
# ============================================================

set.seed(123)

train_index <- createDataPartition(
  model_data$Churn,
  p = 0.80,
  list = FALSE
)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

dim(train_data)
dim(test_data)

# ============================================================
# 12. Logistic Regression Model
# ============================================================

logistic_model <- glm(
  Churn ~ Age +
    Number.of.Dependents +
    Number.of.Referrals +
    Tenure.in.Months +
    Monthly.Charge +
    Contract +
    Internet.Service +
    Paperless.Billing +
    Payment.Method,
  family = binomial,
  data = train_data
)

summary(logistic_model)

# ============================================================
# 13. Prediction
# ============================================================

predicted_probability <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

head(predicted_probability)

# Classification threshold = 0.50
predicted_churn <- ifelse(
  predicted_probability >= 0.50,
  1,
  0
)

head(predicted_churn)

# ============================================================
# 14. Confusion Matrix and Metrics
# ============================================================

confusion_matrix <- table(
  Actual = test_data$Churn,
  Predicted = predicted_churn
)

confusion_matrix

accuracy <- mean(
  predicted_churn == test_data$Churn
)

precision <- confusion_matrix[2, 2] /
  sum(confusion_matrix[, 2])

recall <- confusion_matrix[2, 2] /
  sum(confusion_matrix[2, ])

f1_score <- 2 * (precision * recall) /
  (precision + recall)

accuracy
precision
recall
f1_score

# ============================================================
# 15. Residual Diagnostics
# ============================================================

plot(
  logistic_model,
  which = 1,
  main = "Logistic Regression Residual Diagnostic"
)

# ============================================================
# 16. Multicollinearity - VIF
# ============================================================

vif_values <- vif(logistic_model)
vif_values

# ============================================================
# 17. Odds Ratios
# ============================================================

odds_ratios <- exp(coef(logistic_model))
round(odds_ratios, 3)

# ============================================================
# 18. End of Analysis
# ============================================================
# Key test results:
# Monthly Charge t-test: p < 2.2e-16
# Tenure t-test: p < 2.2e-16
# Shapiro-Wilk: W = 0.92189, p < 2.2e-16
#
# Model evaluation from the completed analysis:
# Accuracy  = 0.8405467
# Precision = 0.7457143
# Recall    = 0.6832461
# F1 Score  = 0.7131148
# ============================================================




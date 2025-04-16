import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# ============================
# Load the dataset
# ============================
# Adjust the path if necessary. Ensure that "movie_metadata.csv" is in your working directory.
df = pd.read_csv("movie_metadata.csv")

# ============================
# Research Question 1:
# “Is there a specific correlation between a movie's budget and its box office revenue?”
# (Regression Analysis)
# ============================

# Clean the data: drop missing values and filter for positive values in budget & gross
df_reg = df[['budget', 'gross']].dropna()
df_reg = df_reg[(df_reg['budget'] > 0) & (df_reg['gross'] > 0)]

# Calculate Pearson correlation (optional if you want to check linearity)
corr_value = df_reg['budget'].corr(df_reg['gross'])
print("Pearson correlation between Budget and Gross: {:.3f}".format(corr_value))

# Prepare data for regression: Add an intercept using statsmodels
X = sm.add_constant(df_reg['budget'])
y = df_reg['gross']

# Fit the simple linear regression model: gross ~ budget
model_reg = sm.OLS(y, X).fit()
print("\nRegression Summary for 'Gross ~ Budget':")
print(model_reg.summary())

# Plot the regression: Scatter plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(x='budget', y='gross', data=df_reg,
            scatter_kws={'s': 10},
            line_kws={'color': 'red'})
plt.title('Regression Analysis: Gross vs. Budget')
plt.xlabel('Budget')
plt.ylabel('Gross Revenue')
plt.tight_layout()
plt.show()

# ============================
# Research Question 2:
# “Is there a difference in the mean number of movie facebook likes, mean audience rating,
#  and mean critic rating?”
# (Repeated Measures ANOVA)
# ============================
#
# In this example we compare:
# - movie_facebook_likes (Facebook popularity of the movie)
# - imdb_score (as a proxy for audience rating)
# - num_critic_for_reviews (using the number of critic reviews as a proxy for critic rating)
#
# Since each movie has all three measures, we treat each movie as a subject
# in a within-subject (repeated measures) design.
#
# Create a long-format dataframe for repeated measures ANOVA.

# Select the relevant columns and drop any rows with missing values
df_anova = df[['movie_facebook_likes', 'imdb_score', 'num_critic_for_reviews']].dropna()

# Create a subject ID; here we use the DataFrame index (each movie is a subject)
df_anova = df_anova.reset_index().rename(columns={'index': 'subject'})

# Convert to long format: each row has one measurement and a factor indicating which measure it is.
df_long = pd.melt(
    df_anova,
    id_vars=['subject'],
    value_vars=['movie_facebook_likes', 'imdb_score', 'num_critic_for_reviews'],
    var_name='Measure',
    value_name='Value'
)

# Run the repeated measures ANOVA using statsmodels
aovrm = AnovaRM(df_long, depvar='Value', subject='subject', within=['Measure'])
res = aovrm.fit()
print("\nRepeated Measures ANOVA Results:")
print(res)

# Plot the means and distributions for each measure using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Measure', y='Value', data=df_long)
plt.title('Distribution of Movie Facebook Likes, IMDB Score, and Critic Reviews')
plt.xlabel('Measure')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Also, plot the means with standard error bars for a clearer comparison
plt.figure(figsize=(8, 6))
sns.pointplot(x='Measure', y='Value', data=df_long, capsize=.1)
plt.title('Mean Comparison Across Measures')
plt.xlabel('Measure')
plt.ylabel('Mean Value (with SE)')
plt.tight_layout()
plt.show()

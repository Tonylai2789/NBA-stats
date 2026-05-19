# AP Stats Final Project Summary

## Research Question

**Question:** Do NBA players perform worse in a game after a stretch of heavy recent playing time?

**Explanatory variable:** minutes played in the player's previous 5 games during the same season.

**Response variable:** Game Score (GmSc) in the current game. Game Score is Basketball-Reference's box-score performance index that combines scoring efficiency, rebounds, assists, steals, blocks, and turnovers into one number.

## Data Collection and Sampling Design

The project uses NBA regular-season player-game records from the 2021-22, 2022-23, and 2023-24 seasons. The database reference for the project is Wyatt Walsh's Basketball Dataset on Kaggle, and the local repository stores cached player-game logs so the analysis can be reproduced quickly.

Players were included in a season if they averaged at least 20.0 minutes per game and appeared in at least 60 games. For each qualifying player-season, I kept games in which the player actually played and recorded the game date, minutes played, and Game Score.

For each player-game, I created the explanatory variable by summing that player's minutes from the previous 5 games in the same season. The first five games of each player-season were removed because they did not yet have five previous games available. This produced **37,441 raw player-game rows** and **34,831 usable player-game rows**.

For inference, I used a **stratified random sample of non-overlapping 5-game blocks** instead of the full usable data set. Each player-season was treated as one stratum. Inside each stratum, the usable games were split into fixed non-overlapping blocks of five consecutive player-games. The script randomly selected **40 player-season strata** and then randomly selected one 5-game block from each selected stratum using random seed `2789`.

This produced **40 blocks** and **200 sampled player-game observations**. Since \(200 < 0.10(34,831) = 3,483.1\), the sample satisfies the 10% condition.

## Data Display

The main data display is a scatterplot of previous-5-game minutes versus current-game Game Score, using the 200 sampled player-games. The graph includes a least-squares regression line.

The script also creates a LaTeX table showing 12 example rows from the sampled data and a CSV containing all 200 sampled rows.

## Model and Hypotheses

The fatigue hypothesis predicts a negative relationship between recent workload and current-game performance, so the formal test is one-sided.

```text
GmSc = b0 + b1(MP_prev5) + e
```

**H0:** b1 = 0  
**HA:** b1 < 0

This tests whether player-games with heavier recent workload tend to have lower current-game Game Scores.

## Results

Sampled pooled regression:

- Full usable population: 34,831 player-games
- Random sample size: 200 player-games
- Sample design: 40 non-overlapping 5-game blocks
- Sample percent: 0.57% of the usable population
- Intercept: -4.0409
- Slope: +0.104427 Game Score points per additional previous-5-game minute
- t statistic: +5.804
- Degrees of freedom: 198
- One-sided p-value for HA: b1 < 0: approximately 1
- R-squared: 0.1454

The sampled regression slope is positive, not negative. Since the test was looking for evidence that the slope is less than 0, the sample gives no evidence for the fatigue hypothesis.

## Conditions for Regression Inference

**Linear:** The scatterplot of previous-5-game minutes and Game Score shows an approximately linear positive trend. The residual plot does not show a strong curved pattern, so the linear condition is reasonably met.

**Independent:** The sample contains 200 player-game observations, which is less than 10% of the 34,831 usable player-game population. The sample was selected as 40 non-overlapping blocks of five games, so no sampled player-game is repeated within a player-season. Observations within the same 5-game block may still be related because they come from the same player, so this is a limitation, but the non-overlapping block design is more defensible than using overlapping sliding-window rows.

**Normal Residuals:** The residual histogram is centered near 0 and does not show extreme skew. It is not perfectly normal, but with \(n = 200\), the regression t procedure is reasonably robust.

**Equal Variance:** The residual plot shows residuals scattered around 0 across the range of previous-5-game minutes. The spread is not perfectly constant, but there is no severe cone or fan shape, so the equal variance condition is reasonably met.

**Random:** The inference data were produced using a random sampling procedure. Player-season strata were randomly selected, and one non-overlapping 5-game block was randomly selected from each chosen stratum using a fixed seed for reproducibility.

## Conclusion

At alpha = 0.05, I fail to reject the null hypothesis. The stratified random non-overlapping block sample does not provide statistically significant evidence that heavier recent workload is associated with lower Game Scores. In fact, the observed relationship in the sample is positive.

## Discussion of Bias and Limitations

The positive relationship is probably influenced by lurking variables:

- **Coach selection and current form:** coaches give more minutes to players who are playing well, so heavy recent minutes can be a sign that a player is in good form.
- **Injury and return-to-play effects:** players coming back from injury may have low recent minutes and lower performance at the same time.
- **Schedule context:** rest days, back-to-backs, travel, and opponent strength were not controlled.

A true fatigue effect may still exist, but this project does not isolate it from those other factors.

## Files Produced

- `make_data_display.py`: creates the non-overlapping 5-game block sample, sampled scatterplot, LaTeX table, CSV, and diagnostics
- `sampled_workload_table.csv`: complete 200-row sampled data set
- `sampled_workload_scatterplot.png`: scatterplot and least-squares line for sampled data
- `sampled_raw_data_table.tex`: LaTeX table with 12 example sampled rows
- `sampled_residual_plot.png`: residual plot used to check linearity and equal variance
- `sampled_residual_histogram.png`: residual histogram used to check normal residuals

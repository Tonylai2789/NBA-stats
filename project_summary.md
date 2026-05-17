# AP Stats Final Project Summary

## Research Question

**Question:** Do NBA players perform worse in a game after a stretch of heavy recent playing time?

**Explanatory variable:** minutes played in the player's previous 5 games during the same season.

**Response variable:** Game Score (GmSc) in the current game. Game Score is Basketball-Reference's box-score performance index that combines scoring efficiency, rebounds, assists, steals, blocks, and turnovers into one number.

## Data Collection

The project uses NBA regular-season player-game records from the 2021-22, 2022-23, and 2023-24 seasons. The database reference for the project is Wyatt Walsh's Basketball Dataset on Kaggle, and the local repository stores cached player-game logs so the analysis can be reproduced quickly. Players were included in a season if they averaged at least 20.0 minutes per game and appeared in at least 60 games. For each qualifying player-season, the game log was collected and games not played were removed.

For each player-game, I recorded the game date, minutes played, and Game Score. I then created the explanatory variable by summing that player's minutes from the previous 5 games in the same season. This produced **37,441 raw player-game rows** and **34,831 usable rows** after the first five games of each player-season were removed for the rolling workload calculation.

## Models and Hypotheses

The fatigue hypothesis predicts a negative relationship between recent workload and current-game performance, so the formal tests are one-sided.

### Model A: Pooled Simple Linear Regression

The pooled model treats each player-game as one observation:

```text
GmSc = b0 + b1(MP_prev5) + e
```

**H0:** b1 = 0  
**HA:** b1 < 0

This tests whether player-games with heavier recent workload tend to have lower current-game Game Scores.

### Model B: Within-Player Slope Analysis

The within-player model separates each player from the rest of the data:

1. For each player with at least 50 usable games, fit a simple linear regression using only that player's games.
2. Record that player's slope.
3. Run a one-sample t-test on the player slopes.

**H0:** mean player slope = 0  
**HA:** mean player slope < 0

This better matches the fatigue question because it asks whether the same player tends to perform worse after heavier-than-usual recent minutes.

## Results

### Model A: Pooled Regression

- Observations: 34,831 player-games
- Slope: +0.1060 Game Score points per additional previous-5-game minute
- t statistic: +80.93
- One-sided p-value for HA: b1 < 0: approximately 1
- R-squared: 0.158
- Pearson r: +0.398

The pooled regression slope is positive, not negative. This is strong evidence against the original fatigue direction in the pooled data.

### Model B: Within-Player Slopes

- Players used: 290
- Mean slope: +0.0323 Game Score points per additional previous-5-game minute
- Standard deviation of slopes: 0.0541
- Median slope: +0.0370
- Percent of players with negative slopes: 21.0%
- t statistic: +10.15 with 289 degrees of freedom
- One-sided p-value for HA: mean slope < 0: approximately 1

Even after focusing on within-player changes, most players had positive slopes. The data do not support the claim that heavier recent workload lowers Game Score.

## Conclusion

At alpha = 0.05, I fail to reject the null hypothesis in both analyses. The data do not provide statistically significant evidence that heavier recent workload is associated with lower Game Scores. In this sample, the relationship is actually positive.

## Discussion of Bias and Limitations

The positive relationship is probably influenced by lurking variables:

- **Coach selection and current form:** coaches give more minutes to players who are playing well, so heavy recent minutes can be a sign that a player is in good form.
- **Injury and return-to-play effects:** players coming back from injury may have low recent minutes and lower performance at the same time.
- **Schedule context:** rest days, back-to-backs, travel, and opponent strength were not controlled.

A true fatigue effect may still exist, but this project does not isolate it from those other factors.

## Inference Conditions

For the pooled regression, the scatterplot and binned means suggest a roughly linear trend. The large sample size makes the inference fairly stable, but independence is imperfect because the same players appear repeatedly. That is why the within-player analysis is included.

For the within-player t-test, the observations are player-level slopes, which are more independent than individual games. With 290 slopes, the sampling distribution of the mean slope should be close to normal.

## Files Produced

- `nba_workload.py`: full data collection and analysis script
- `make_data_display.py`: reproducible script for the scatterplot and LaTeX raw-data table
- `raw_data_table.tex`: sample raw data table for the written report
- `recent_workload_scatterplot.png`: requested scatterplot display
- `workload_vs_gamescore.png`: original pooled regression plot with binned means
- `within_player_slopes.png`: histogram of within-player slopes
- `per_player_slopes.csv`: player-level slope table

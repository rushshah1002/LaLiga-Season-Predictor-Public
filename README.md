# LaLiga-Season-Predictor-Public

Fits a Poisson team-strength model to La Liga results and scores it against de-vigged
bookmaker odds on a held-out season.

## Approach

Goals are Poisson with team-level attack and defense terms on a log link:

log(lambda_home) = mu + home_adv + attack[home] + defense[away]

log(lambda_away) = mu + attack[away] + defense[home]

Attack and defense are centered to sum to zero. Fit by L2-penalized maximum likelihood with L-BFGS-B. Match probabilities come from
summing the scoreline grid out to 10 goals a side.

Bookmaker odds get inverted to implied probabilities and normalized to sum to one.

## Results

Seasons 2019/20 through 2024/25 from football-data.co.uk, 2,280 matches. Train on 2019/20
through 2022/23, validate on 2023/24, test on 2024/25. Model is fit on train + val.

Test season, 380 matches:

Log loss: 0.9911 model, 0.9532 market

Brier: 0.5908 model, 0.5639 market

Average overround on the odds is about 5%. The model trails the market by roughly 4% on log
loss. Bootstrap resampling over the training set puts a CI on that gap and calibration curves
show predicted probabilities tracking observed frequencies reasonably closely.

## Caveats

One test season, 380 matches. Model sees goals only, nothing about injuries, lineups, red cards, or form. Average best EV is +0.21 a match but that's max-EV selection across three outcomes, not edge. No realized-profit backtest. Team strengths are frozen after training and nothing updates them during the test season.

## Running it

pip install pandas numpy scipy scikit-learn matplotlib
python -m src.pipeline

Run from the project root. Seasons are set by `DEFAULT_SEASONS` in `data_loader.py`; adding
"2526" pulls the current season and turns on the prediction branch in the pipeline.

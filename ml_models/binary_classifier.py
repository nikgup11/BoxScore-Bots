import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# if you ever use GPU, these help determinism
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


def get_model_and_prediction(player_file):
    # import data
    data = pd.read_csv('./data-collection/cleaned_data/' + player_file)

    # features (previous game)
    features = [
        'MP', 'FGA', '3PA', 'FTA',
        'FG%', '3P%', '2P%', 'eFG%', 'FT%',
        'TRB', 'AST', 'TOV',
        'GmSc', '+/-', 'PTS', 'is_home', 'result_win',
        'OppOffRtg', 'OppDefRtg', 'OppNetRtg', 'OppPace'
    ]

    # create TARGET from *current* game points (before shifting)
    current_season = data['SEASON_END'].max()
    season_mask = data['SEASON_END'] == current_season
    current_ppg = data.loc[season_mask, 'PTS'].mean()

    data['above_ppg'] = (data['PTS'] > current_ppg).astype(float)
    target = ['above_ppg']

    # regression target: current game points
    data['pts_target'] = data['PTS']

    # now shift features so they refer to the *previous* game
    data[features] = data[features].shift(1)

    # drop first row and rows with NaNs
    data = data.dropna(subset=features + target + ['pts_target'])

    # >>> MOVE the last_game_features_np here, after dropna <<<
    last_game_features_np = (
        data[features].iloc[-1].values.astype(np.float32).reshape(1, -1)
    )

    # numpy arrays
    X_np = data[features].values
    y_np = data[target].values
    y_pts_np = data[['pts_target']].values

    # chronological split: last 30 games as test
    # When holdout_for_eval=True, the last n_test games are kept ONLY for evaluation.
    # When holdout_for_eval=False, all games (including those last n_test) are used for training.
    holdout_for_eval = False  # set to False when training on all history to predict the next game
    n_test = 30
    if holdout_for_eval and n_test > 0:
        X_train_np, X_test_np = X_np[:-n_test], X_np[-n_test:]
        y_train_np, y_test_np = y_np[:-n_test], y_np[-n_test:]
        y_pts_train_np, y_pts_test_np = y_pts_np[:-n_test], y_pts_np[-n_test:]
    else:
        # train on all available games
        X_train_np, y_train_np = X_np, y_np
        y_pts_train_np = y_pts_np
        # still define a test slice (e.g., last n_test games) for inspection if desired
        if n_test > 0:
            X_test_np, y_test_np = X_np[-n_test:], y_np[-n_test:]
            y_pts_test_np = y_pts_np[-n_test:]
        else:
            # fall back to using the last game as a "test" point
            X_test_np, y_test_np = X_np[[-1]], y_np[[-1]]
            y_pts_test_np = y_pts_np[[-1]]

    print(len(X_train_np))
    print(len(X_test_np))

    # to torch
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    y_pts_train = torch.tensor(y_pts_train_np, dtype=torch.float32)
    y_pts_test = torch.tensor(y_pts_test_np, dtype=torch.float32)

    # define model, loss function, and optimizer (strict binary classifier)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 1),
        nn.Sigmoid()
    )

    # per-sample BCE so we can apply recency weights
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    epochs = 1000
    lambda_penalty = 0.8  # tune this (start small like 0.1–0.5)

    # recency weights: older games get lower weight, recent games higher
    n_train = X_train.shape[0]
    idx = torch.arange(n_train, dtype=torch.float32)
    # weights in [0.5, 1.0]; adjust endpoints as desired
    sample_weights = 0.5 + 0.5 * (idx / (n_train - 1))
    sample_weights = sample_weights.view(-1, 1)  # (n_train, 1) to broadcast with probs

    for _ in range(epochs):
        optimizer.zero_grad()
        probs = model(X_train)

        # base BCE loss (per-sample), then weighted by recency
        bce_per_sample = criterion(probs, y_train)
        bce = (bce_per_sample * sample_weights).mean()

        # confidence in [0,1]: 0 at 0.5, 1 near 0 or 1
        confidence = torch.abs(probs - 0.5) * 2.0

        # wrong predictions: 1 if wrong, 0 if correct
        hard_preds = (probs >= 0.5).float()
        wrong = torch.abs(hard_preds - y_train)

        # extra penalty:
        #   - all wrong predictions get weighted by confidence AND recency
        #   - WRONG & very high-confidence get an extra (confidence^2) penalty, also recency-weighted
        high_conf_mask = confidence > 0.8
        if high_conf_mask.any():
            high_conf_wrong = (
                confidence[high_conf_mask] ** 2
                * wrong[high_conf_mask]
                * sample_weights[high_conf_mask]
            ).mean()
        else:
            high_conf_wrong = torch.tensor(0.0, device=probs.device)

        conf_wrong_penalty = (confidence * wrong * sample_weights).mean() + high_conf_wrong

        loss = bce + lambda_penalty * conf_wrong_penalty
        loss.backward()
        optimizer.step()

    print("Classifier weights:", model[0].weight.data)
    print("Classifier bias:", model[0].bias.data)

    # --- projection for the NEXT game (using last game's stats) ---
    X_next = torch.tensor(last_game_features_np, dtype=torch.float32)
    with torch.no_grad():
        prob_next = model(X_next).item()
    print(f"\nNext game: P(above {current_ppg:.1f} PPG) = {prob_next:.3f}")

    # evaluation on test set
    with torch.no_grad():
        y_pred = model(X_test)

    def interpret_prediction(prob):
        if prob >= 0.80:
            return "High confidence above average"
        elif prob >= 0.60:
            return "Moderate confidence above average"
        elif prob > 0.40:
            return "Uncertain, could be average"
        elif prob <= 0.40:
            return "Moderate confidence below average"
        elif prob <= 0.20:
            return "High confidence below average"
        else:
            return "High confidence below average"

    labels = []
    correct = []

    for i, prob in enumerate(y_pred):
        p = prob.item()
        lbl = interpret_prediction(p)
        labels.append(lbl)

        # model's hard prediction: above if p >= 0.5
        pred_cls = 1.0 if p >= 0.5 else 0.0
        true_cls = y_test[i].item()
        correct.append(1 if pred_cls == true_cls else 0)

    labels = np.array(labels)
    correct = np.array(correct)

    # counters for right and wrong decisions (overall)
    total_predictions = len(correct)
    total_correct = int(correct.sum())
    total_wrong = int(total_predictions - total_correct)
    print(f"\nModel decisions: {total_correct} correct, {total_wrong} wrong out of {total_predictions}")

    # split counters by confidence level (high, moderate, low)
    high_total = high_correct = 0
    moderate_total = moderate_correct = 0
    low_total = low_correct = 0

    for lbl, is_corr in zip(labels, correct):
        if lbl.startswith("High confidence"):
            high_total += 1
            if is_corr:
                high_correct += 1
        elif lbl.startswith("Moderate confidence"):
            moderate_total += 1
            if is_corr:
                moderate_correct += 1
        else:  # treat everything else as low confidence / uncertain
            low_total += 1
            if is_corr:
                low_correct += 1

    def _print_conf_summary(name, total, correct_):
        if total == 0:
            print(f"{name}: no examples")
        else:
            wrong_ = total - correct_
            acc = correct_ / total
            print(f"{name}: {correct_} correct, {wrong_} wrong out of {total} (acc={acc:5.2f})")

    print("\nBy confidence level:")
    _print_conf_summary("High confidence", high_total, high_correct)
    _print_conf_summary("Moderate confidence", moderate_total, moderate_correct)
    _print_conf_summary("Low confidence", low_total, low_correct)

    print("\nCalibration per verbal label:")
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        n = mask.sum()
        acc = correct[mask].mean() if n > 0 else float("nan")
        print(f"{lbl:35s} n={n:2d}, accuracy={acc:5.2f}")

    # optionally return the trained classifier and current_ppg threshold
    return model, current_ppg


if __name__ == "__main__":
    get_model_and_prediction('gilgesh01_Shai_Gilgeous-Alexander_last3.csv')
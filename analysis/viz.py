from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from src.evaluation import calibration_data_binary

def plot_calibration_curves(model_probs, market_probs, y_true, outcome_names=None, figsize=(15, 5)):
    if outcome_names is None:
        outcome_names = ['Home Win', 'Draw', 'Away Win']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for outcome_idx in range(3):
        ax = axes[outcome_idx]
        y_binary = (y_true == outcome_idx).astype(int)

        try:
            mp, fp = calibration_data_binary(model_probs[:, outcome_idx], y_binary, n_bins=10)
            ax.plot(mp, fp, 'o-', linewidth=2, label='Model', color='#2E86AB', markersize=8)
        except ValueError:
            pass
        try:
            mp, fp = calibration_data_binary(market_probs[:, outcome_idx], y_binary, n_bins=10)
            ax.plot(mp, fp, 's-', linewidth=2, label='Market', color='#A23B72', markersize=8)
        except ValueError:
            pass

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{outcome_names[outcome_idx]} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig

def plot_team_strengths(model, figsize=(12, 8)):
    if not model.fitted_:
        raise ValueError("Model must be fitted first")

    teams = model.teams_
    mu, ha, attack, defense = model._unpack(model.params_, len(teams))
    df = pd.DataFrame({'Team': teams, 'Attack': attack, 'Defense': -defense}).sort_values('Attack', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors_attack = ['#27ae60' if x > 0 else '#e74c3c' for x in df['Attack']]
    axes[0].barh(df['Team'], df['Attack'], color=colors_attack, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=2)
    axes[0].set_xlabel('Attack Strength (log scale)')
    axes[0].set_title('Team Attack Strengths')
    axes[0].grid(True, alpha=0.3, axis='x')

    df_def = df.sort_values('Defense', ascending=False)
    colors_def = ['#27ae60' if x > 0 else '#e74c3c' for x in df_def['Defense']]
    axes[1].barh(df_def['Team'], df_def['Defense'], color=colors_def, alpha=0.8, edgecolor='black')
    axes[1].axvline(x=0, color='black', linewidth=2)
    axes[1].set_xlabel('Defense Strength (log scale, flipped)')
    axes[1].set_title('Team Defense Strengths')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig

def plot_ev_distribution(test_df, figsize=(14, 5)):
    evs = test_df['best_ev'].values
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].hist(evs, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero EV')
    axes[0].axvline(x=np.mean(evs), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(evs):.4f}')
    axes[0].set_xlabel('Expected Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Best EV per Match')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(np.cumsum(evs), linewidth=2, color='#9b59b6')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Match Number')
    axes[1].set_ylabel('Cumulative EV')
    axes[1].set_title('Cumulative Expected Value')
    axes[1].grid(True, alpha=0.3)

    positive_ev, negative_ev = evs[evs > 0], evs[evs <= 0]
    axes[2].pie([len(positive_ev), len(negative_ev)],
                labels=[f'Positive EV\n({len(positive_ev)} matches)', f'Non-positive EV\n({len(negative_ev)} matches)'],
                colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Positive vs Non-positive EV')

    plt.tight_layout()
    return fig

def plot_model_vs_market(test_df, figsize=(15, 5)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    outcomes = ['home', 'draw', 'away']
    titles = ['Home Win', 'Draw', 'Away Win']
    colors = ['#3498db', '#f39c12', '#e74c3c']

    for i, (outcome, title, color) in enumerate(zip(outcomes, titles, colors)):
        x = test_df[f'market_p_{outcome}'].values
        y = test_df[f'p_{outcome}_model'].values

        axes[i].scatter(x, y, alpha=0.5, s=30, color=color, edgecolor='black', linewidth=0.5)
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='y=x')

        corr = np.corrcoef(x, y)[0, 1]
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[i].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[i].set_xlabel('Market Probability')
        axes[i].set_ylabel('Model Probability')
        axes[i].set_title(f'{title} Probability Comparison')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    plt.tight_layout()
    return fig

def plot_performance_by_favorite(test_df, figsize=(12, 5)):
    test_df = test_df.copy()
    odds_array = test_df[['odds_home', 'odds_draw', 'odds_away']].values
    favorite_idx = np.argmin(odds_array, axis=1)
    test_df['favorite'] = favorite_idx

    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    test_df['model_agrees'] = (favorite_idx == np.argmax(model_probs, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    agree_rate = test_df['model_agrees'].mean()
    axes[0].pie([agree_rate * 100, (1 - agree_rate) * 100],
                labels=[f'Agree\n({agree_rate:.1%})', f'Disagree\n({1-agree_rate:.1%})'],
                colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Model-Market Agreement on Favorite')

    agree_subset = test_df[test_df['model_agrees']].copy()
    disagree_subset = test_df[~test_df['model_agrees']].copy()
    agree_subset['favorite_won'] = (agree_subset['y_true'] == agree_subset['favorite'])
    disagree_subset['favorite_won'] = (disagree_subset['y_true'] == disagree_subset['favorite'])

    agree_win_rate = agree_subset['favorite_won'].mean() if len(agree_subset) > 0 else 0
    disagree_win_rate = disagree_subset['favorite_won'].mean() if len(disagree_subset) > 0 else 0

    bars = axes[1].bar(['Model Agrees\nwith Market', 'Model Disagrees\nwith Market'],
                        [agree_win_rate * 100, disagree_win_rate * 100],
                        color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Favorite Win Rate (%)')
    axes[1].set_title('Favorite Win Rate by Agreement')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [agree_win_rate * 100, disagree_win_rate * 100]):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.1f}%',
                     ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def create_summary_report(results: dict, save_path: Optional[str] = None):
    test_df = results['test_df']
    model = results['model']

    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    market_probs = test_df[['market_p_home', 'market_p_draw', 'market_p_away']].values
    y_true = test_df['y_true'].values

    fig1 = plot_calibration_curves(model_probs, market_probs, y_true)
    fig2 = plot_team_strengths(model)
    fig3 = plot_ev_distribution(test_df)
    fig4 = plot_model_vs_market(test_df)
    fig5 = plot_performance_by_favorite(test_df)

    if save_path:
        fig1.savefig(f'{save_path}_calibration.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{save_path}_strengths.png', dpi=300, bbox_inches='tight')
        fig3.savefig(f'{save_path}_ev_dist.png', dpi=300, bbox_inches='tight')
        fig4.savefig(f'{save_path}_model_vs_market.png', dpi=300, bbox_inches='tight')
        fig5.savefig(f'{save_path}_performance.png', dpi=300, bbox_inches='tight')

    return [fig1, fig2, fig3, fig4, fig5]

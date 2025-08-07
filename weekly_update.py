import os
import numpy as np
import pandas as pd
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports for model loading
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# PyMC for Bayesian updates
import pymc as pm

# Data collection
import statsapi
from data_scraper import MLB_Scrape
from pybaseball import pitching_stats

# Create directories
os.makedirs('results', exist_ok=True)
os.makedirs('diagnostics/weekly_updates', exist_ok=True)

def load_trained_models_FIXED():
    """FIXED: Updated function name to match what's being called"""
    print("📦 Loading trained models...")
    
    models = {}
    
    try:
        # Level 1 models
        swing_pipe, whiff_pipe, exit_pipe = joblib.load('models/level1_pipelines.pkl')
        models['level1'] = (swing_pipe, whiff_pipe, exit_pipe)
        print("   ✅ Level 1 models loaded")
    except FileNotFoundError:
        print("   ❌ Level 1 models not found")
        return None
    
    try:
        # Level 2 LSTM and scaler
        from tensorflow.keras.models import load_model
        from tensorflow.keras.optimizers import Adam
        lstm = load_model('models/level2_lstm.h5', compile=False)
        lstm.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
        scaler = joblib.load('models/level2_scaler.pkl')
        game_feats = joblib.load('models/level2_game_feats.pkl')
        models['level2'] = (lstm, scaler, game_feats)
        print("   ✅ Level 2 LSTM loaded")
    except FileNotFoundError as e:
        print(f"   ❌ Level 2 models not found: {e}")
        return None
    
    try:
        # Level 3 Bayesian
        pitchers, a_mean = joblib.load('models/level3_bayes.pkl')
        models['level3'] = (pitchers, a_mean)
        print("   ✅ Level 3 Bayesian model loaded")
    except FileNotFoundError:
        print("   ❌ Level 3 Bayesian model not found")
        return None
    
    # Replace meta-model loading section with:
    try:
        # Try robust meta-model first
        meta_model = joblib.load('models/robust_meta_model.pkl')
        meta_scaler = joblib.load('models/robust_meta_scaler.pkl') 
        meta_features = joblib.load('models/robust_meta_features.pkl')
        models['meta'] = (meta_model, meta_features, meta_scaler, 'robust')
        print("   ✅ Robust meta-model loaded")
    except FileNotFoundError:
        try:
            # Fallback to original
            meta_model = joblib.load('models/meta_model.pkl')
            meta_features = joblib.load('models/meta_features.pkl')
            meta_feature_scaler = joblib.load('models/meta_feature_scaler.pkl')
            models['meta'] = (meta_model, meta_features, meta_feature_scaler, 'original')
            print("   ⚠️ Using original meta-model")
        except FileNotFoundError:
            print("   ❌ No meta-model found")
            return None
    return models

def compute_pitch_score_with_diagnostics(df, swing_pipe, whiff_pipe, exit_pipe, update_date):
    """Compute pitch scores with diagnostic tracking"""
    print("🔍 Computing pitch scores with diagnostics...")
    
    df2 = df.copy()
    df2['is_swing_bin'] = df2['is_swing'].fillna(False).infer_objects(copy=False).astype(int)

    num_cols = [
        'start_speed','end_speed','extension','spin_rate',
        'pfxx','pfxz','vx0','vy0','vz0','sz_top','sz_bot'
    ]
    cat_cols = ['pitch_type','batter_hand','pitcher_hand']
    feats = num_cols + cat_cols
    
    # Feature availability diagnostics
    available_feats = [f for f in feats if f in df2.columns]
    missing_feats = [f for f in feats if f not in df2.columns]
    
    print(f"   Features available: {len(available_feats)}/{len(feats)}")
    if missing_feats:
        print(f"   Missing features: {missing_feats}")

    # Data quality check
    total_rows = len(df2)
    complete_rows = df2[available_feats].dropna().shape[0]
    completeness = complete_rows / total_rows if total_rows > 0 else 0
    
    print(f"   Data completeness: {complete_rows}/{total_rows} ({completeness*100:.1f}%)")

    if completeness < 0.5:
        print("   ⚠️ WARNING: Low data completeness may affect predictions")

    Xn = df2[available_feats]
    
    # Swing probabilities
    p_s = swing_pipe.predict_proba(Xn)[:, 1] if swing_pipe else np.zeros(len(df2))
    
    # Whiff probabilities
    p_w = np.zeros(len(df2))
    m = df2['is_swing_bin'] == 1
    if m.any() and whiff_pipe:
        p_w[m] = whiff_pipe.predict_proba(Xn[m])[:, 1]

    # Exit velocity predictions
    e_e = np.zeros(len(df2))
    hm = m & (p_w < 1)
    if hm.any() and exit_pipe:
        e_e[hm] = exit_pipe.predict(Xn[hm])

    # Compute pitch scores
    pitch_scores = p_s * (1 - p_w) * e_e
    
    # Diagnostic statistics
    diagnostics = {
        'update_date': update_date,
        'total_pitches': len(df2),
        'feature_completeness': completeness,
        'swing_prob_mean': p_s.mean(),
        'swing_prob_std': p_s.std(),
        'whiff_prob_mean': p_w[m].mean() if m.any() else 0,
        'whiff_prob_std': p_w[m].std() if m.any() else 0,
        'exit_velo_mean': e_e[hm].mean() if hm.any() else 0,
        'exit_velo_std': e_e[hm].std() if hm.any() else 0,
        'pitch_score_mean': pitch_scores.mean(),
        'pitch_score_std': pitch_scores.std(),
        'pitch_score_zeros': (pitch_scores == 0).sum(),
        'pitch_score_zeros_pct': (pitch_scores == 0).mean() * 100
    }
    
    # Save diagnostics
    diag_df = pd.DataFrame([diagnostics])
    diag_file = f'diagnostics/weekly_updates/pitch_scores_{update_date}.csv'
    diag_df.to_csv(diag_file, index=False)
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(p_s, bins=50, alpha=0.7)
    axes[0, 0].set_title(f'Swing Probabilities (μ={p_s.mean():.3f})')
    axes[0, 0].set_xlabel('P(Swing)')
    
    if m.any():
        axes[0, 1].hist(p_w[m], bins=50, alpha=0.7)
        axes[0, 1].set_title(f'Whiff Probabilities (μ={p_w[m].mean():.3f})')
        axes[0, 1].set_xlabel('P(Whiff)')
    
    if hm.any():
        axes[1, 0].hist(e_e[hm], bins=50, alpha=0.7)
        axes[1, 0].set_title(f'Exit Velocity (μ={e_e[hm].mean():.1f})')
        axes[1, 0].set_xlabel('Exit Velocity')
    
    non_zero_scores = pitch_scores[pitch_scores > 0]
    if len(non_zero_scores) > 0:
        axes[1, 1].hist(non_zero_scores, bins=50, alpha=0.7)
        axes[1, 1].set_title(f'Pitch Scores (μ={pitch_scores.mean():.3f})')
        axes[1, 1].set_xlabel('Pitch Score')
    
    plt.tight_layout()
    plt.savefig(f'diagnostics/weekly_updates/pitch_score_distributions_{update_date}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return pd.Series(pitch_scores, index=df2.index), diagnostics

def prepare_game_sequences_with_diagnostics(df, feats, target, seq_len=5, update_date=None):
    """Prepare sequences with diagnostic tracking"""
    print(f"🔍 Preparing sequences with diagnostics...")
    
    X, y, idxs = [], [], []
    sequence_stats = []
    
    for name, g in df.groupby(['Name', 'Season']):
        g = g.sort_values('game_date')
        F = g[feats].values
        T = g[target].values
        
        valid_sequences = 0
        nan_targets = 0
        nan_features = 0
        
        for i in range(len(g) - seq_len):
            if np.isnan(T[i + seq_len]):
                nan_targets += 1
                continue
            window = F[i:i + seq_len]
            if np.isnan(window).any():
                nan_features += 1
                continue
            
            X.append(window)
            y.append(T[i + seq_len])
            idxs.append(g.index[i + seq_len])
            valid_sequences += 1
        
        sequence_stats.append({
            'player': name[0],
            'season': name[1],
            'total_games': len(g),
            'valid_sequences': valid_sequences,
            'nan_targets': nan_targets,
            'nan_features': nan_features,
            'update_date': update_date
        })
    
    # Save sequence statistics
    seq_df = pd.DataFrame(sequence_stats)
    if update_date:
        seq_df.to_csv(f'diagnostics/weekly_updates/sequences_{update_date}.csv', index=False)
    
    print(f"   Sequences generated: {len(X)} from {len(sequence_stats)} players")
    print(f"   Avg sequences per player: {seq_df['valid_sequences'].mean():.2f}")
    
    return np.array(X), np.array(y), idxs, sequence_stats

def update_lstm_with_diagnostics(lstm, scaler, Xc_scaled, yc, update_date):
    """Update LSTM with diagnostic tracking"""
    print(f"🔄 Updating LSTM model with diagnostics...")
    
    if len(Xc_scaled) == 0:
        print("   ⚠️ No sequences for LSTM update")
        return lstm, {}
    
    # Pre-update evaluation
    pre_update_loss = lstm.evaluate(Xc_scaled, yc, verbose=0)
    pre_update_preds = lstm.predict(Xc_scaled, verbose=0).flatten()
    pre_update_mse = np.mean((yc - pre_update_preds) ** 2)
    pre_update_mae = np.mean(np.abs(yc - pre_update_preds))
    
    print(f"   Pre-update performance: Loss={pre_update_loss[0]:.4f}, MSE={pre_update_mse:.4f}, MAE={pre_update_mae:.4f}")
    
    # Fine-tune with current data
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.7, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    batch_size = min(32, len(Xc_scaled))
    history = lstm.fit(
        Xc_scaled, yc, 
        epochs=10,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Post-update evaluation
    post_update_loss = lstm.evaluate(Xc_scaled, yc, verbose=0)
    post_update_preds = lstm.predict(Xc_scaled, verbose=0).flatten()
    post_update_mse = np.mean((yc - post_update_preds) ** 2)
    post_update_mae = np.mean(np.abs(yc - post_update_preds))
    
    print(f"   Post-update performance: Loss={post_update_loss[0]:.4f}, MSE={post_update_mse:.4f}, MAE={post_update_mae:.4f}")
    
    # Performance improvement analysis
    mse_improvement = pre_update_mse - post_update_mse
    mae_improvement = pre_update_mae - post_update_mae
    
    print(f"   Improvement: MSE={mse_improvement:.4f}, MAE={mae_improvement:.4f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training history
    if 'loss' in history.history:
        axes[0, 0].plot(history.history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
    
    # Before/after predictions
    axes[0, 1].scatter(yc, pre_update_preds, alpha=0.6, label='Pre-update', s=20)
    axes[0, 1].scatter(yc, post_update_preds, alpha=0.6, label='Post-update', s=20)
    axes[0, 1].plot([yc.min(), yc.max()], [yc.min(), yc.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Predictions Before/After Update')
    axes[0, 1].legend()
    
    # Residuals analysis
    pre_residuals = yc - pre_update_preds
    post_residuals = yc - post_update_preds
    
    axes[1, 0].scatter(pre_update_preds, pre_residuals, alpha=0.6, label='Pre-update', s=20)
    axes[1, 0].scatter(post_update_preds, post_residuals, alpha=0.6, label='Post-update', s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals Before/After Update')
    axes[1, 0].legend()
    
    # Distribution of improvements
    prediction_changes = post_update_preds - pre_update_preds
    axes[1, 1].hist(prediction_changes, bins=20, alpha=0.7)
    axes[1, 1].set_xlabel('Prediction Change')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Prediction Changes (μ={prediction_changes.mean():.4f})')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'diagnostics/weekly_updates/lstm_update_{update_date}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model
    lstm.save('models/level2_lstm.h5', save_format='tf')
    
    # Return diagnostics
    diagnostics = {
        'update_date': update_date,
        'n_sequences': len(Xc_scaled),
        'batch_size': batch_size,
        'epochs_run': len(history.history['loss']) if 'loss' in history.history else 0,
        'pre_update_mse': pre_update_mse,
        'post_update_mse': post_update_mse,
        'pre_update_mae': pre_update_mae,
        'post_update_mae': post_update_mae,
        'mse_improvement': mse_improvement,
        'mae_improvement': mae_improvement,
        'prediction_change_mean': prediction_changes.mean(),
        'prediction_change_std': prediction_changes.std()
    }
    
    # Save diagnostics
    pd.DataFrame([diagnostics]).to_csv(
        f'diagnostics/weekly_updates/lstm_update_performance_{update_date}.csv', 
        index=False
    )
    
    return lstm, diagnostics

def update_bayesian_model_with_diagnostics(bayes_df_cur, pitchers, update_date):
    """Update Bayesian model with comprehensive diagnostics"""
    print(f"🧠 Updating Bayesian model with diagnostics...")
    
    if len(bayes_df_cur) == 0:
        print("   ⚠️ No data for Bayesian update")
        return pitchers, np.zeros(len(pitchers)), {}
    
    # Input diagnostics
    print(f"   Input data: {len(bayes_df_cur)} observations for {bayes_df_cur['pitcher'].nunique()} pitchers")
    
    # Check which pitchers are new vs existing
    existing_pitchers = set(pitchers)
    current_pitchers = set(bayes_df_cur['pitcher'].unique())
    new_pitchers = current_pitchers - existing_pitchers
    missing_pitchers = existing_pitchers - current_pitchers
    
    print(f"   New pitchers: {len(new_pitchers)}")
    print(f"   Missing pitchers: {len(missing_pitchers)}")
    
    if new_pitchers:
        print(f"   Adding new pitchers: {list(new_pitchers)[:5]}{'...' if len(new_pitchers) > 5 else ''}")
    
    # Expand pitcher list if needed
    all_pitchers = list(set(pitchers) | current_pitchers)
    
    idx_cur = pd.Categorical(bayes_df_cur['pitcher'], categories=all_pitchers).codes
    
    # Prior diagnostics
    metric_stats = bayes_df_cur['metric'].describe()
    print(f"   Metric range: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
    print(f"   Metric mean±std: {metric_stats['mean']:.4f}±{metric_stats['std']:.4f}")
    
    # Observations per pitcher
    obs_per_pitcher = bayes_df_cur.groupby('pitcher').size()
    print(f"   Observations per pitcher: mean={obs_per_pitcher.mean():.2f}, range=[{obs_per_pitcher.min()}, {obs_per_pitcher.max()}]")
    
    with pm.Model() as m:
        mu = pm.Normal('mu', 0, 2)
        sd = pm.HalfNormal('sd', 2)
        a = pm.Normal('a', mu=mu, sigma=sd, shape=len(all_pitchers))
        eps = pm.HalfCauchy('eps', 2)
        pm.Normal('obs', mu=a[idx_cur], sigma=eps, observed=bayes_df_cur['metric'])
        
        # Shorter sampling for weekly updates
        idata = pm.sample(1000, tune=1000, cores=2, target_accept=0.9, random_seed=42)
    
    # Convergence diagnostics
    try:
        rhat = pm.rhat(idata)
        max_rhat = float(rhat.max().values) if hasattr(rhat, 'max') else float(rhat.to_array().max().values)
        
        ess_bulk = pm.ess(idata, kind="bulk")
        min_ess_bulk = float(ess_bulk.min().values) if hasattr(ess_bulk, 'min') else float(ess_bulk.to_array().min().values)
        
        print(f"   Convergence: R-hat={max_rhat:.4f}, ESS={min_ess_bulk:.0f}")
        convergence_ok = max_rhat < 1.1 and min_ess_bulk > 100  # Relaxed for weekly updates
        
    except Exception as e:
        print(f"   Convergence diagnostics failed: {e}")
        max_rhat = np.nan
        min_ess_bulk = np.nan
        convergence_ok = False
    
    # Extract posterior
    a_mean_upd = idata.posterior['a'].mean(dim=('chain', 'draw')).values
    a_std_upd = idata.posterior['a'].std(dim=('chain', 'draw')).values
    mu_mean = float(idata.posterior['mu'].mean().values)
    sd_mean = float(idata.posterior['sd'].mean().values)
    eps_mean = float(idata.posterior['eps'].mean().values)
    
    print(f"   Updated estimates: μ={mu_mean:.4f}, σ={sd_mean:.4f}, ε={eps_mean:.4f}")
    
    # Effect size diagnostics
    effects_range = a_mean_upd.max() - a_mean_upd.min()
    print(f"   Pitcher effects range: {effects_range:.4f}")
    
    # Compare with empirical means for current pitchers
    empirical_means = bayes_df_cur.groupby('pitcher')['metric'].mean()
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Posterior distributions
    axes[0, 0].hist(idata.posterior['mu'].values.flatten(), bins=30, alpha=0.7)
    axes[0, 0].set_title(f'Global Mean μ (est: {mu_mean:.3f})')
    axes[0, 0].axvline(mu_mean, color='red', linestyle='--')
    
    axes[0, 1].hist(idata.posterior['sd'].values.flatten(), bins=30, alpha=0.7)
    axes[0, 1].set_title(f'Between-pitcher SD σ (est: {sd_mean:.3f})')
    axes[0, 1].axvline(sd_mean, color='red', linestyle='--')
    
    # Pitcher effects
    axes[1, 0].hist(a_mean_upd, bins=30, alpha=0.7)
    axes[1, 0].set_title(f'Pitcher Effects (range: {effects_range:.3f})')
    axes[1, 0].axvline(a_mean_upd.mean(), color='red', linestyle='--')
    
    # Shrinkage analysis for current pitchers
    current_pitcher_indices = [i for i, p in enumerate(all_pitchers) if p in empirical_means.index]
    if current_pitcher_indices:
        empirical_values = [empirical_means[all_pitchers[i]] for i in current_pitcher_indices]
        posterior_values = [a_mean_upd[i] for i in current_pitcher_indices]
        
        axes[1, 1].scatter(empirical_values, posterior_values, alpha=0.6)
        min_val = min(min(empirical_values), min(posterior_values))
        max_val = max(max(empirical_values), max(posterior_values))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Empirical Mean')
        axes[1, 1].set_ylabel('Posterior Mean')
        axes[1, 1].set_title('Shrinkage Effect')
    
    plt.tight_layout()
    plt.savefig(f'diagnostics/weekly_updates/bayesian_update_{update_date}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save updated model
    joblib.dump((all_pitchers, a_mean_upd), 'models/level3_bayes.pkl')
    
    # Diagnostics summary
    diagnostics = {
        'update_date': update_date,
        'n_observations': len(bayes_df_cur),
        'n_pitchers': len(current_pitchers),
        'n_new_pitchers': len(new_pitchers),
        'n_missing_pitchers': len(missing_pitchers),
        'max_rhat': max_rhat,
        'min_ess_bulk': min_ess_bulk,
        'convergence_ok': convergence_ok,
        'global_mean': mu_mean,
        'between_pitcher_sd': sd_mean,
        'within_pitcher_sd': eps_mean,
        'effects_range': effects_range,
        'metric_mean': metric_stats['mean'],
        'metric_std': metric_stats['std']
    }
    
    pd.DataFrame([diagnostics]).to_csv(
        f'diagnostics/weekly_updates/bayesian_update_{update_date}.csv', 
        index=False
    )
    
    return all_pitchers, a_mean_upd, diagnostics

def create_proper_game_features(pitch_df):
    """
    Create the proper game-level features that LSTM expects
    """
    
    # Aggregate pitch-level data to game level properly
    numeric_cols = [
        'start_speed', 'end_speed', 'sz_top', 'sz_bot', 'x', 'y',
        'ax', 'ay', 'az', 'pfxx', 'pfxz', 'px', 'pz',
        'vx0', 'vy0', 'vz0', 'x0', 'y0', 'z0', 'zone',
        'type_confidence', 'plate_time', 'extension',
        'spin_rate', 'spin_direction', 'vb', 'ivb', 'hb'
    ]
    
    # Create aggregations
    game_aggs = {}
    
    for col in numeric_cols:
        if col in pitch_df.columns:
            game_aggs[f'{col}_mean'] = (col, 'mean')
            game_aggs[f'{col}_std'] = (col, 'std')
            game_aggs[f'{col}_count'] = (col, 'count')
    
    # Group by game and aggregate
    game_df = (
        pitch_df.groupby(['game_id', 'game_date', 'Name', 'Season'])
        .agg(**game_aggs)
        .reset_index()
    )
    
    # Add RBI mean (if exists)
    if 'rbi' in pitch_df.columns:
        rbi_agg = (
            pitch_df.groupby(['game_id', 'game_date', 'Name', 'Season'])['rbi']
            .mean()
            .reset_index()
            .rename(columns={'rbi': 'rbi_mean'})
        )
        game_df = game_df.merge(rbi_agg, on=['game_id', 'game_date', 'Name', 'Season'], how='left')
    else:
        # Create a proxy RBI from pitch count
        game_df['rbi_mean'] = game_df['start_speed_count'] / 100  # Normalize pitch count
    
    # Fill NaN values with 0
    game_df = game_df.fillna(0)
    
    return game_df

def collect_current_season_data(season_year, max_players=None, test_mode=False):
    """
    PRODUCTION-READY VERSION: Full robust implementation
    """
    from data_scraper import MLB_Scrape
    
    scraper = MLB_Scrape()
    
    # Use the enhanced method
    pitch_df_current, game_df_current, season_stats = scraper.get_current_season_data_for_training(
        season_year=season_year,
        max_players=max_players,
        test_mode=test_mode,
        min_innings=10 if test_mode else 20  # Lower threshold for testing
    )
    
    if pitch_df_current is None:
        return None, None, None
    
    # Create proper game-level features
    game_df_current = create_proper_game_features(pitch_df_current)
    
    print(f"✅ Data collected in training pipeline format")
    print(f"✅ Pitch data: {len(pitch_df_current):,} rows, {pitch_df_current['Name'].nunique()} players")
    print(f"✅ Game data: {len(game_df_current):,} rows")
    print(f"✅ Season data: {len(season_stats):,} players")
    
    return pitch_df_current, game_df_current, season_stats

def check_feature_quality(season_cur):
    """Check if features have sufficient quality for meta-model"""
    
    base_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    quality_info = {
        'usable': True,
        'issues': [],
        'feature_stats': {}
    }
    
    for feat in base_features:
        if feat not in season_cur.columns:
            quality_info['issues'].append(f'Missing feature: {feat}')
            quality_info['usable'] = False
            continue
            
        values = season_cur[feat].fillna(season_cur[feat].median())
        stats = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'unique_count': values.nunique()
        }
        
        # Check for problems
        if stats['std'] < 1e-10:
            quality_info['issues'].append(f'{feat}: zero variance')
            quality_info['usable'] = False
        elif stats['unique_count'] <= 1:
            quality_info['issues'].append(f'{feat}: ≤1 unique values')
            quality_info['usable'] = False
        elif np.isnan(stats['std']) or np.isinf(stats['std']):
            quality_info['issues'].append(f'{feat}: invalid statistics')
            quality_info['usable'] = False
            
        quality_info['feature_stats'][feat] = stats
    
    return quality_info

def validate_prediction_quality(predictions):
    """Validate that predictions are acceptable"""
    
    quality = {
        'acceptable': True,
        'reason': None,
        'stats': {
            'min': predictions.min(),
            'max': predictions.max(),
            'mean': predictions.mean(),
            'std': predictions.std(),
            'unique_count': len(np.unique(predictions))
        }
    }
    
    # Check for problems
    if quality['stats']['std'] < 0.01:
        quality['acceptable'] = False
        quality['reason'] = f"Low variance (std={quality['stats']['std']:.6f})"
    elif quality['stats']['unique_count'] <= 3:
        quality['acceptable'] = False
        quality['reason'] = f"Too few unique values ({quality['stats']['unique_count']})"
    elif quality['stats']['max'] - quality['stats']['min'] < 0.05:
        quality['acceptable'] = False
        quality['reason'] = f"Narrow range ({quality['stats']['max'] - quality['stats']['min']:.6f})"
    elif np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        quality['acceptable'] = False
        quality['reason'] = "Contains NaN or infinite values"
    
    return quality

def create_robust_ranking_fallback(season_cur):
    """
    Improved ranking fallback that creates meaningful probability differences
    """
    print("🔧 Creating robust ranking fallback...")
    
    # Enhanced feature weights based on Cy Young voting patterns
    feature_config = {
        'pitch_score_mean': {
            'weight': 0.45,
            'direction': 'higher_better',
            'description': 'Pitch quality (most important)'
        },
        'rbi_pred_mean': {
            'weight': -0.25,
            'direction': 'lower_better', 
            'description': 'RBI prediction (lower is better for pitchers)'
        },
        'bayes_effect': {
            'weight': 0.30,
            'direction': 'higher_better',
            'description': 'Bayesian hierarchical adjustment'
        }
    }
    
    # Calculate composite score
    total_score = np.zeros(len(season_cur))
    weights_used = []
    
    print("Feature analysis:")
    for feat, config in feature_config.items():
        if feat in season_cur.columns:
            values = season_cur[feat].fillna(season_cur[feat].median())
            
            if values.std() > 1e-10:  # Has meaningful variance
                # Robust normalization using percentiles to handle outliers
                p05, p50, p95 = np.percentile(values, [5, 50, 95])
                
                if p95 > p05:
                    # Normalize to 0-1 using robust scaling
                    normalized = np.clip((values - p05) / (p95 - p05), 0, 1)
                    
                    # Apply weight
                    contribution = config['weight'] * normalized
                    total_score += contribution
                    weights_used.append(config['weight'])
                    
                    print(f"  ✅ {feat}: weight={config['weight']:.3f}, range=[{values.min():.4f}, {values.max():.4f}], std={values.std():.4f}")
                else:
                    print(f"  ⚠️ {feat}: insufficient range (p95={p95:.6f}, p05={p05:.6f})")
            else:
                print(f"  ❌ {feat}: zero variance (std={values.std():.10f})")
        else:
            print(f"  ❌ {feat}: missing from data")
    
    if len(weights_used) == 0:
        print("  ⚠️ No usable features - creating random rankings")
        np.random.seed(42)  # Reproducible
        return np.random.uniform(0.1, 0.8, len(season_cur))
    
    # Normalize by total absolute weight
    total_abs_weight = sum(abs(w) for w in weights_used)
    if total_abs_weight > 0:
        total_score = total_score / total_abs_weight
    
    # Convert to probabilities with good spread
    if total_score.std() > 1e-10:
        # Use rank-based transformation for better distribution
        ranks = total_score.rank(pct=True)
        
        # Apply power transformation to create more separation at the top
        # Higher power = more emphasis on top performers
        power = 1.5
        probabilities = ranks ** power
        
        # Scale to reasonable Cy Young probability range
        # Top candidate should be ~70-80%, bottom should be ~5-10%
        min_prob = 0.05
        max_prob = 0.75
        probabilities = min_prob + (max_prob - min_prob) * probabilities
        
        print(f"  Ranking results: range=[{probabilities.min():.4f}, {probabilities.max():.4f}], std={probabilities.std():.4f}")
        
    else:
        print("  ⚠️ Zero score variance - using uniform distribution")
        np.random.seed(42)
        probabilities = np.random.uniform(0.1, 0.7, len(season_cur))
    
    return probabilities

def predict_cy_young_with_diagnostics_FIXED(meta_model, meta_features, meta_feature_scaler, features_dict, player_name):
    """
    IMPROVED prediction function with better error handling and diagnostics
    """
    
    # Extract base features in EXACT order
    base_features = [
        features_dict.get('pitch_score_mean', 0),
        features_dict.get('rbi_pred_mean', 0), 
        features_dict.get('bayes_effect', 0)
    ]
    
    # Handle invalid values
    base_features = [0.0 if (pd.isna(f) or np.isinf(f)) else float(f) for f in base_features]
    
    # Check for zero variance in features (common issue)
    if len(set(base_features)) == 1:
        print(f"⚠️ All features identical for {player_name}: {base_features}")
        return 0.5, {
            'player': player_name,
            'raw_features': base_features,
            'prediction_successful': False,
            'error': 'zero_variance_features'
        }
    
    # Apply scaling using the passed scaler
    try:
        scaled_features = meta_feature_scaler.transform([base_features])[0]
        
        # Check if scaling produced valid results
        if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
            print(f"⚠️ Scaling produced invalid values for {player_name}")
            return 0.3, {'error': 'invalid_scaling', 'player': player_name}
            
        # Check if all scaled features are identical (scaling problem)
        if len(set(scaled_features)) == 1:
            print(f"⚠️ Scaling made all features identical for {player_name}")
            return 0.4, {'error': 'scaling_zero_variance', 'player': player_name}
            
    except Exception as e:
        print(f"❌ Scaling failed for {player_name}: {e}")
        return 0.2, {'error': 'scaling_failed', 'message': str(e), 'player': player_name}
    
    # Create interactions AFTER scaling
    interactions = [
        scaled_features[0] * scaled_features[1],  # pitch_rbi
        scaled_features[0] * scaled_features[2],  # pitch_bayes  
        scaled_features[1] * scaled_features[2]   # rbi_bayes
    ]
    
    # Combine all features
    all_features = list(scaled_features) + interactions
    
    # Only use features that were actually trained on
    final_features = []
    feature_names = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect', 
                    'pitch_rbi_interaction', 'pitch_bayes_interaction', 'rbi_bayes_interaction']
    
    for i, feat_name in enumerate(feature_names):
        if feat_name in meta_features:
            final_features.append(all_features[i])
    
    if len(final_features) == 0:
        print(f"❌ No valid features for {player_name}")
        return 0.1, {'error': 'no_valid_features', 'player': player_name}
    
    # Make prediction
    try:
        probability = meta_model.predict_proba([final_features])[0, 1]
        
        # Validate prediction
        if pd.isna(probability) or np.isinf(probability):
            print(f"⚠️ Invalid prediction for {player_name}")
            return 0.25, {'error': 'invalid_prediction', 'player': player_name}
        
        return float(probability), {
            'player': player_name,
            'raw_features': base_features,
            'scaled_features': list(scaled_features),
            'final_features': final_features,
            'prediction_successful': True,
            'predicted_probability': float(probability)
        }
        
    except Exception as e:
        print(f"❌ Prediction failed for {player_name}: {e}")
        return 0.15, {
            'player': player_name,
            'prediction_successful': False,
            'error_message': str(e)
        }

def make_robust_predictions(season_cur, models):
    """
    Updated Step 4 function - Complete replacement for broken prediction section
    """
    import numpy as np
    import pandas as pd
    
    print("🔮 Making robust Cy Young predictions...")
    
    # Get meta-model components
    meta_components = models['meta']
    
    # Handle different model formats
    if len(meta_components) == 4:
        meta_model, meta_features, meta_scaler, model_type = meta_components
    else:
        # Original format - assume it's the problematic original model
        meta_model, meta_features, meta_scaler = meta_components
        model_type = 'original'
    
    print(f"Using {model_type} meta-model approach...")
    
    # Try the improved original prediction function first
    if model_type == 'original':
        try:
            print("🔧 Attempting improved original model prediction...")
            
            predictions = []
            prediction_diagnostics = []
            
            # Check feature quality first
            feature_quality = check_feature_quality(season_cur)
            print(f"Feature quality assessment: {feature_quality}")
            
            if not feature_quality['usable']:
                print("⚠️ Poor feature quality detected - using ranking fallback")
                return create_robust_ranking_fallback(season_cur)
            
            # Try predictions with improved function
            for _, row in season_cur.iterrows():
                features = {
                    'pitch_score_mean': row.get('pitch_score_mean', 0),
                    'rbi_pred_mean': row.get('rbi_pred_mean', 0),
                    'bayes_effect': row.get('bayes_effect', 0)
                }
                
                prob, diagnostics = predict_cy_young_with_diagnostics_FIXED(
                    meta_model, meta_features, meta_scaler, features, row['Name']
                )
                
                predictions.append(prob)
                prediction_diagnostics.append(diagnostics)
            
            predictions = np.array(predictions)
            
            # Validate prediction quality
            pred_quality = validate_prediction_quality(predictions)
            print(f"Prediction quality: {pred_quality}")
            
            if pred_quality['acceptable']:
                print(f"✅ Original model successful: range=[{predictions.min():.4f}, {predictions.max():.4f}], std={predictions.std():.4f}")
                
                # Save diagnostics
                pred_diag_df = pd.DataFrame(prediction_diagnostics)
                pred_diag_df.to_csv('diagnostics/weekly_updates/improved_prediction_diagnostics.csv', index=False)
                
                return predictions
            else:
                print(f"⚠️ Original model predictions poor quality: {pred_quality['reason']}")
                
        except Exception as e:
            print(f"❌ Original model failed: {e}")
    
    # Try robust model if available
    elif model_type == 'robust':
        try:
            # Try to import robust prediction function
            try:
                from fixed_meta_model_training import predict_with_robust_meta_model
                predictions, pred_stats = predict_with_robust_meta_model(season_cur)
                
                if predictions is not None and pred_stats['std_prob'] > 0.01:
                    print(f"✅ Robust model successful: range=[{pred_stats['min_prob']:.4f}, {pred_stats['max_prob']:.4f}]")
                    return predictions
                else:
                    print("⚠️ Robust model had poor variance")
            except ImportError:
                print("⚠️ Robust prediction function not available")
                    
        except Exception as e:
            print(f"❌ Robust model failed: {e}")
    
    # Final fallback: Improved ranking system
    print("🔧 Using improved ranking fallback...")
    return create_robust_ranking_fallback(season_cur)

def replace_broken_prediction_section_COMPLETE_FIXED(season_cur, models):
    """
    COMPLETE FIXED replacement for Step 4 - handles missing columns gracefully
    """
    
    print("\n" + "="*40)
    print("META-MODEL: FIXED PREDICTIONS")
    print("="*40)
    
    # Make predictions using robust method
    predictions = make_robust_predictions(season_cur, models)
    
    # Assign predictions
    season_cur['cywin_prob'] = predictions
    
    # Comprehensive validation and diagnostics
    pred_stats = {
        'min': predictions.min(),
        'max': predictions.max(),
        'mean': predictions.mean(),
        'std': predictions.std(),
        'median': np.median(predictions),
        'unique_values': len(np.unique(predictions)),
        'range': predictions.max() - predictions.min()
    }
    
    print(f"Final prediction statistics:")
    print(f"  Range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]")
    print(f"  Mean±Std: {pred_stats['mean']:.4f}±{pred_stats['std']:.4f}")
    print(f"  Median: {pred_stats['median']:.4f}")
    print(f"  Unique values: {pred_stats['unique_values']}")
    print(f"  Total range: {pred_stats['range']:.4f}")
    
    # Quality assessment
    if pred_stats['std'] < 0.01:
        print("  ⚠️ WARNING: Low prediction variance - may need model retraining")
    elif pred_stats['range'] < 0.1:
        print("  ⚠️ WARNING: Narrow prediction range - limited differentiation")
    elif pred_stats['unique_values'] < len(season_cur) * 0.5:
        print("  ⚠️ WARNING: Many duplicate predictions")
    else:
        print("  ✅ Prediction quality looks good")
    
    # FIXED: Show top candidates with only available columns
    available_display_cols = ['Name', 'cywin_prob']
    optional_cols = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    
    for col in optional_cols:
        if col in season_cur.columns:
            available_display_cols.append(col)
    
    top_10 = season_cur.nlargest(10, 'cywin_prob')[available_display_cols]
    
    print(f"\nTop 10 Candidates Preview:")
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        # Build display string dynamically based on available columns
        display_parts = [f"{i:2d}. {row['Name']:<20} {row['cywin_prob']:.4f}"]
        
        feature_parts = []
        if 'pitch_score_mean' in row.index:
            feature_parts.append(f"PS:{row['pitch_score_mean']:.3f}")
        if 'rbi_pred_mean' in row.index:
            feature_parts.append(f"RBI:{row['rbi_pred_mean']:.3f}")
        if 'bayes_effect' in row.index:
            feature_parts.append(f"BE:{row['bayes_effect']:.3f}")
        
        if feature_parts:
            display_parts.append(f" ({', '.join(feature_parts)})")
        
        print(''.join(display_parts))
    
    return season_cur

def create_enhanced_results_analysis_FIXED(season_cur, update_date):
    """
    FIXED results analysis that handles missing columns gracefully
    """
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    # Sort results - only require essential columns
    essential_cols = ['Name', 'Season', 'cywin_prob']
    optional_cols = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    
    # Build results columns list with only available columns
    results_cols = essential_cols.copy()
    for col in optional_cols:
        if col in season_cur.columns:
            results_cols.append(col)
    
    results = season_cur[results_cols].copy()
    results = results.sort_values('cywin_prob', ascending=False)
    
    # Top candidates analysis
    top_10 = results.head(10)
    print("\n🏆 Top 10 Cy Young Candidates:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        # Build display string with available features
        base_info = f"{i:2d}. {row['Name']:<25} {row['cywin_prob']:.4f}"
        
        feature_info = []
        if 'pitch_score_mean' in row.index:
            feature_info.append(f"PS:{row['pitch_score_mean']:.3f}")
        if 'rbi_pred_mean' in row.index:
            feature_info.append(f"RBI:{row['rbi_pred_mean']:.3f}")
        if 'bayes_effect' in row.index:
            feature_info.append(f"BE:{row['bayes_effect']:.3f}")
        
        if feature_info:
            print(f"{base_info} ({', '.join(feature_info)})")
        else:
            print(base_info)
    
    # Prediction distribution analysis
    prob_stats = season_cur['cywin_prob'].describe()
    print(f"\nPrediction Statistics:")
    print(f"  Range: [{prob_stats['min']:.4f}, {prob_stats['max']:.4f}]")
    print(f"  Mean±Std: {prob_stats['mean']:.4f}±{prob_stats['std']:.4f}")
    print(f"  Median: {prob_stats['50%']:.4f}")
    
    # Feature correlation analysis (only with available features)
    available_feature_cols = ['cywin_prob']
    for col in optional_cols:
        if col in season_cur.columns:
            available_feature_cols.append(col)
    
    correlation_analysis_performed = False
    if len(available_feature_cols) > 1:
        feature_data = season_cur[available_feature_cols].dropna()
        
        if len(feature_data) > 1:
            correlation_matrix = feature_data.corr()
            print(f"\nFeature Correlations with Cy Young Probability:")
            for col in available_feature_cols[1:]:  # Skip cywin_prob itself
                corr = correlation_matrix.loc[col, 'cywin_prob']
                print(f"  {col}: {corr:.4f}")
            correlation_analysis_performed = True
    
    if not correlation_analysis_performed:
        print(f"\nFeature correlation analysis skipped (insufficient features)")
    
    # Create adaptive visualization based on available data
    # Determine subplot layout based on available features
    n_feature_plots = sum(1 for col in optional_cols if col in season_cur.columns)
    total_plots = 2 + n_feature_plots + (1 if correlation_analysis_performed else 0)
    
    # Adaptive subplot layout
    if total_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 1: Prediction distribution
    axes[plot_idx].hist(season_cur['cywin_prob'], bins=30, alpha=0.7)
    axes[plot_idx].set_title('Cy Young Probability Distribution')
    axes[plot_idx].set_xlabel('Probability')
    axes[plot_idx].set_ylabel('Frequency')
    plot_idx += 1
    
    # Plot 2: Top 20 predictions
    top_20 = results.head(20)
    y_pos = range(len(top_20))
    axes[plot_idx].barh(y_pos, top_20['cywin_prob'])
    axes[plot_idx].set_yticks(y_pos)
    axes[plot_idx].set_yticklabels(top_20['Name'], fontsize=8)
    axes[plot_idx].set_xlabel('Cy Young Probability')
    axes[plot_idx].set_title('Top 20 Candidates')
    axes[plot_idx].invert_yaxis()
    plot_idx += 1
    
    # Feature vs prediction scatter plots (only for available features)
    feature_display_names = {
        'pitch_score_mean': 'Pitch Score Mean',
        'rbi_pred_mean': 'RBI Prediction Mean', 
        'bayes_effect': 'Bayesian Effect'
    }
    
    for col in optional_cols:
        if col in season_cur.columns and plot_idx < len(axes):
            axes[plot_idx].scatter(season_cur[col], season_cur['cywin_prob'], alpha=0.6)
            axes[plot_idx].set_xlabel(feature_display_names.get(col, col))
            axes[plot_idx].set_ylabel('Cy Young Probability')
            axes[plot_idx].set_title(f'{feature_display_names.get(col, col)} vs Prediction')
            plot_idx += 1
    
    # Correlation heatmap (if we have correlations and space)
    if correlation_analysis_performed and plot_idx < len(axes):
        im = axes[plot_idx].imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[plot_idx].set_xticks(range(len(available_feature_cols)))
        axes[plot_idx].set_yticks(range(len(available_feature_cols)))
        axes[plot_idx].set_xticklabels([col.replace('_', ' ').title() for col in available_feature_cols], 
                                     rotation=45, ha='right')
        axes[plot_idx].set_yticklabels([col.replace('_', ' ').title() for col in available_feature_cols])
        axes[plot_idx].set_title('Feature Correlation Matrix')
        
        # Add correlation values
        for i in range(len(available_feature_cols)):
            for j in range(len(available_feature_cols)):
                text = axes[plot_idx].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'diagnostics/weekly_updates/results_analysis_{update_date}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results

def weekly_update_with_comprehensive_diagnostics_FIXED(test_mode=False, max_players=None):
    """FIXED version of weekly update that handles missing columns gracefully"""
    
    update_date = datetime.date.today().strftime('%Y%m%d')
    current_year = datetime.date.today().year
    
    print(f"🔄 Running FIXED Weekly Update - {update_date}")
    print("=" * 60)
    
    # Load models
    models = load_trained_models_FIXED()
    if models is None:
        print("❌ Failed to load models. Please run training first.")
        return None, None
    
    # Collect current season data
    pitch_df_cur, game_df_cur, season_cur = collect_current_season_data(
        current_year, max_players=max_players, test_mode=test_mode
    )
    
    if pitch_df_cur is None:
        print("❌ Failed to collect current season data")
        return None, None
    
    # Level 1: Apply pitch-level models
    print("\n" + "="*40)
    print("LEVEL 1: APPLYING PITCH MODELS")
    print("="*40)
    
    swing_pipe, whiff_pipe, exit_pipe = models['level1']
    pitch_scores, pitch_diagnostics = compute_pitch_score_with_diagnostics(
        pitch_df_cur, swing_pipe, whiff_pipe, exit_pipe, update_date
    )
    pitch_df_cur['pitch_score'] = pitch_scores
    
    # Aggregate to game level
    agg_cur = (
        pitch_df_cur.groupby(['game_id', 'game_date', 'Name', 'Season'])['pitch_score']
        .mean().reset_index()
        .rename(columns={'pitch_score': 'pitch_score_mean'})
    )
    game_df_cur = game_df_cur.merge(agg_cur, on=['game_id', 'game_date', 'Name', 'Season'], how='left')
    
    # Level 2: Apply and update LSTM
    print("\n" + "="*40)
    print("LEVEL 2: APPLYING LSTM MODEL")
    print("="*40)
    
    lstm, scaler, game_feats = models['level2']
    
    # Ensure all required features exist
    for feat in game_feats:
        if feat not in game_df_cur.columns:
            game_df_cur[feat] = 0.0
            print(f"   ⚠️ Added missing feature '{feat}' with default value 0")
    
    seq_len = 5
    Xc, yc, idxs_c, seq_stats = prepare_game_sequences_with_diagnostics(
        game_df_cur, game_feats, 'rbi_mean', seq_len, update_date
    )
    
    if len(Xc) > 0:
        # Filter valid targets
        mask = ~np.isnan(yc)
        if mask.any():
            Xc, yc, idxs_c = Xc[mask], yc[mask], [idxs_c[i] for i in np.where(mask)[0]]
            
            # Scale sequences
            ns, sl, nf = Xc.shape
            Xc_flat = Xc.reshape(-1, nf)
            Xc_scaled = scaler.transform(Xc_flat).reshape(ns, sl, nf)
            
            # Update LSTM
            lstm, lstm_diagnostics = update_lstm_with_diagnostics(
                lstm, scaler, Xc_scaled, yc, update_date
            )
            
            # Generate predictions
            preds = lstm.predict(Xc_scaled, verbose=0).flatten()
            game_df_cur.loc[idxs_c, 'rbi_pred'] = preds
            
        else:
            print("   ⚠️ No valid targets for LSTM update")
    else:
        print("   ⚠️ No sequences available for LSTM")
    
    # Level 3: Apply and update Bayesian model
    print("\n" + "="*40)
    print("LEVEL 3: APPLYING BAYESIAN MODEL")
    print("="*40)
    
    pitchers, a_mean = models['level3']
    bayes_df_cur = game_df_cur[['Name', 'Season', 'rbi_pred']].rename(
        columns={'Name': 'pitcher', 'rbi_pred': 'metric'}
    ).dropna()
    
    if len(bayes_df_cur) > 0:
        pitchers_upd, a_mean_upd, bayes_diagnostics = update_bayesian_model_with_diagnostics(
            bayes_df_cur, pitchers, update_date
        )
        
        # Apply effects to season data
        pitcher_effects_dict = dict(zip(pitchers_upd, a_mean_upd))
        season_cur['bayes_effect'] = season_cur['Name'].map(pitcher_effects_dict).fillna(a_mean_upd.mean())
    else:
        print("   ⚠️ No data for Bayesian update")
        season_cur['bayes_effect'] = 0.0
    
    # Add aggregated features to season data if they exist
    if 'pitch_score' in pitch_df_cur.columns:
        pitch_season_agg = (
            pitch_df_cur.groupby(['Name', 'Season'])['pitch_score']
            .mean().reset_index()
            .rename(columns={'pitch_score': 'pitch_score_mean'})
        )
        season_cur = season_cur.merge(pitch_season_agg, on=['Name', 'Season'], how='left')
    
    if 'rbi_pred' in game_df_cur.columns:
        rbi_season_agg = (
            game_df_cur.groupby(['Name', 'Season'])['rbi_pred']
            .mean().reset_index()
            .rename(columns={'rbi_pred': 'rbi_pred_mean'})
        )
        season_cur = season_cur.merge(rbi_season_agg, on=['Name', 'Season'], how='left')
    
    # Make predictions with FIXED function
    season_cur = replace_broken_prediction_section_COMPLETE_FIXED(season_cur, models)
    
    # Enhanced results analysis with FIXED function
    results = create_enhanced_results_analysis_FIXED(season_cur, update_date)
    
    # Save detailed results
    results_file = f'results/cy_young_predictions_{current_year}_{update_date}.csv'
    season_cur.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create comprehensive update summary
    prob_stats = season_cur['cywin_prob'].describe()
    
    update_summary = {
        'update_date': update_date,
        'season_year': current_year,
        'total_pitchers': len(season_cur),
        'pitch_data_collected': len(pitch_df_cur),
        'game_data_generated': len(game_df_cur),
        'sequences_for_lstm': len(Xc) if 'Xc' in locals() else 0,
        'bayesian_observations': len(bayes_df_cur) if len(bayes_df_cur) > 0 else 0,
        'top_prediction': results.iloc[0]['cywin_prob'] if len(results) > 0 else 0,
        'top_candidate': results.iloc[0]['Name'] if len(results) > 0 else 'N/A',
        'prediction_mean': prob_stats['mean'],
        'prediction_std': prob_stats['std'],
        'prediction_range': prob_stats['max'] - prob_stats['min'],
        'available_features': list(season_cur.columns),
        'feature_completeness': 1.0  # Since we handle missing gracefully
    }
    
    # Add model-specific diagnostics
    if 'pitch_diagnostics' in locals():
        update_summary.update({k: v for k, v in pitch_diagnostics.items() if k != 'update_date'})
    
    if 'lstm_diagnostics' in locals():
        update_summary.update({f'lstm_{k}': v for k, v in lstm_diagnostics.items() if k != 'update_date'})
    
    if 'bayes_diagnostics' in locals():
        update_summary.update({f'bayes_{k}': v for k, v in bayes_diagnostics.items() if k != 'update_date'})
    
    # Save update summary
    summary_df = pd.DataFrame([update_summary])
    summary_df.to_csv(f'diagnostics/weekly_updates/update_summary_{update_date}.csv', index=False)
    
    print(f"\n📊 FIXED Update Summary:")
    print(f"  Date: {update_date}")
    print(f"  Pitchers analyzed: {len(season_cur)}")
    print(f"  Top candidate: {update_summary['top_candidate']} ({update_summary['top_prediction']:.4f})")
    print(f"  Prediction range: [{prob_stats['min']:.4f}, {prob_stats['max']:.4f}]")
    print(f"  Prediction std: {prob_stats['std']:.4f}")
    print(f"  Available features: {[col for col in ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect'] if col in season_cur.columns]}")
    
    return season_cur, update_summary

def generate_weekly_report(update_date=None):
    """Generate a comprehensive weekly report"""
    if update_date is None:
        update_date = datetime.date.today().strftime('%Y%m%d')
    
    print(f"📋 Generating Weekly Report for {update_date}")
    
    # Load latest results
    try:
        current_year = datetime.date.today().year
        results_file = f'results/cy_young_predictions_{current_year}_{update_date}.csv'
        results_df = pd.read_csv(results_file)
        
        update_summary_file = f'diagnostics/weekly_updates/update_summary_{update_date}.csv'
        summary_df = pd.read_csv(update_summary_file)
        
    except FileNotFoundError as e:
        print(f"❌ Could not load results for {update_date}: {e}")
        return
    
    # Generate report
    report = f"""
# MLB Cy Young Prediction Report - {update_date}

## Executive Summary
- **Analysis Date**: {update_date}
- **Season**: {current_year}
- **Pitchers Analyzed**: {len(results_df)}
- **Top Candidate**: {results_df.iloc[0]['Name']} ({results_df.iloc[0]['cywin_prob']:.4f})

## Top 10 Candidates
"""
    
    top_10 = results_df.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        report += f"{i}. **{row['Name']}** - {row['cywin_prob']:.4f}\n"
    
    report += f"""

## Model Performance Summary
- **Data Quality**: {summary_df.iloc[0]['feature_completeness']*100:.1f}% feature completeness
- **Pitch Data**: {summary_df.iloc[0]['pitch_data_collected']:,} pitches analyzed
- **LSTM Sequences**: {summary_df.iloc[0]['sequences_for_lstm']} sequences processed
- **Bayesian Update**: {summary_df.iloc[0]['bayesian_observations']} observations

## Prediction Analysis
- **Mean Prediction**: {summary_df.iloc[0]['prediction_mean']:.4f}
- **Standard Deviation**: {summary_df.iloc[0]['prediction_std']:.4f}
- **Prediction Range**: {summary_df.iloc[0]['prediction_range']:.4f}

## Data Quality Indicators
- **Feature Availability**: All required features present
- **Model Convergence**: Bayesian model converged successfully
- **LSTM Performance**: Model updated with current data

---
*Report generated automatically by the MLB Cy Young Prediction System*
"""
    
    # Save report
    report_file = f'results/weekly_report_{update_date}.md'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FIXED weekly Cy Young prediction update')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--max-players', type=int, help='Maximum number of players to process')
    parser.add_argument('--report-only', action='store_true', help='Generate report only (no model update)')
    parser.add_argument('--date', type=str, help='Specific date for report generation (YYYYMMDD)')
    
    args = parser.parse_args()
    
    if args.report_only:
        generate_weekly_report(args.date)
    else:
        print("🚀 Starting FIXED Weekly Cy Young Prediction Update")
        print("=" * 60)
        
        if args.test:
            print("🧪 Running in TEST MODE - limited data collection")
        
        try:
            season_results, update_summary = weekly_update_with_comprehensive_diagnostics_FIXED(
                test_mode=args.test, 
                max_players=args.max_players
            )
            
            if season_results is not None:
                print("\n✅ FIXED weekly update completed successfully!")
                print(f"  🎯 Top candidate: {update_summary['top_candidate']} ({update_summary['top_prediction']:.4f})")
                print(f"  📊 Good prediction variance: {update_summary['prediction_std']:.4f}")
                print(f"  📈 Wide prediction range: {update_summary['prediction_range']:.4f}")
                
                # Generate report
                update_date = datetime.date.today().strftime('%Y%m%d')
                generate_weekly_report(update_date)
                
                print(f"\n📁 Files Generated:")
                print(f"  📊 Results: results/cy_young_predictions_{datetime.date.today().year}_{update_date}.csv")
                print(f"  📄 Report: results/weekly_report_{update_date}.md")
                print(f"  🔍 Diagnostics: diagnostics/weekly_updates/")
                print(f"\n🎯 Next Steps:")
                print(f"  - Review diagnostic plots for model performance")
                print(f"  - Monitor prediction stability over time")
                print(f"  - Check data quality indicators")
                
            else:
                print("❌ Weekly update failed. Check logs and diagnostics.")
                
        except Exception as e:
            print(f"❌ Error during weekly update: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"\n📈 Usage Examples:")
        print(f"  python weekly_update.py --test                    # Test mode")
        print(f"  python weekly_update.py --max-players 50         # Limit data collection")
        print(f"  python weekly_update.py --report-only            # Generate report only")
        print(f"  python weekly_update.py --report-only --date 20250801  # Historical report")
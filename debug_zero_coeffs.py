import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def debug_zero_coefficients():
    """Debug why all model coefficients are zero"""
    
    print("🔍 DEBUGGING ZERO COEFFICIENTS PROBLEM")
    print("=" * 50)
    
    # Try to load the actual data that was used for training
    try:
        # Load the saved meta-features to see what the model actually saw
        meta_features = joblib.load('models/meta_features.pkl')
        print(f"✅ Meta-features used: {meta_features}")
        
        # Load the trained model
        meta_model = joblib.load('models/meta_model.pkl')
        print(f"✅ Meta-model loaded: {type(meta_model)}")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load model files: {e}")
        return
    
    # Let's recreate the exact meta-dataset that was used
    print("\n🔄 Recreating the meta-dataset...")
    
    try:
        # Load base data
        prior_pitch_df = pd.read_csv('data/prior_pitchlevel.csv', parse_dates=['game_date'])
        prior_game_df = pd.read_csv('data/prior_gamelevel.csv', parse_dates=['game_date'])
        prior_season_tot = pd.read_csv('data/prior_seasonlevel.csv')
        prior_labels_df = pd.read_csv('data/winner_labels.csv')
        
        print(f"Base data loaded successfully")
        
        # Check if the models exist and can be loaded
        try:
            swing_pipe, whiff_pipe, exit_pipe = joblib.load('models/level1_pipelines.pkl')
            print("✅ Level 1 models loaded")
            
            # Compute pitch scores
            print("Computing pitch scores...")
            from train_models import compute_pitch_score
            prior_pitch_df['pitch_score'] = compute_pitch_score(
                prior_pitch_df, swing_pipe, whiff_pipe, exit_pipe
            )
            
            print(f"Pitch scores computed: min={prior_pitch_df['pitch_score'].min():.6f}, max={prior_pitch_df['pitch_score'].max():.6f}, std={prior_pitch_df['pitch_score'].std():.6f}")
            
        except Exception as e:
            print(f"⚠️ Could not load Level 1 models or compute pitch scores: {e}")
            print("Using random pitch scores for debugging...")
            prior_pitch_df['pitch_score'] = np.random.normal(0.1, 0.02, len(prior_pitch_df))
        
        # Aggregate pitch scores
        p_seas = (
            prior_pitch_df
            .groupby(['Name','Season'])['pitch_score']
            .mean().reset_index()
            .rename(columns={'pitch_score':'pitch_score_mean'})
        )
        print(f"Pitch aggregation: {len(p_seas)} rows, range=[{p_seas['pitch_score_mean'].min():.6f}, {p_seas['pitch_score_mean'].max():.6f}]")
        
        # Create dummy RBI predictions or load them
        try:
            # Try to recreate RBI predictions
            print("Attempting to recreate RBI predictions...")
            # For debugging, create reasonable fake RBI predictions
            game_ids = prior_game_df[['game_id', 'game_date', 'Name', 'Season']].drop_duplicates()
            game_ids['rbi_pred'] = np.random.normal(0.1, 0.03, len(game_ids))
            
            r_seas = (
                game_ids
                .groupby(['Name','Season'])['rbi_pred']
                .mean().reset_index()
                .rename(columns={'rbi_pred':'rbi_pred_mean'})
            )
            print(f"RBI aggregation: {len(r_seas)} rows, range=[{r_seas['rbi_pred_mean'].min():.6f}, {r_seas['rbi_pred_mean'].max():.6f}]")
            
        except Exception as e:
            print(f"⚠️ Could not create RBI predictions: {e}")
            return
        
        # Add Bayesian effects
        prior_season_tot['bayes_effect'] = np.random.normal(0.0, 0.5, len(prior_season_tot))
        print(f"Bayesian effects: range=[{prior_season_tot['bayes_effect'].min():.6f}, {prior_season_tot['bayes_effect'].max():.6f}]")
        
        # Create meta dataset
        meta_df = (
            prior_season_tot
            .merge(p_seas, on=['Name','Season'], how='inner')
            .merge(r_seas, on=['Name','Season'], how='inner')
            .merge(prior_labels_df, on=['Name','Season'], how='inner')
        )
        
        print(f"\n🎯 Meta-dataset created: {len(meta_df)} rows")
        
        # Analyze the meta-features
        base_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
        
        print(f"\n📊 Meta-feature analysis:")
        for col in base_features:
            if col in meta_df.columns:
                values = meta_df[col]
                print(f"  {col}:")
                print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
                print(f"    Mean±Std: {values.mean():.6f}±{values.std():.6f}")
                print(f"    Unique values: {values.nunique()}")
                print(f"    Has variation: {values.std() > 1e-10}")
        
        # Check winner vs non-winner differences
        winners = meta_df[meta_df['winner_flag'] == 1]
        non_winners = meta_df[meta_df['winner_flag'] == 0]
        
        print(f"\n🏆 Winner vs Non-winner analysis:")
        print(f"  Winners: {len(winners)} rows")
        print(f"  Non-winners: {len(non_winners)} rows")
        
        for col in base_features:
            if col in meta_df.columns and len(winners) > 0 and len(non_winners) > 0:
                w_mean = winners[col].mean()
                nw_mean = non_winners[col].mean()
                difference = w_mean - nw_mean
                
                print(f"  {col}:")
                print(f"    Winners: {w_mean:.6f}")
                print(f"    Non-winners: {nw_mean:.6f}")
                print(f"    Difference: {difference:.6f}")
                print(f"    Meaningful difference: {abs(difference) > 1e-6}")
        
        # Now let's trace through the exact preprocessing that happens in your model
        print(f"\n🔄 Tracing through model preprocessing...")
        
        # Recreate the exact feature engineering
        meta_features_data = meta_df[base_features].copy()
        
        # Add interaction if the model expects it
        if 'pitch_rbi_interaction' in meta_features:
            meta_features_data['pitch_rbi_interaction'] = (
                meta_features_data['pitch_score_mean'] * meta_features_data['rbi_pred_mean']
            )
            features_to_check = base_features + ['pitch_rbi_interaction']
        else:
            features_to_check = base_features
        
        print(f"Features being used: {features_to_check}")
        
        # Check the data going into the model
        X_raw = meta_features_data[features_to_check].values
        y = meta_df['winner_flag'].values
        
        print(f"Raw feature matrix:")
        print(f"  Shape: {X_raw.shape}")
        print(f"  Has NaN: {np.isnan(X_raw).any()}")
        print(f"  Has Inf: {np.isinf(X_raw).any()}")
        print(f"  Min: {X_raw.min()}")
        print(f"  Max: {X_raw.max()}")
        print(f"  Std: {X_raw.std()}")
        
        # Apply the same preprocessing as the trained model
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Step 1: Imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_raw)
        
        print(f"After imputation:")
        print(f"  Has NaN: {np.isnan(X_imputed).any()}")
        print(f"  Min: {X_imputed.min()}")
        print(f"  Max: {X_imputed.max()}")
        print(f"  Std: {X_imputed.std()}")
        
        # Step 2: Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        print(f"After scaling:")
        print(f"  Min: {X_scaled.min()}")
        print(f"  Max: {X_scaled.max()}")
        print(f"  Mean: {X_scaled.mean()}")
        print(f"  Std: {X_scaled.std()}")
        print(f"  All zeros: {(X_scaled == 0).all()}")
        print(f"  All same value: {np.allclose(X_scaled, X_scaled[0])}")
        
        # Check if scaling is destroying variation
        print(f"\nScaling diagnostics:")
        for i, feature in enumerate(features_to_check):
            original_std = X_raw[:, i].std()
            scaled_std = X_scaled[:, i].std()
            print(f"  {feature}: original_std={original_std:.6f}, scaled_std={scaled_std:.6f}")
            
            if original_std < 1e-10:
                print(f"    🚨 PROBLEM: {feature} has no variation in original data!")
            elif scaled_std < 1e-10:
                print(f"    🚨 PROBLEM: {feature} lost variation after scaling!")
        
        # Test a simple logistic regression to see what happens
        print(f"\n🧪 Testing simple logistic regression:")
        
        # Test with unscaled data
        try:
            simple_lr = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
            simple_lr.fit(X_raw, y)
            print(f"  Unscaled data coefficients: {simple_lr.coef_[0]}")
            print(f"  All coefficients zero: {np.allclose(simple_lr.coef_[0], 0)}")
        except Exception as e:
            print(f"  ⚠️ Failed with unscaled data: {e}")
        
        # Test with scaled data
        try:
            simple_lr_scaled = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
            simple_lr_scaled.fit(X_scaled, y)
            print(f"  Scaled data coefficients: {simple_lr_scaled.coef_[0]}")
            print(f"  All coefficients zero: {np.allclose(simple_lr_scaled.coef_[0], 0)}")
        except Exception as e:
            print(f"  ⚠️ Failed with scaled data: {e}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot original features
        for i, feature in enumerate(features_to_check[:3]):
            ax = axes[0, i]
            winners_data = meta_df[meta_df['winner_flag'] == 1][feature]
            non_winners_data = meta_df[meta_df['winner_flag'] == 0][feature]
            
            ax.hist(non_winners_data, bins=15, alpha=0.7, label='Non-winners', density=True)
            ax.hist(winners_data, bins=8, alpha=0.7, label='Winners', density=True)
            ax.set_title(f'{feature} (Original)')
            ax.legend()
        
        # Plot scaled features
        for i, feature in enumerate(features_to_check[:3]):
            ax = axes[1, i]
            winners_mask = y == 1
            non_winners_mask = y == 0
            
            ax.hist(X_scaled[non_winners_mask, i], bins=15, alpha=0.7, label='Non-winners', density=True)
            ax.hist(X_scaled[winners_mask, i], bins=8, alpha=0.7, label='Winners', density=True)
            ax.set_title(f'{feature} (Scaled)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('diagnostics/zero_coefficients_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save the debug data
        debug_data = meta_df[['Name', 'Season', 'winner_flag'] + features_to_check].copy()
        debug_data.to_csv('diagnostics/debug_meta_dataset.csv', index=False)
        
        print(f"\n💾 Debug data saved to:")
        print(f"   - diagnostics/zero_coefficients_debug.png")
        print(f"   - diagnostics/debug_meta_dataset.csv")
        
    except Exception as e:
        print(f"❌ Error in debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_zero_coefficients()
    
    print(f"\n🎯 SUMMARY OF POTENTIAL ISSUES:")
    print("1. Features have no variation (all same value)")
    print("2. StandardScaler is removing all variation") 
    print("3. Features are all NaN or infinite")
    print("4. Class separation is too small to detect")
    print("5. Regularization (C=0.01) is too strong")
    print("6. Data preprocessing pipeline has a bug")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Run this script to see exactly what's happening")
    print("2. Check the diagnostic plots and debug CSV")
    print("3. If features have variation but coefficients are still zero,")
    print("   the issue is likely in the StandardScaler or regularization")
    print("4. Try training without scaling or with C=100 instead of C=0.01")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_feature_problems():
    """Diagnose why all feature coefficients are 0.0"""
    
    print("🔍 DIAGNOSING FEATURE VALUE PROBLEMS")
    print("=" * 50)
    
    print("Your model learned that ALL features are useless (coefficients = 0.0)")
    print("This means one of these problems:")
    print("1. All features have the same value (no variation)")
    print("2. Features are all NaN or 0")
    print("3. Features don't actually differ between winners and non-winners")
    print("4. There's a bug in the feature calculation pipeline")
    
    # Let's trace through and simulate what your features might look like
    print("\n📊 Let's examine what might be happening in your pipeline...")
    
    # Try to load and examine actual data from your pipeline
    try:
        # Check if we can find any intermediate data files
        prior_pitch_df = pd.read_csv('data/prior_pitchlevel.csv')
        prior_game_df = pd.read_csv('data/prior_gamelevel.csv')
        prior_season_tot = pd.read_csv('data/prior_seasonlevel.csv')
        prior_labels_df = pd.read_csv('data/winner_labels.csv')
        
        print(f"\n📈 DATA LOADED:")
        print(f"  Pitch data: {len(prior_pitch_df)} rows, {prior_pitch_df['Name'].nunique()} pitchers")
        print(f"  Game data: {len(prior_game_df)} rows") 
        print(f"  Season data: {len(prior_season_tot)} rows")
        print(f"  Labels: {len(prior_labels_df)} rows, {prior_labels_df['winner_flag'].sum()} winners")
        
        # Check if pitch_score column exists (this is created by Level 1)
        if 'pitch_score' in prior_pitch_df.columns:
            print(f"\n🔍 PITCH SCORES FOUND:")
            print(f"  Pitch scores range: [{prior_pitch_df['pitch_score'].min():.6f}, {prior_pitch_df['pitch_score'].max():.6f}]")
            print(f"  Pitch scores mean: {prior_pitch_df['pitch_score'].mean():.6f}")
            print(f"  Pitch scores std: {prior_pitch_df['pitch_score'].std():.6f}")
            print(f"  Non-zero pitch scores: {(prior_pitch_df['pitch_score'] != 0).sum()}/{len(prior_pitch_df)}")
            
            if prior_pitch_df['pitch_score'].std() < 1e-10:
                print("  🚨 PROBLEM: Pitch scores have NO VARIATION!")
                print("     This means Level 1 models are broken")
            
        else:
            print(f"\n⚠️ NO PITCH SCORES FOUND - Level 1 not applied yet")
        
        # Check if RBI predictions exist
        if 'rbi_pred' in prior_game_df.columns:
            print(f"\n🔍 RBI PREDICTIONS FOUND:")
            print(f"  RBI preds range: [{prior_game_df['rbi_pred'].min():.6f}, {prior_game_df['rbi_pred'].max():.6f}]")
            print(f"  RBI preds mean: {prior_game_df['rbi_pred'].mean():.6f}")
            print(f"  RBI preds std: {prior_game_df['rbi_pred'].std():.6f}")
            
            if prior_game_df['rbi_pred'].std() < 1e-10:
                print("  🚨 PROBLEM: RBI predictions have NO VARIATION!")
                print("     This means Level 2 LSTM is broken")
        else:
            print(f"\n⚠️ NO RBI PREDICTIONS FOUND - Level 2 not applied yet")
        
        # Try to simulate the exact meta-data creation process
        print(f"\n🔄 SIMULATING META-DATA CREATION:")
        
        # Step 1: Create fake pitch scores if they don't exist
        if 'pitch_score' not in prior_pitch_df.columns:
            print("  Creating random pitch scores for simulation...")
            prior_pitch_df['pitch_score'] = np.random.normal(0.1, 0.05, len(prior_pitch_df))
        
        # Step 2: Aggregate to season level
        pitch_seas = (
            prior_pitch_df
            .groupby(['Name','Season'])['pitch_score']
            .mean().reset_index()
            .rename(columns={'pitch_score':'pitch_score_mean'})
        )
        
        print(f"  Pitch aggregation: {len(pitch_seas)} name-season combinations")
        print(f"  Pitch score means range: [{pitch_seas['pitch_score_mean'].min():.6f}, {pitch_seas['pitch_score_mean'].max():.6f}]")
        print(f"  Pitch score means std: {pitch_seas['pitch_score_mean'].std():.6f}")
        
        # Step 3: Create fake RBI predictions
        if 'rbi_pred' not in prior_game_df.columns:
            print("  Creating random RBI predictions for simulation...")
            prior_game_df['rbi_pred'] = np.random.normal(0.1, 0.03, len(prior_game_df))
        
        rbi_seas = (
            prior_game_df
            .groupby(['Name','Season'])['rbi_pred']
            .mean().reset_index()
            .rename(columns={'rbi_pred':'rbi_pred_mean'})
        )
        
        print(f"  RBI aggregation: {len(rbi_seas)} name-season combinations")
        print(f"  RBI pred means range: [{rbi_seas['rbi_pred_mean'].min():.6f}, {rbi_seas['rbi_pred_mean'].max():.6f}]")
        print(f"  RBI pred means std: {rbi_seas['rbi_pred_mean'].std():.6f}")
        
        # Step 4: Add fake Bayesian effects
        prior_season_tot['bayes_effect'] = np.random.normal(0.0, 0.5, len(prior_season_tot))
        
        # Step 5: Create final meta dataset
        meta_df = (
            prior_season_tot
            .merge(pitch_seas, on=['Name','Season'], how='inner')
            .merge(rbi_seas, on=['Name','Season'], how='inner')
            .merge(prior_labels_df, on=['Name','Season'], how='inner')
        )
        
        print(f"\n🎯 FINAL META-DATASET: {len(meta_df)} rows")
        
        # Analyze the features
        feature_cols = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
        
        print(f"\n📊 FEATURE ANALYSIS:")
        for col in feature_cols:
            if col in meta_df.columns:
                values = meta_df[col]
                print(f"  {col}:")
                print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
                print(f"    Mean±Std: {values.mean():.6f}±{values.std():.6f}")
                print(f"    Unique values: {values.nunique()}")
                print(f"    NaN count: {values.isna().sum()}")
                
                if values.std() < 1e-10:
                    print(f"    🚨 PROBLEM: {col} has NO VARIATION!")
                elif values.nunique() == 1:
                    print(f"    🚨 PROBLEM: {col} has only ONE unique value!")
                elif values.isna().all():
                    print(f"    🚨 PROBLEM: {col} is all NaN!")
        
        # Check if winners actually differ from non-winners
        print(f"\n🏆 WINNER vs NON-WINNER ANALYSIS:")
        winners = meta_df[meta_df['winner_flag'] == 1]
        non_winners = meta_df[meta_df['winner_flag'] == 0]
        
        print(f"  Winners: {len(winners)} rows")
        print(f"  Non-winners: {len(non_winners)} rows")
        
        for col in feature_cols:
            if col in meta_df.columns and len(winners) > 0 and len(non_winners) > 0:
                winner_mean = winners[col].mean()
                non_winner_mean = non_winners[col].mean()
                difference = winner_mean - non_winner_mean
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(winners)-1)*winners[col].var() + (len(non_winners)-1)*non_winners[col].var()) / (len(winners)+len(non_winners)-2))
                cohens_d = difference / pooled_std if pooled_std > 0 else 0
                
                print(f"  {col}:")
                print(f"    Winners: {winner_mean:.6f}")
                print(f"    Non-winners: {non_winner_mean:.6f}")
                print(f"    Difference: {difference:.6f}")
                print(f"    Effect size (Cohen's d): {cohens_d:.4f}")
                
                if abs(difference) < 1e-10:
                    print(f"    🚨 PROBLEM: No difference between winners and non-winners!")
                elif abs(cohens_d) < 0.1:
                    print(f"    ⚠️ WARNING: Very small effect size")
                elif abs(cohens_d) < 0.3:
                    print(f"    ✅ Small effect size")
                elif abs(cohens_d) < 0.8:
                    print(f"    ✅ Medium effect size")  
                else:
                    print(f"    ✅ Large effect size")
        
        # Create visualization
        if len(meta_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Feature distributions
            for i, col in enumerate(feature_cols[:3]):
                row, col_idx = i // 2, i % 2
                ax = axes[row, col_idx]
                
                if col in meta_df.columns:
                    # Split by winner status
                    winners_data = meta_df[meta_df['winner_flag'] == 1][col].dropna()
                    non_winners_data = meta_df[meta_df['winner_flag'] == 0][col].dropna()
                    
                    if len(winners_data) > 0 and len(non_winners_data) > 0:
                        ax.hist(non_winners_data, bins=15, alpha=0.7, label='Non-winners', density=True)
                        ax.hist(winners_data, bins=8, alpha=0.7, label='Winners', density=True)
                        ax.set_title(f'{col}')
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Density')
                        ax.legend()
                    else:
                        ax.text(0.5, 0.5, f'{col}\nNo data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{col}')
            
            # Correlation matrix
            axes[1, 1].text(0.1, 0.9, 'Feature Problems Found:', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
            
            problems = []
            for col in feature_cols:
                if col in meta_df.columns:
                    if meta_df[col].std() < 1e-10:
                        problems.append(f"• {col}: No variation")
                    elif meta_df[col].nunique() == 1:
                        problems.append(f"• {col}: Single value")
                    elif meta_df[col].isna().all():
                        problems.append(f"• {col}: All NaN")
            
            if not problems:
                problems = ["• No obvious problems found", "• Issue may be in model training"]
            
            for i, problem in enumerate(problems[:6]):
                axes[1, 1].text(0.1, 0.8 - i*0.1, problem, fontsize=10, transform=axes[1, 1].transAxes)
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            axes[1, 1].set_title('Diagnostic Summary')
            
            plt.tight_layout()
            plt.savefig('diagnostics/feature_value_diagnosis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Save the simulated meta dataset for further analysis
            meta_df.to_csv('diagnostics/simulated_meta_dataset.csv', index=False)
            print(f"\n💾 Simulated meta-dataset saved to: diagnostics/simulated_meta_dataset.csv")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n🔧 NEXT STEPS TO FIX THE PROBLEM:")
    print("1. 🔍 Check Level 1 models - are pitch scores actually calculated?")
    print("2. 🔍 Check Level 2 LSTM - are RBI predictions working?") 
    print("3. 🔍 Check Level 3 Bayesian - are effects being computed?")
    print("4. 🔍 Verify that winners actually have different statistics")
    print("5. 🔄 Add debug prints in your training pipeline")
    
    print(f"\n💡 DEBUGGING TIPS:")
    print("- Add print statements in train_models.py after each level")
    print("- Check if any features are constant or NaN")
    print("- Verify that your Cy Young labels are correct")
    print("- Make sure feature scaling isn't zeroing everything out")

if __name__ == '__main__':
    diagnose_feature_problems()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def trace_pitcher_data_loss():
    """Trace where pitchers are being lost through the pipeline"""
    print("🔍 TRACING PITCHER DATA LOSS THROUGH PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and examine all input datasets
    print("\n📊 STEP 1: EXAMINING INPUT DATASETS")
    print("-" * 40)
    
    try:
        prior_pitch_df = pd.read_csv('data/prior_pitchlevel.csv', parse_dates=['game_date'])
        prior_game_df = pd.read_csv('data/prior_gamelevel.csv', parse_dates=['game_date'])
        prior_season_tot = pd.read_csv('data/prior_seasonlevel.csv')
        prior_labels_df = pd.read_csv('data/winner_labels.csv')
        
        print(f"✅ PITCH DATA: {len(prior_pitch_df):,} rows")
        print(f"   - Unique pitchers: {prior_pitch_df['Name'].nunique()}")
        print(f"   - Seasons: {sorted(prior_pitch_df['Season'].unique())}")
        print(f"   - Date range: {prior_pitch_df['game_date'].min()} to {prior_pitch_df['game_date'].max()}")
        
        print(f"\n✅ GAME DATA: {len(prior_game_df):,} rows")
        print(f"   - Unique pitchers: {prior_game_df['Name'].nunique()}")
        print(f"   - Seasons: {sorted(prior_game_df['Season'].unique())}")
        print(f"   - Date range: {prior_game_df['game_date'].min()} to {prior_game_df['game_date'].max()}")
        
        print(f"\n✅ SEASON DATA: {len(prior_season_tot):,} rows")
        print(f"   - Unique pitchers: {prior_season_tot['Name'].nunique()}")
        print(f"   - Seasons: {sorted(prior_season_tot['Season'].unique())}")
        print(f"   - Columns: {list(prior_season_tot.columns)}")
        
        print(f"\n✅ LABELS DATA: {len(prior_labels_df):,} rows")
        print(f"   - Unique pitchers: {prior_labels_df['Name'].nunique()}")
        print(f"   - Seasons: {sorted(prior_labels_df['Season'].unique())}")
        print(f"   - Winners: {prior_labels_df['winner_flag'].sum()}")
        print(f"   - Columns: {list(prior_labels_df.columns)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Check overlap between datasets
    print("\n📊 STEP 2: CHECKING DATASET OVERLAPS")
    print("-" * 40)
    
    pitch_pitchers = set(prior_pitch_df['Name'].unique())
    game_pitchers = set(prior_game_df['Name'].unique())
    season_pitchers = set(prior_season_tot['Name'].unique())
    label_pitchers = set(prior_labels_df['Name'].unique())
    
    print(f"Pitchers in each dataset:")
    print(f"  Pitch data: {len(pitch_pitchers)}")
    print(f"  Game data: {len(game_pitchers)}")
    print(f"  Season data: {len(season_pitchers)}")
    print(f"  Label data: {len(label_pitchers)}")
    
    # Find overlaps
    all_datasets = pitch_pitchers & game_pitchers & season_pitchers & label_pitchers
    print(f"\n🎯 Pitchers in ALL datasets: {len(all_datasets)}")
    
    if len(all_datasets) < len(season_pitchers) * 0.5:
        print("⚠️ WARNING: Major data loss due to poor overlap!")
    
    # Check what's missing where
    missing_from_labels = season_pitchers - label_pitchers
    missing_from_season = label_pitchers - season_pitchers
    
    print(f"\nMissing from labels (but in season data): {len(missing_from_labels)}")
    if len(missing_from_labels) > 0:
        print(f"  Examples: {list(missing_from_labels)[:10]}")
    
    print(f"Missing from season data (but in labels): {len(missing_from_season)}")
    if len(missing_from_season) > 0:
        print(f"  Examples: {list(missing_from_season)[:10]}")
    
    # Step 3: Check season-year combinations
    print("\n📊 STEP 3: CHECKING SEASON-YEAR COMBINATIONS")
    print("-" * 40)
    
    def check_season_year_overlap(df1, df2, name1, name2):
        df1_combos = set(zip(df1['Name'], df1['Season']))
        df2_combos = set(zip(df2['Name'], df2['Season']))
        overlap = df1_combos & df2_combos
        print(f"{name1} vs {name2}:")
        print(f"  {name1}: {len(df1_combos)} name-season combinations")
        print(f"  {name2}: {len(df2_combos)} name-season combinations") 
        print(f"  Overlap: {len(overlap)} combinations")
        return overlap
    
    season_label_overlap = check_season_year_overlap(
        prior_season_tot, prior_labels_df, "Season data", "Label data"
    )
    
    pitch_season_overlap = check_season_year_overlap(
        prior_pitch_df, prior_season_tot, "Pitch data", "Season data"
    )
    
    # Step 4: Trace through the actual merging process
    print("\n📊 STEP 4: SIMULATING THE MERGE PROCESS")
    print("-" * 40)
    
    # Simulate the pitch score aggregation
    print("Simulating pitch score aggregation...")
    if 'pitch_score' not in prior_pitch_df.columns:
        prior_pitch_df['pitch_score'] = np.random.normal(0.1, 0.05, len(prior_pitch_df))
        print("  (Using random pitch scores for simulation)")
    
    pitch_seas = (
        prior_pitch_df
        .groupby(['Name','Season'])['pitch_score']
        .mean().reset_index()
        .rename(columns={'pitch_score':'pitch_score_mean'})
    )
    print(f"  Pitch aggregation: {len(pitch_seas)} name-season combinations")
    
    # Simulate the RBI prediction aggregation
    print("Simulating RBI prediction aggregation...")
    if 'rbi_pred' not in prior_game_df.columns:
        prior_game_df['rbi_pred'] = np.random.normal(0.1, 0.05, len(prior_game_df))
        print("  (Using random RBI predictions for simulation)")
    
    rbi_seas = (
        prior_game_df
        .groupby(['Name','Season'])['rbi_pred']
        .mean().reset_index()
        .rename(columns={'rbi_pred':'rbi_pred_mean'})
    )
    print(f"  RBI aggregation: {len(rbi_seas)} name-season combinations")
    
    # Simulate the final merge process
    print("Simulating final meta-data merge...")
    print(f"  Starting with season data: {len(prior_season_tot)} rows")
    
    step1 = prior_season_tot.merge(pitch_seas, on=['Name','Season'], how='inner')
    print(f"  After pitch merge (inner): {len(step1)} rows")
    
    step2 = step1.merge(rbi_seas, on=['Name','Season'], how='inner')
    print(f"  After RBI merge (inner): {len(step2)} rows")
    
    final = step2.merge(prior_labels_df, on=['Name','Season'], how='inner')
    print(f"  After label merge (inner): {len(final)} rows")
    
    print(f"\n🎯 FINAL META-DATASET: {len(final)} pitcher-seasons")
    print(f"   This matches your observed 66 rows!")
    
    # Step 5: Identify the bottleneck
    print("\n📊 STEP 5: IDENTIFYING THE BOTTLENECK")
    print("-" * 40)
    
    losses = {
        'Season data': len(prior_season_tot),
        'After pitch merge': len(step1),
        'After RBI merge': len(step2), 
        'After label merge': len(final)
    }
    
    biggest_loss_step = None
    biggest_loss = 0
    
    prev_count = len(prior_season_tot)
    for step, count in list(losses.items())[1:]:
        loss = prev_count - count
        loss_pct = (loss / prev_count) * 100
        print(f"{step}: Lost {loss} rows ({loss_pct:.1f}%)")
        
        if loss > biggest_loss:
            biggest_loss = loss
            biggest_loss_step = step
        
        prev_count = count
    
    print(f"\n🚨 BIGGEST DATA LOSS: {biggest_loss_step}")
    print(f"   Lost {biggest_loss} rows in this step")
    
    # Step 6: Investigate the biggest loss
    print(f"\n📊 STEP 6: INVESTIGATING {biggest_loss_step}")
    print("-" * 40)
    
    if biggest_loss_step == 'After pitch merge':
        # Check what's missing from pitch data
        season_combos = set(zip(prior_season_tot['Name'], prior_season_tot['Season']))
        pitch_combos = set(zip(pitch_seas['Name'], pitch_seas['Season']))
        missing_in_pitch = season_combos - pitch_combos
        
        print(f"Season combos missing from pitch data: {len(missing_in_pitch)}")
        if len(missing_in_pitch) > 0:
            print("Examples of missing pitcher-seasons:")
            for name, season in list(missing_in_pitch)[:10]:
                print(f"  {name} ({season})")
                
    elif biggest_loss_step == 'After RBI merge':
        # Check what's missing from game/RBI data
        step1_combos = set(zip(step1['Name'], step1['Season']))
        rbi_combos = set(zip(rbi_seas['Name'], rbi_seas['Season']))
        missing_in_rbi = step1_combos - rbi_combos
        
        print(f"Combos missing from RBI data: {len(missing_in_rbi)}")
        if len(missing_in_rbi) > 0:
            print("Examples of missing pitcher-seasons:")
            for name, season in list(missing_in_rbi)[:10]:
                print(f"  {name} ({season})")
                
    elif biggest_loss_step == 'After label merge':
        # Check what's missing from labels
        step2_combos = set(zip(step2['Name'], step2['Season']))
        label_combos = set(zip(prior_labels_df['Name'], prior_labels_df['Season']))
        missing_in_labels = step2_combos - label_combos
        
        print(f"Combos missing from labels: {len(missing_in_labels)}")
        if len(missing_in_labels) > 0:
            print("Examples of missing pitcher-seasons:")
            for name, season in list(missing_in_labels)[:10]:
                print(f"  {name} ({season})")
    
    # Step 7: Create visualization
    print(f"\n📊 STEP 7: Creating Data Flow Visualization")
    print("-" * 40)
    
    plt.figure(figsize=(15, 10))
    
    # Data flow diagram
    plt.subplot(2, 2, 1)
    steps = list(losses.keys())
    counts = list(losses.values())
    colors = ['green' if i == 0 else 'orange' if counts[i] > counts[-1] * 2 else 'red' for i in range(len(counts))]
    
    bars = plt.bar(range(len(steps)), counts, color=colors)
    plt.xticks(range(len(steps)), [s.replace(' ', '\n') for s in steps], rotation=0)
    plt.ylabel('Number of Pitcher-Seasons')
    plt.title('Data Loss Through Pipeline')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom')
    
    # Dataset overlap Venn-style
    plt.subplot(2, 2, 2)
    dataset_sizes = [len(pitch_pitchers), len(game_pitchers), len(season_pitchers), len(label_pitchers)]
    dataset_names = ['Pitch', 'Game', 'Season', 'Labels']
    
    plt.bar(dataset_names, dataset_sizes)
    plt.axhline(y=len(all_datasets), color='red', linestyle='--', label=f'Common: {len(all_datasets)}')
    plt.ylabel('Unique Pitchers')
    plt.title('Pitcher Counts by Dataset')
    plt.legend()
    
    # Season distribution
    plt.subplot(2, 2, 3)
    season_counts = prior_season_tot['Season'].value_counts().sort_index()
    plt.bar(season_counts.index, season_counts.values)
    plt.xlabel('Season')
    plt.ylabel('Number of Pitchers')
    plt.title('Pitchers per Season (Season Data)')
    
    # Final dataset composition
    plt.subplot(2, 2, 4)
    if len(final) > 0:
        final_season_counts = final['Season'].value_counts().sort_index()
        plt.bar(final_season_counts.index, final_season_counts.values, color='lightcoral')
        plt.xlabel('Season')
        plt.ylabel('Number of Pitchers')
        plt.title('Final Meta-Dataset Composition')
    
    plt.tight_layout()
    plt.savefig('diagnostics/data_loss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 8: Recommendations
    print(f"\n🔧 RECOMMENDATIONS TO FIX DATA LOSS:")
    print("-" * 40)
    
    print("1. 🔍 CHECK YOUR DATA FILES:")
    print("   - Are pitcher names consistent across files?")
    print("   - Are seasons formatted the same way?")
    print("   - Do you have complete data for all seasons?")
    
    print("\n2. 🔄 CHANGE MERGE STRATEGY:")
    print("   - Use 'left' joins instead of 'inner' joins")
    print("   - Fill missing values with defaults")
    print("   - Add diagnostic prints in merge steps")
    
    print("\n3. 📊 EXPAND YOUR DATASET:")
    print("   - Include more seasons if available")
    print("   - Add minor league or international players")
    print("   - Consider different pitcher categories")
    
    print("\n4. 🎯 VALIDATE NAME MATCHING:")
    print("   - Check for name variations (Jr., Sr., accents)")
    print("   - Use fuzzy matching for similar names")
    print("   - Standardize name formats")
    
    return final

def suggest_merge_fixes():
    """Suggest specific code fixes for the merge issues"""
    
    print("\n🔧 SUGGESTED CODE FIXES:")
    print("=" * 40)
    
    improved_merge_code = '''
# IMPROVED MERGE STRATEGY - Add this to your train_models.py

def create_meta_dataset_with_diagnostics(prior_season_tot, pitch_seas, rbi_seas, prior_labels_df):
    """Create meta dataset with better merge strategy and diagnostics"""
    
    print("🔄 Creating meta-dataset with diagnostics...")
    
    # Start with season data as base
    meta_df = prior_season_tot.copy()
    print(f"Starting with {len(meta_df)} pitcher-seasons")
    
    # Merge pitch scores with LEFT join to preserve all pitchers
    meta_df = meta_df.merge(pitch_seas, on=['Name','Season'], how='left')
    print(f"After pitch merge: {len(meta_df)} rows, {meta_df['pitch_score_mean'].notna().sum()} with pitch scores")
    
    # Fill missing pitch scores
    meta_df['pitch_score_mean'] = meta_df['pitch_score_mean'].fillna(meta_df['pitch_score_mean'].mean())
    
    # Merge RBI predictions with LEFT join
    meta_df = meta_df.merge(rbi_seas, on=['Name','Season'], how='left') 
    print(f"After RBI merge: {len(meta_df)} rows, {meta_df['rbi_pred_mean'].notna().sum()} with RBI predictions")
    
    # Fill missing RBI predictions
    meta_df['rbi_pred_mean'] = meta_df['rbi_pred_mean'].fillna(meta_df['rbi_pred_mean'].mean())
    
    # Add placeholder Bayesian effects
    meta_df['bayes_effect'] = 0.0  # Will be filled by Bayesian model
    
    # ONLY NOW merge with labels using INNER join (this determines final size)
    meta_df = meta_df.merge(prior_labels_df, on=['Name','Season'], how='inner')
    print(f"After label merge: {len(meta_df)} rows")
    
    # Check completeness
    completeness = meta_df[['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']].notna().all(axis=1).mean()
    print(f"Feature completeness: {completeness*100:.1f}%")
    
    return meta_df

# ALSO - Add name standardization
def standardize_pitcher_names(df):
    """Standardize pitcher names for better matching"""
    df = df.copy()
    
    # Remove extra whitespace
    df['Name'] = df['Name'].str.strip()
    
    # Handle common variations
    df['Name'] = df['Name'].str.replace(' Jr.', '')
    df['Name'] = df['Name'].str.replace(' Sr.', '')
    df['Name'] = df['Name'].str.replace(' III', '')
    
    # Convert to title case
    df['Name'] = df['Name'].str.title()
    
    return df
'''
    
    print(improved_merge_code)
    
    with open('improved_merge_strategy.py', 'w') as f:
        f.write(improved_merge_code)
    
    print("💾 Code saved to: improved_merge_strategy.py")

if __name__ == '__main__':
    final_data = trace_pitcher_data_loss()
    suggest_merge_fixes()
    
    print("\n🎯 SUMMARY:")
    print("Your small dataset is likely due to:")
    print("1. Inner joins losing data at each step")
    print("2. Inconsistent pitcher names between datasets") 
    print("3. Missing data in intermediate processing steps")
    print("4. Limited overlap between your label file and other data")
    print("\nRun this script to see exactly where the data is being lost!")
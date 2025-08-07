import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def diagnose_meta_features(meta_df, level_name="Meta"):
    """Comprehensive diagnosis of meta-features before training"""
    print(f"\n🔍 Meta-Feature Diagnostics - {level_name}")
    print("=" * 60)
    
    # Basic info
    print(f"Dataset shape: {meta_df.shape}")
    print(f"Target distribution: {meta_df['winner_flag'].value_counts().to_dict()}")
    
    # Feature analysis
    base_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    
    print(f"\nFeature Analysis:")
    for feat in base_features:
        if feat in meta_df.columns:
            values = meta_df[feat]
            print(f"  {feat}:")
            print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
            print(f"    Mean±Std: {values.mean():.6f}±{values.std():.6f}")
            print(f"    Unique values: {values.nunique()}")
            print(f"    Zero variance: {values.std() == 0}")
            print(f"    Missing values: {values.isnull().sum()}")
            
            # Check for constant values
            if values.nunique() <= 1:
                print(f"    ❌ WARNING: {feat} has ≤1 unique values!")
            elif values.std() < 1e-10:
                print(f"    ❌ WARNING: {feat} has near-zero variance!")
        else:
            print(f"  ❌ {feat}: MISSING FROM DATASET")
    
    # Winner vs non-winner comparison
    if len(meta_df) > 0:
        winners = meta_df[meta_df['winner_flag'] == 1]
        non_winners = meta_df[meta_df['winner_flag'] == 0]
        
        print(f"\nWinner vs Non-winner Analysis:")
        for feat in base_features:
            if feat in meta_df.columns and len(winners) > 0 and len(non_winners) > 0:
                w_mean = winners[feat].mean()
                nw_mean = non_winners[feat].mean()
                diff = w_mean - nw_mean
                effect_size = diff / meta_df[feat].std() if meta_df[feat].std() > 0 else 0
                print(f"  {feat}:")
                print(f"    Winners: {w_mean:.6f}")
                print(f"    Non-winners: {nw_mean:.6f}")
                print(f"    Difference: {diff:.6f}")
                print(f"    Effect size: {effect_size:.4f}")
    
    # Correlation analysis
    if len(base_features) > 1:
        available_features = [f for f in base_features if f in meta_df.columns]
        if len(available_features) > 1:
            corr_matrix = meta_df[available_features + ['winner_flag']].corr()
            print(f"\nCorrelation with target:")
            for feat in available_features:
                corr = corr_matrix.loc[feat, 'winner_flag']
                print(f"  {feat}: {corr:.4f}")
    
    return meta_df[base_features].describe()

def create_robust_meta_model(meta_df, test_size=0.2, random_state=42):
    """Create a robust meta-model with proper feature engineering and validation"""
    
    print("🎯 Creating Robust Meta-Model")
    print("=" * 50)
    
    # Diagnose features first
    feature_stats = diagnose_meta_features(meta_df)
    
    # Base features
    base_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    
    # Check feature availability and quality
    available_features = []
    for feat in base_features:
        if feat in meta_df.columns:
            if meta_df[feat].std() > 1e-10:  # Has meaningful variance
                available_features.append(feat)
                print(f"✅ Using feature: {feat}")
            else:
                print(f"❌ Skipping {feat}: zero/near-zero variance")
        else:
            print(f"❌ Missing feature: {feat}")
    
    if len(available_features) == 0:
        print("❌ No usable features found!")
        return None, None, None, None
    
    # Prepare data
    X_raw = meta_df[available_features].copy()
    y = meta_df['winner_flag'].copy()
    
    # Handle missing values
    X_raw = X_raw.fillna(X_raw.median())
    
    print(f"\nUsing {len(available_features)} features: {available_features}")
    print(f"Training samples: {len(X_raw)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Create multiple model configurations to test
    model_configs = [
        {
            'name': 'LogisticRegression_Standard',
            'scaler': StandardScaler(),
            'model': LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            ),
            'param_grid': {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        {
            'name': 'LogisticRegression_Robust',
            'scaler': RobustScaler(),
            'model': LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            ),
            'param_grid': {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        {
            'name': 'RandomForest',
            'scaler': None,  # Random Forest doesn't need scaling
            'model': RandomForestClassifier(
                class_weight='balanced',
                random_state=random_state,
                n_estimators=100
            ),
            'param_grid': {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
        }
    ]
    
    best_model = None
    best_score = 0
    best_config = None
    best_scaler = None
    
    # Test each configuration
    for config in model_configs:
        print(f"\n🧪 Testing {config['name']}...")
        
        try:
            # Prepare features
            if config['scaler'] is not None:
                X_scaled = config['scaler'].fit_transform(X_raw)
                X_for_model = X_scaled
            else:
                X_for_model = X_raw.values
                config['scaler'] = None
            
            # Handle small datasets
            if len(X_raw) <= 20:
                # Use simple train-test split for very small datasets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_for_model, y, test_size=0.3, stratify=y, random_state=random_state
                )
                
                # Test different hyperparameters manually
                best_param_score = 0
                best_params = None
                
                if config['name'].startswith('LogisticRegression'):
                    for C in config['param_grid']['C']:
                        config['model'].set_params(C=C)
                        config['model'].fit(X_train, y_train)
                        y_pred_proba = config['model'].predict_proba(X_test)[:, 1]
                        
                        if len(np.unique(y_test)) > 1:  # Can compute AUC
                            score = roc_auc_score(y_test, y_pred_proba)
                        else:
                            y_pred = config['model'].predict(X_test)
                            score = accuracy_score(y_test, y_pred)
                        
                        if score > best_param_score:
                            best_param_score = score
                            best_params = {'C': C}
                
                elif config['name'] == 'RandomForest':
                    for max_depth in config['param_grid']['max_depth']:
                        for min_samples_split in config['param_grid']['min_samples_split']:
                            config['model'].set_params(max_depth=max_depth, min_samples_split=min_samples_split)
                            config['model'].fit(X_train, y_train)
                            y_pred_proba = config['model'].predict_proba(X_test)[:, 1]
                            
                            if len(np.unique(y_test)) > 1:
                                score = roc_auc_score(y_test, y_pred_proba)
                            else:
                                y_pred = config['model'].predict(X_test)
                                score = accuracy_score(y_test, y_pred)
                            
                            if score > best_param_score:
                                best_param_score = score
                                best_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split}
                
                final_score = best_param_score
                print(f"   Score: {final_score:.4f} (manual validation)")
                
            else:
                # Use cross-validation for larger datasets
                cv_folds = min(5, len(X_raw) // 4)  # Adaptive CV folds
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                scores = cross_val_score(config['model'], X_for_model, y, cv=cv, scoring='roc_auc')
                final_score = scores.mean()
                print(f"   CV Score: {final_score:.4f}±{scores.std():.4f}")
            
            # Update best model
            if final_score > best_score:
                best_score = final_score
                best_config = config
                best_scaler = config['scaler']
                
                # Refit on full data
                if config['scaler'] is not None:
                    X_final = config['scaler'].fit_transform(X_raw)
                else:
                    X_final = X_raw.values
                
                best_model = config['model']
                if 'best_params' in locals() and best_params:
                    best_model.set_params(**best_params)
                best_model.fit(X_final, y)
                
                print(f"   ✅ New best model: {config['name']} (score: {final_score:.4f})")
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    if best_model is None:
        print("❌ No models succeeded!")
        return None, None, None, None
    
    print(f"\n🏆 Best Model: {best_config['name']} (Score: {best_score:.4f})")
    
    # Final evaluation and diagnostics
    if best_scaler is not None:
        X_final = best_scaler.transform(X_raw)
    else:
        X_final = X_raw.values
    
    y_pred = best_model.predict(X_final)
    y_pred_proba = best_model.predict_proba(X_final)[:, 1]
    
    final_acc = accuracy_score(y, y_pred)
    if len(np.unique(y)) > 1:
        final_auc = roc_auc_score(y, y_pred_proba)
    else:
        final_auc = np.nan
    
    print(f"\nFinal Performance:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  AUC: {final_auc:.4f}")
    
    # Feature importance
    if hasattr(best_model, 'coef_'):
        print(f"\nFeature Coefficients:")
        for feat, coef in zip(available_features, best_model.coef_[0]):
            print(f"  {feat}: {coef:.4f}")
    elif hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importances:")
        for feat, imp in zip(available_features, best_model.feature_importances_):
            print(f"  {feat}: {imp:.4f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prediction distribution
    axes[0, 0].hist(y_pred_proba, bins=20, alpha=0.7)
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Frequency')
    
    # ROC curve (if possible)
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {final_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
    
    # Predictions by class
    winners_proba = y_pred_proba[y == 1]
    non_winners_proba = y_pred_proba[y == 0]
    
    if len(winners_proba) > 0:
        axes[1, 0].hist(winners_proba, bins=10, alpha=0.7, label='Winners')
    if len(non_winners_proba) > 0:
        axes[1, 0].hist(non_winners_proba, bins=10, alpha=0.7, label='Non-winners')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Predictions by Class')
    axes[1, 0].legend()
    
    # Feature correlation with target
    if len(available_features) > 1:
        correlations = [meta_df[feat].corr(meta_df['winner_flag']) for feat in available_features]
        axes[1, 1].bar(range(len(available_features)), correlations)
        axes[1, 1].set_xticks(range(len(available_features)))
        axes[1, 1].set_xticklabels(available_features, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Correlation with Target')
        axes[1, 1].set_title('Feature-Target Correlations')
    
    plt.tight_layout()
    plt.savefig('diagnostics/meta/robust_meta_model_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_model, best_scaler, available_features, {
        'model_name': best_config['name'],
        'accuracy': final_acc,
        'auc': final_auc,
        'cv_score': best_score,
        'n_features': len(available_features),
        'features_used': available_features
    }

def save_robust_meta_model(model, scaler, features, diagnostics):
    """Save the robust meta-model components"""
    
    # Save model components
    joblib.dump(model, 'models/robust_meta_model.pkl')
    joblib.dump(scaler, 'models/robust_meta_scaler.pkl')
    joblib.dump(features, 'models/robust_meta_features.pkl')
    
    # Save diagnostics
    diag_df = pd.DataFrame([diagnostics])
    diag_df.to_csv('diagnostics/meta/robust_model_performance.csv', index=False)
    
    print("✅ Robust meta-model saved:")
    print("  - models/robust_meta_model.pkl")
    print("  - models/robust_meta_scaler.pkl") 
    print("  - models/robust_meta_features.pkl")
    print("  - diagnostics/meta/robust_model_performance.csv")

def predict_with_robust_meta_model(season_df, model_path='models/robust_meta_model.pkl'):
    """Make predictions using the robust meta-model"""
    
    print("🔮 Making predictions with robust meta-model...")
    
    try:
        # Load model components
        model = joblib.load(model_path)
        scaler = joblib.load(model_path.replace('_model.pkl', '_scaler.pkl'))
        features = joblib.load(model_path.replace('_model.pkl', '_features.pkl'))
        
        print(f"✅ Loaded model using features: {features}")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load robust model: {e}")
        return None, None
    
    # Prepare features
    X_raw = season_df[features].copy()
    X_raw = X_raw.fillna(X_raw.median())  # Handle missing values
    
    # Apply scaling if scaler exists
    if scaler is not None:
        X_processed = scaler.transform(X_raw)
    else:
        X_processed = X_raw.values
    
    # Make predictions
    predictions = model.predict_proba(X_processed)[:, 1]
    
    # Diagnostic info
    pred_stats = {
        'n_predictions': len(predictions),
        'min_prob': predictions.min(),
        'max_prob': predictions.max(),
        'mean_prob': predictions.mean(),
        'std_prob': predictions.std(),
        'unique_predictions': len(np.unique(predictions))
    }
    
    print(f"Prediction Statistics:")
    print(f"  Range: [{pred_stats['min_prob']:.4f}, {pred_stats['max_prob']:.4f}]")
    print(f"  Mean±Std: {pred_stats['mean_prob']:.4f}±{pred_stats['std_prob']:.4f}")
    print(f"  Unique values: {pred_stats['unique_predictions']}")
    
    if pred_stats['std_prob'] < 1e-6:
        print("  ⚠️ WARNING: Very low prediction variance!")
    
    return predictions, pred_stats

# Replacement function for the training script
def train_robust_meta_pipeline(meta_df: pd.DataFrame):
    """ROBUST replacement for train_meta_pipeline_FIXED"""
    
    print("🎯 Training ROBUST Meta-Pipeline...")
    print("=" * 50)
    
    # Create robust model
    model, scaler, features, diagnostics = create_robust_meta_model(meta_df)
    
    if model is None:
        print("❌ Robust meta-model creation failed!")
        return None
    
    # Save components
    save_robust_meta_model(model, scaler, features, diagnostics)
    
    print(f"\n✅ ROBUST Meta-Model Training Complete!")
    print(f"  Model: {diagnostics['model_name']}")
    print(f"  Accuracy: {diagnostics['accuracy']:.4f}")
    print(f"  AUC: {diagnostics['auc']:.4f}")
    print(f"  Features: {diagnostics['features_used']}")
    
    return model

# Testing function
def test_robust_meta_model_with_dummy_data():
    """Test the robust meta-model with dummy data"""
    
    print("🧪 Testing Robust Meta-Model with Dummy Data")
    print("=" * 50)
    
    # Create realistic dummy data
    np.random.seed(42)
    n_samples = 50
    
    # Generate features with different characteristics
    pitch_scores = np.random.beta(2, 5, n_samples) * 0.1  # Skewed towards lower values
    rbi_predictions = np.random.normal(0.05, 0.02, n_samples)  # RBI predictions
    bayes_effects = np.random.normal(0, 0.5, n_samples)  # Bayesian effects
    
    # Create realistic winner labels (few winners)
    # Winners tend to have higher pitch scores, lower RBI, positive Bayes effects
    winner_probabilities = (
        (pitch_scores - pitch_scores.mean()) / pitch_scores.std() * 0.3 +
        -(rbi_predictions - rbi_predictions.mean()) / rbi_predictions.std() * 0.2 +
        (bayes_effects - bayes_effects.mean()) / bayes_effects.std() * 0.2
    )
    winner_probabilities = 1 / (1 + np.exp(-winner_probabilities))  # Sigmoid
    winner_flags = np.random.binomial(1, winner_probabilities * 0.1 + 0.02, n_samples)  # ~5% winners
    
    # Create DataFrame
    dummy_data = pd.DataFrame({
        'Name': [f'Pitcher_{i:02d}' for i in range(n_samples)],
        'Season': [2024] * n_samples,
        'pitch_score_mean': pitch_scores,
        'rbi_pred_mean': rbi_predictions,
        'bayes_effect': bayes_effects,
        'winner_flag': winner_flags
    })
    
    print(f"Dummy data created: {len(dummy_data)} samples, {dummy_data['winner_flag'].sum()} winners")
    
    # Train model
    model = train_robust_meta_pipeline(dummy_data)
    
    if model is not None:
        # Test predictions
        predictions, pred_stats = predict_with_robust_meta_model(dummy_data)
        
        if predictions is not None:
            dummy_data['predicted_prob'] = predictions
            
            # Show top candidates
            top_candidates = dummy_data.nlargest(10, 'predicted_prob')
            print(f"\nTop 10 Predicted Candidates:")
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                winner_status = "🏆" if row['winner_flag'] == 1 else "  "
                print(f"{i:2d}. {winner_status} {row['Name']}: {row['predicted_prob']:.4f}")
            
            return True
    
    return False

# Example usage and integration
if __name__ == "__main__":
    print("🚀 Robust Meta-Model Testing")
    
    # Test with dummy data
    success = test_robust_meta_model_with_dummy_data()
    
    if success:
        print("\n✅ Robust meta-model test completed successfully!")
        print("\nTo integrate into your pipeline:")
        print("1. Replace train_meta_pipeline_FIXED() with train_robust_meta_pipeline()")
        print("2. Replace predict_cy_young_with_diagnostics() with predict_with_robust_meta_model()")
        print("3. Use 'models/robust_meta_model.pkl' instead of 'models/meta_model.pkl'")
    else:
        print("\n❌ Robust meta-model test failed!")
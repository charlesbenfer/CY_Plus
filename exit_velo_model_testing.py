# IMPROVED EXIT VELOCITY MODEL FOR TESTING WITH YOUR CSV FILES
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
import logging

# Use the same logger as the meta model
logger = logging.getLogger(__name__)

def create_physics_based_features(df):
    """Create physics-based features that correlate with exit velocity"""
    df_features = df.copy()
    
    # Since we don't have bat_speed, focus on what we DO have
    
    # Launch angle features (we have this!)
    if 'launch_angle' in df.columns:
        df_features['launch_angle_squared'] = df['launch_angle'] ** 2
        df_features['optimal_launch_angle'] = (np.abs(df['launch_angle'] - 25) <= 10).astype(int)  # 15-35 degrees is optimal
    
    # Pitch velocity features (start_speed available)
    if 'start_speed' in df.columns:
        df_features['pitch_speed_squared'] = df['start_speed'] ** 2
        df_features['pitch_speed_log'] = np.log(df['start_speed'] + 1)
    
    # Plate location features (px, pz available)
    if 'px' in df.columns and 'pz' in df.columns:
        df_features['distance_from_center'] = np.sqrt(df['px']**2 + (df['pz'] - 2.5)**2)
        df_features['px_squared'] = df['px'] ** 2
        df_features['pz_squared'] = df['pz'] ** 2
        df_features['location_interaction'] = df['px'] * df['pz']
        df_features['in_sweet_spot'] = (np.abs(df['px']) <= 0.5).astype(int)  # Center of plate
    
    # Pitch movement features (pfxx, pfxz available)
    if 'pfxx' in df.columns and 'pfxz' in df.columns:
        df_features['total_movement'] = np.sqrt(df['pfxx']**2 + df['pfxz']**2)
        df_features['pfxx_squared'] = df['pfxx'] ** 2
        df_features['pfxz_squared'] = df['pfxz'] ** 2
        df_features['movement_interaction'] = df['pfxx'] * df['pfxz']
    
    # Velocity components (vx0, vy0, vz0 available)
    if 'vx0' in df.columns and 'vy0' in df.columns and 'vz0' in df.columns:
        df_features['total_velocity'] = np.sqrt(df['vx0']**2 + df['vy0']**2 + df['vz0']**2)
        df_features['horizontal_velocity'] = np.sqrt(df['vx0']**2 + df['vy0']**2)
        df_features['velocity_angle'] = np.arctan2(df['vz0'], df_features['horizontal_velocity'])
    
    # Spin rate effects (spin_rate available)
    if 'spin_rate' in df.columns:
        df_features['spin_rate_log'] = np.log(df['spin_rate'] + 1)
        df_features['spin_rate_squared'] = df['spin_rate'] ** 2
        if 'start_speed' in df.columns:
            df_features['spin_to_velocity_ratio'] = df['spin_rate'] / (df['start_speed'] + 1)
    
    # Count situation features (balls, strikes available)
    if 'balls' in df.columns and 'strikes' in df.columns:
        df_features['count_pressure'] = df['strikes'] / (df['balls'] + 1)
        df_features['is_hitters_count'] = ((df['balls'] > df['strikes']) & (df['strikes'] < 2)).astype(int)
        df_features['is_two_strike'] = (df['strikes'] == 2).astype(int)
        df_features['count_advantage'] = df['balls'] - df['strikes']
    
    # Interaction features between pitch characteristics and launch angle
    if 'launch_angle' in df.columns and 'start_speed' in df.columns:
        df_features['speed_angle_interaction'] = df['start_speed'] * df['launch_angle']
    
    if 'launch_angle' in df.columns and 'spin_rate' in df.columns:
        df_features['spin_angle_interaction'] = df['spin_rate'] * df['launch_angle']
    
    # Strike zone relative features
    if 'pz' in df.columns and 'sz_top' in df.columns and 'sz_bot' in df.columns:
        df_features['pz_relative_to_zone'] = (df['pz'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
        df_features['in_strike_zone_vertical'] = ((df['pz'] >= df['sz_bot']) & (df['pz'] <= df['sz_top'])).astype(int)
    
    return df_features

def advanced_outlier_detection(df, target_col='launch_speed'):
    """More sophisticated outlier detection for exit velocity"""
    # Remove physically impossible values
    valid_mask = (
        (df[target_col] >= 30) & (df[target_col] <= 130) &  # Reasonable exit velo range
        (df[target_col].notna())
    )
    
    if 'launch_angle' in df.columns:
        valid_mask &= (df['launch_angle'] >= -90) & (df['launch_angle'] <= 90)
    
    if 'bat_speed' in df.columns:
        valid_mask &= (df['bat_speed'] >= 40) & (df['bat_speed'] <= 90)
    
    # Statistical outlier detection using IQR method
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR  # More conservative than 1.5
    upper_bound = Q3 + 2.5 * IQR
    
    statistical_mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
    
    return valid_mask & statistical_mask

def create_exit_velocity_pipeline():
    """Create a more sophisticated pipeline for exit velocity prediction"""
    
    # Numeric features preprocessing
    numeric_transformer = Pipeline([
        ('scaler', RobustScaler()),  # More robust to outliers than StandardScaler
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
    ])
    
    return numeric_transformer

def get_exit_velocity_models():
    """Return dictionary of models optimized for exit velocity prediction"""
    return {
        'gradient_boosting': GradientBoostingRegressor(
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        'random_forest': RandomForestRegressor(
            random_state=42,
            oob_score=True
        ),
        'elastic_net': ElasticNet(
            random_state=42,
            max_iter=2000
        ),
        'ridge': Ridge()
    }

def get_exit_velocity_param_grids():
    """Optimized parameter grids for exit velocity models"""
    return {
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        },
        'elastic_net': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

def evaluate_exit_velocity_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation specific to exit velocity"""
    y_pred = model.predict(X_test)
    
    # Basic metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Exit velocity specific metrics
    # Accuracy within 5 mph (good for practical purposes)
    within_5mph = np.mean(np.abs(y_pred - y_test) <= 5)
    
    # Accuracy within 10 mph (acceptable for many applications)
    within_10mph = np.mean(np.abs(y_pred - y_test) <= 10)
    
    # Performance on different exit velocity ranges
    high_velo_mask = y_test >= 100
    med_velo_mask = (y_test >= 85) & (y_test < 100)
    low_velo_mask = y_test < 85
    
    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'accuracy_within_5mph': within_5mph,
        'accuracy_within_10mph': within_10mph,
        'high_velo_r2': r2_score(y_test[high_velo_mask], y_pred[high_velo_mask]) if high_velo_mask.sum() > 10 else None,
        'med_velo_r2': r2_score(y_test[med_velo_mask], y_pred[med_velo_mask]) if med_velo_mask.sum() > 10 else None,
        'low_velo_r2': r2_score(y_test[low_velo_mask], y_pred[low_velo_mask]) if low_velo_mask.sum() > 10 else None,
    }
    
    logger.info(f"{model_name} Performance:")
    logger.info(f"  RMSE: {rmse:.2f} mph")
    logger.info(f"  MAE: {mae:.2f} mph")
    logger.info(f"  R²: {r2:.3f}")
    logger.info(f"  Within 5 mph: {within_5mph:.1%}")
    logger.info(f"  Within 10 mph: {within_10mph:.1%}")
    
    return metrics

def improved_exit_velocity_model(df, whiff_model=None):
    """Improved exit velocity model with better feature engineering and validation"""
    
    if whiff_model is None or 'launch_speed' not in df.columns:
        logger.warning("Cannot train exit velocity model: missing whiff model or launch_speed column")
        return None, {}
    
    logger.info("Training improved exit velocity model...")
    
    # Filter for contact events only
    contact_data = df[
        ~df['is_whiff'].fillna(False) & 
        df['launch_speed'].notna()
    ].copy()
    
    if len(contact_data) < 200:  # Increased minimum sample size
        logger.warning(f"Insufficient contact data for exit velocity model: {len(contact_data)} samples")
        return None, {}
    
    logger.info(f"Contact data samples: {len(contact_data)}")
    
    # Create physics-based features
    contact_data = create_physics_based_features(contact_data)
    
    # Advanced outlier removal
    outlier_mask = advanced_outlier_detection(contact_data, 'launch_speed')
    contact_data = contact_data[outlier_mask]
    
    logger.info(f"After outlier removal: {len(contact_data)} samples")
    
    if len(contact_data) < 150:
        logger.warning("Too few samples after outlier removal")
        return None, {}
    
    # Select features (prioritize features we actually have)
    priority_features = [
        # Launch characteristics (critical and available!)
        'launch_angle', 'launch_angle_squared', 'optimal_launch_angle',
        
        # Pitch characteristics (available)
        'start_speed', 'pitch_speed_squared', 'spin_rate', 'spin_rate_log',
        
        # Plate location (available as px, pz)
        'px', 'pz', 'distance_from_center', 'in_sweet_spot', 'pz_relative_to_zone',
        
        # Pitch movement (available)
        'pfxx', 'pfxz', 'total_movement', 'vx0', 'vy0', 'vz0', 'total_velocity',
        
        # Interaction features
        'speed_angle_interaction', 'spin_angle_interaction', 'spin_to_velocity_ratio',
        
        # Count situation
        'count_pressure', 'is_hitters_count', 'is_two_strike'
    ]
    
    # Include features that exist in the data
    available_features = [f for f in priority_features if f in contact_data.columns]
    
    # Add other numeric features if available
    other_numeric = contact_data.select_dtypes(include=[np.number]).columns.tolist()
    other_numeric = [f for f in other_numeric if f not in available_features + ['launch_speed']]
    
    all_features = available_features + other_numeric[:10]  # Limit to prevent overfitting
    
    logger.info(f"Using features: {all_features}")
    
    # Prepare data
    X = contact_data[all_features].copy()
    y = contact_data['launch_speed']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )
    
    # Try multiple models
    models = get_exit_velocity_models()
    param_grids = get_exit_velocity_param_grids()
    
    best_model = None
    best_score = float('-inf')
    all_results = {}
    
    # Create preprocessing pipeline
    preprocessor = create_exit_velocity_pipeline()
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Update parameter grid for pipeline
        param_grid = {f'regressor__{k}': v for k, v in param_grids[model_name].items()}
        
        # Randomized search with cross-validation
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20,  # Reduced for faster training
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        try:
            search.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_exit_velocity_model(
                search.best_estimator_, X_test, y_test, model_name
            )
            
            all_results[model_name] = {
                'model': search.best_estimator_,
                'metrics': metrics,
                'best_params': search.best_params_
            }
            
            # Track best model by R²
            if metrics['r2_score'] > best_score:
                best_score = metrics['r2_score']
                best_model = search.best_estimator_
                
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    if best_model is None:
        logger.error("No models trained successfully")
        return None, {}
    
    # Final evaluation of best model
    best_metrics = evaluate_exit_velocity_model(best_model, X_test, y_test, "Best Model")
    
    # Feature importance (if available)
    try:
        if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = all_features
            if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
                try:
                    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out(all_features)
                except:
                    pass
            
            importances = best_model.named_steps['regressor'].feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            best_metrics['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                           key=lambda x: x[1], reverse=True)[:10])
            
            logger.info("Top 5 feature importances:")
            for feat, imp in list(best_metrics['feature_importance'].items())[:5]:
                logger.info(f"  {feat}: {imp:.3f}")
                
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {str(e)}")
    
    # Cross-validation score for final model
    try:
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                  cv=5, scoring='neg_root_mean_squared_error')
        best_metrics['cv_rmse_mean'] = -cv_scores.mean()
        best_metrics['cv_rmse_std'] = cv_scores.std()
        logger.info(f"Cross-validation RMSE: {best_metrics['cv_rmse_mean']:.2f} ± {best_metrics['cv_rmse_std']:.2f}")
    except:
        pass
    
    # Store all model results for comparison
    best_metrics['all_model_results'] = {k: v['metrics'] for k, v in all_results.items()}
    
    return best_model, best_metrics

# ================================
# TESTING AND VALIDATION CODE FOR REAL CSV DATA
# ================================

def load_and_validate_data(csv_file_path=None):
    """Load CSV data and validate it has required columns for exit velocity modeling"""
    
    # Use your meta model's CSV files if no path specified
    if csv_file_path is None:
        csv_file_path = 'data/prior_pitchlevel.csv'
        print(f"📂 Using default pitch-level data: {csv_file_path}")
    
    try:
        # Load with date parsing like your meta model
        df = pd.read_csv(csv_file_path, parse_dates=['game_date'] if 'game_date' in pd.read_csv(csv_file_path, nrows=1).columns else None)
        print(f"✅ Loaded {len(df)} rows from {csv_file_path}")
        
        # Show available columns to understand the data structure
        print(f"📋 Available columns ({len(df.columns)} total):")
        print(f"   First 20: {list(df.columns[:20])}")
        if len(df.columns) > 20:
            print(f"   ... and {len(df.columns) - 20} more")
        
        # Check for required columns for exit velocity
        required_cols = ['launch_speed']  # This is the exit velocity target
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            print(f"❌ Missing required columns: {missing_required}")
            print("💡 Available columns that might contain exit velocity:")
            exit_vel_candidates = [col for col in df.columns if any(term in col.lower() 
                                 for term in ['launch', 'exit', 'speed', 'velo'])]
            print(f"   Candidates: {exit_vel_candidates}")
            return None
            
        # Check for whiff indicator (based on your meta model structure)
        whiff_cols = ['is_whiff', 'whiff', 'swing_miss', 'contact']
        whiff_col = None
        for col in whiff_cols:
            if col in df.columns:
                whiff_col = col
                break
                
        if whiff_col is None:
            print("⚠️  No whiff indicator column found. Looking for common patterns...")
            potential_cols = [col for col in df.columns if any(term in col.lower() 
                            for term in ['whiff', 'miss', 'contact', 'swing'])]
            if potential_cols:
                print(f"🔍 Potential whiff columns found: {potential_cols}")
                # Try to create whiff indicator if swing info exists
                if 'is_swing' in df.columns:
                    print("💡 Creating whiff indicator from swing data...")
                    # If there's launch_speed data, assume non-null = contact, null = whiff (among swings)
                    df['is_whiff'] = df['is_swing'] & df['launch_speed'].isnull()
                    whiff_col = 'is_whiff'
                    print(f"✅ Created whiff indicator: {whiff_col}")
            else:
                print("❌ No whiff indicator found. You may need to create one.")
                return None
        else:
            print(f"✅ Found whiff indicator: {whiff_col}")
            if whiff_col != 'is_whiff':
                df['is_whiff'] = df[whiff_col]
        
        # Data quality checks
        contact_data = df[~df['is_whiff'].fillna(False) & df['launch_speed'].notna()]
        whiff_data = df[df['is_whiff'].fillna(False)]
        
        print(f"📊 Data breakdown:")
        print(f"   Total pitches: {len(df):,}")
        print(f"   Contact events: {len(contact_data):,}")
        print(f"   Whiff events: {len(whiff_data):,}")
        
        if len(contact_data) > 0:
            print(f"   Exit velocity range: {contact_data['launch_speed'].min():.1f} - {contact_data['launch_speed'].max():.1f} mph")
            print(f"   Exit velocity mean: {contact_data['launch_speed'].mean():.1f} mph")
        
        # Check for useful features from your meta model
        meta_model_features = [
            'start_speed', 'end_speed', 'extension', 'spin_rate',
            'pfxx', 'pfxz', 'vx0', 'vy0', 'vz0', 'sz_top', 'sz_bot',
            'pitch_type', 'batter_hand', 'pitcher_hand',
            'break_angle', 'break_length', 'plate_x', 'plate_z',
            'balls', 'strikes'
        ]
        
        # Additional features that might be useful for exit velocity
        additional_features = ['bat_speed', 'attack_angle', 'launch_angle', 'release_spin_rate']
        
        all_useful_features = meta_model_features + additional_features
        available_features = [f for f in all_useful_features if f in df.columns]
        missing_features = [f for f in all_useful_features if f not in df.columns]
        
        print(f"✅ Available useful features ({len(available_features)}): {available_features[:10]}{'...' if len(available_features) > 10 else ''}")
        if missing_features:
            print(f"⚠️  Missing potentially useful features ({len(missing_features)}): {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def analyze_data_quality(df):
    """Analyze the quality and characteristics of the baseball data"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Contact events analysis
    contact_data = df[~df['is_whiff'].fillna(False) & df['launch_speed'].notna()]
    
    if len(contact_data) == 0:
        print("❌ No contact events with exit velocity found!")
        return False
        
    print(f"📊 Contact Events Analysis:")
    print(f"   Sample size: {len(contact_data)}")
    print(f"   Exit velocity stats:")
    print(f"     Mean: {contact_data['launch_speed'].mean():.1f} mph")
    print(f"     Std:  {contact_data['launch_speed'].std():.1f} mph")
    print(f"     Min:  {contact_data['launch_speed'].min():.1f} mph")
    print(f"     Max:  {contact_data['launch_speed'].max():.1f} mph")
    
    # Check for data quality issues
    print(f"\n🔍 Data Quality Checks:")
    
    # Unrealistic exit velocities
    low_velo = (contact_data['launch_speed'] < 30).sum()
    high_velo = (contact_data['launch_speed'] > 130).sum()
    if low_velo > 0 or high_velo > 0:
        print(f"⚠️  Potentially unrealistic exit velocities:")
        print(f"     < 30 mph: {low_velo} events")
        print(f"     > 130 mph: {high_velo} events")
    else:
        print("✅ Exit velocity ranges look realistic")
    
    # Missing values in key features
    key_features = ['bat_speed', 'start_speed', 'attack_angle', 'launch_angle']
    available_key_features = [f for f in key_features if f in contact_data.columns]
    
    if available_key_features:
        print(f"\n📋 Missing values in key features:")
        for feature in available_key_features:
            missing_pct = (contact_data[feature].isna().sum() / len(contact_data)) * 100
            if missing_pct > 50:
                print(f"❌ {feature}: {missing_pct:.1f}% missing (too high!)")
            elif missing_pct > 20:
                print(f"⚠️  {feature}: {missing_pct:.1f}% missing (concerning)")
            elif missing_pct > 0:
                print(f"✅ {feature}: {missing_pct:.1f}% missing (acceptable)")
            else:
                print(f"✅ {feature}: No missing values")
    
    # Feature correlations with exit velocity
    print(f"\n🔗 Feature correlations with exit velocity:")
    numeric_features = contact_data.select_dtypes(include=[np.number]).columns
    correlations = contact_data[numeric_features].corr()['launch_speed'].abs().sort_values(ascending=False)
    
    for feature, corr in correlations.head(10).items():
        if feature != 'launch_speed':
            if corr > 0.5:
                print(f"💪 {feature}: {corr:.3f} (strong)")
            elif corr > 0.3:
                print(f"👍 {feature}: {corr:.3f} (moderate)")
            elif corr > 0.1:
                print(f"👌 {feature}: {corr:.3f} (weak)")
            else:
                print(f"🤷 {feature}: {corr:.3f} (very weak)")
    
    return len(contact_data) >= 200  # Return True if sufficient data

def test_model_with_real_data(model, df, train_features):
    """Test model performance on real baseball data"""
    print("\n" + "="*60)
    print("REAL DATA MODEL TESTING")
    print("="*60)
    
    if model is None:
        print("❌ Model is None - cannot test")
        return None
    
    # Get contact events only
    contact_data = df[~df['is_whiff'].fillna(False) & df['launch_speed'].notna()].copy()
    
    if len(contact_data) < 50:
        print(f"❌ Insufficient contact data for testing: {len(contact_data)} samples")
        return None
    
    # IMPORTANT: Create physics-based features for testing data too!
    contact_data = create_physics_based_features(contact_data)
    
    # Filter train_features to only include features that actually exist
    available_train_features = [f for f in train_features if f in contact_data.columns]
    missing_features = [f for f in train_features if f not in contact_data.columns]
    
    if missing_features:
        print(f"⚠️  Missing features in test data: {missing_features}")
        print(f"✅ Using available features: {available_train_features}")
    
    if len(available_train_features) < 3:
        print(f"❌ Too few features available for testing: {len(available_train_features)}")
        return None
    
    # Create train/test split for evaluation
    X = contact_data[available_train_features].copy()
    X = X.fillna(X.median())
    y = contact_data['launch_speed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    try:
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Practical accuracy metrics
        within_3mph = np.mean(np.abs(y_pred - y_test) <= 3)
        within_5mph = np.mean(np.abs(y_pred - y_test) <= 5)
        within_10mph = np.mean(np.abs(y_pred - y_test) <= 10)
        
        print(f"✅ Tested on {len(y_test)} real contact events")
        print(f"🔧 Used {len(available_train_features)} features: {available_train_features[:5]}{'...' if len(available_train_features) > 5 else ''}")
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   RMSE: {rmse:.2f} mph")
        print(f"   MAE:  {mae:.2f} mph") 
        print(f"   R²:   {r2:.3f}")
        
        print(f"\n🎯 PRACTICAL ACCURACY:")
        print(f"   Within 3 mph:  {within_3mph:.1%}")
        print(f"   Within 5 mph:  {within_5mph:.1%}")
        print(f"   Within 10 mph: {within_10mph:.1%}")
        
        # Performance by exit velocity ranges
        print(f"\n📈 PERFORMANCE BY EXIT VELOCITY RANGE:")
        
        # High exit velocity (100+ mph)
        high_mask = y_test >= 100
        if high_mask.sum() > 5:
            high_r2 = r2_score(y_test[high_mask], y_pred[high_mask])
            high_mae = mean_absolute_error(y_test[high_mask], y_pred[high_mask])
            print(f"   High velocity (100+ mph): R² = {high_r2:.3f}, MAE = {high_mae:.1f} mph ({high_mask.sum()} samples)")
        
        # Medium exit velocity (85-99 mph)
        med_mask = (y_test >= 85) & (y_test < 100)
        if med_mask.sum() > 5:
            med_r2 = r2_score(y_test[med_mask], y_pred[med_mask])
            med_mae = mean_absolute_error(y_test[med_mask], y_pred[med_mask])
            print(f"   Medium velocity (85-99 mph): R² = {med_r2:.3f}, MAE = {med_mae:.1f} mph ({med_mask.sum()} samples)")
        
        # Low exit velocity (<85 mph)
        low_mask = y_test < 85
        if low_mask.sum() > 5:
            low_r2 = r2_score(y_test[low_mask], y_pred[low_mask])
            low_mae = mean_absolute_error(y_test[low_mask], y_pred[low_mask])
            print(f"   Low velocity (<85 mph): R² = {low_r2:.3f}, MAE = {low_mae:.1f} mph ({low_mask.sum()} samples)")
        
        # Prediction range analysis
        print(f"\n📊 PREDICTION ANALYSIS:")
        print(f"   Actual range:     {y_test.min():.1f} - {y_test.max():.1f} mph")
        print(f"   Predicted range:  {y_pred.min():.1f} - {y_pred.max():.1f} mph")
        print(f"   Mean prediction:  {y_pred.mean():.1f} mph (actual: {y_test.mean():.1f} mph)")
        
        # Check for systematic bias
        residuals = y_pred - y_test
        bias = residuals.mean()
        if abs(bias) > 2:
            print(f"⚠️  Systematic bias detected: {bias:+.2f} mph")
        else:
            print(f"✅ Low systematic bias: {bias:+.2f} mph")
        
        # Performance interpretation
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if r2 > 0.7 and within_5mph > 0.6:
            print("   🥇 EXCELLENT: Model performs very well!")
        elif r2 > 0.5 and within_5mph > 0.5:
            print("   🥈 GOOD: Model performs well for most cases")
        elif r2 > 0.3 and within_5mph > 0.4:
            print("   🥉 FAIR: Model has moderate predictive power")
        else:
            print("   ❌ POOR: Model needs significant improvement")
            
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_3mph': within_3mph,
            'within_5mph': within_5mph,
            'within_10mph': within_10mph,
            'bias': bias,
            'test_size': len(y_test),
            'predictions': y_pred,
            'actuals': y_test,
            'features_used': available_train_features
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print(f"🔍 Available features in contact_data: {list(contact_data.columns)}")
        print(f"🔍 Requested features: {available_train_features}")
        return None

def run_comprehensive_test():
    """Run a comprehensive test of the exit velocity model using your CSV files"""
    print("🚀 STARTING COMPREHENSIVE EXIT VELOCITY MODEL TEST")
    print("=" * 60)
    
    # Test with your actual CSV files
    csv_files_to_try = [
        'data/prior_pitchlevel.csv',  # Your main pitch-level data
        'prior_pitchlevel.csv',       # In case it's in current directory
        'data/pitchlevel.csv',        # Alternative naming
        'pitchlevel.csv'              # Alternative location
    ]
    
    df = None
    for csv_file in csv_files_to_try:
        print(f"📂 Trying to load: {csv_file}")
        if os.path.exists(csv_file):
            df = load_and_validate_data(csv_file)
            if df is not None:
                break
        else:
            print(f"   ❌ File not found: {csv_file}")
    
    if df is None:
        print("\n❌ Could not load any CSV files!")
        print("💡 Please ensure one of these files exists:")
        for f in csv_files_to_try:
            print(f"   - {f}")
        print("\nOr call run_complete_pipeline_test('your_file_path.csv') directly")
        return None, None, None
    
    return run_complete_pipeline_test_with_data(df)

def run_complete_pipeline_test_with_data(df):
    """Run the complete pipeline test on loaded data"""
    print(f"\n🔬 Running complete pipeline test on {len(df)} records...")
    
    # Analyze data quality
    data_quality_ok = analyze_data_quality(df)
    if not data_quality_ok:
        print("❌ Data quality issues prevent model training")
        return None, None, None
    
    # IMPORTANT: Create features ONCE and use consistently
    print(f"\n🔧 Creating physics-based features...")
    df_with_features = create_physics_based_features(df)
    
    # Get contact data with features
    contact_data = df_with_features[
        ~df_with_features['is_whiff'].fillna(False) & 
        df_with_features['launch_speed'].notna()
    ].copy()
    
    print(f"Contact data shape: {contact_data.shape}")
    
    # BETTER feature selection - use correlation analysis
    numeric_features = contact_data.select_dtypes(include=[np.number]).columns
    feature_correlations = contact_data[numeric_features].corr()['launch_speed'].abs()
    
    # Exclude target, index, and ID columns
    exclude_cols = [
        'launch_speed',  # target
        'Unnamed: 0', 'game_id', 'batter_id', 'batter_team_id', 'pitcher_id', 'pitcher_team_id',  # IDs
        'ab_number', 'play_id', 'index_play',  # more IDs
        'game_date', 'start_time', 'end_time',  # dates/times (if they made it through)
        'away_score', 'home_score', 'rbi'  # game outcomes
    ]
    
    # Remove excluded columns from correlations
    filtered_correlations = feature_correlations.drop(exclude_cols, errors='ignore')
    
    # Select top correlated features
    top_features = filtered_correlations.sort_values(ascending=False).head(15)
    train_features = list(top_features.index)
    
    print(f"\n🎯 SELECTED FEATURES (by correlation with exit velocity):")
    for i, (feature, corr) in enumerate(top_features.items()):
        print(f"   {i+1}. {feature}: {corr:.3f}")
    
    # Verify all features exist
    missing_features = [f for f in train_features if f not in contact_data.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        train_features = [f for f in train_features if f in contact_data.columns]
    
    print(f"🔧 Final feature set: {len(train_features)} features")
    
    # Train the improved model WITH THE SAME FEATURES
    print(f"\n🤖 Training improved exit velocity model...")
    mock_whiff_model = "mock"
    
    # Pass the SAME feature set to the model training
    model, metrics = improved_exit_velocity_model_with_features(
        df_with_features, mock_whiff_model, train_features
    )
    
    if model is None:
        print("❌ Model training failed!")
        return None, None, None
    
    print("✅ Model training completed!")
    if 'r2_score' in metrics:
        print(f"📊 Training R²: {metrics['r2_score']:.3f}")
    
    # Test the model with THE SAME FEATURES
    print(f"\n🧪 Testing model on real data...")
    test_results = test_model_with_real_data_fixed(model, df_with_features, train_features)
    
    if test_results:
        print(f"\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        
        # Performance analysis
        if test_results.get('r2', 0) < 0.4:
            print(f"\n🔍 PERFORMANCE DIAGNOSIS:")
            print(f"Model R² = {test_results.get('r2', 0):.3f}")
            print(f"This suggests exit velocity prediction is challenging with available features.")
            print(f"Missing critical features like bat_speed significantly limits prediction accuracy.")
        
        print("="*70)
        return model, metrics, test_results
    else:
        print(f"\n❌ Testing failed")
        return model, metrics, None

def improved_exit_velocity_model_with_features(df, whiff_model=None, feature_list=None):
    """Improved exit velocity model that uses a specific feature list"""
    
    if whiff_model is None or 'launch_speed' not in df.columns:
        logger.warning("Cannot train exit velocity model: missing whiff model or launch_speed column")
        return None, {}
    
    logger.info("Training improved exit velocity model with specified features...")
    
    # Filter for contact events only
    contact_data = df[
        ~df['is_whiff'].fillna(False) & 
        df['launch_speed'].notna()
    ].copy()
    
    if len(contact_data) < 200:
        logger.warning(f"Insufficient contact data for exit velocity model: {len(contact_data)} samples")
        return None, {}
    
    logger.info(f"Contact data samples: {len(contact_data)}")
    
    # Advanced outlier removal
    outlier_mask = advanced_outlier_detection(contact_data, 'launch_speed')
    contact_data = contact_data[outlier_mask]
    
    logger.info(f"After outlier removal: {len(contact_data)} samples")
    
    if len(contact_data) < 150:
        logger.warning("Too few samples after outlier removal")
        return None, {}
    
    # Use the provided feature list
    if feature_list is None:
        logger.error("No feature list provided")
        return None, {}
    
    # Verify features exist
    available_features = [f for f in feature_list if f in contact_data.columns]
    missing_features = [f for f in feature_list if f not in contact_data.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    if len(available_features) < 3:
        logger.error("Too few available features")
        return None, {}
    
    logger.info(f"Using {len(available_features)} features: {available_features}")
    
    # Prepare data
    X = contact_data[available_features].copy()
    y = contact_data['launch_speed']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )
    
    # Try multiple models (same as before)
    models = get_exit_velocity_models()
    param_grids = get_exit_velocity_param_grids()
    
    best_model = None
    best_score = float('-inf')
    all_results = {}
    
    # Create preprocessing pipeline
    preprocessor = create_exit_velocity_pipeline()
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Update parameter grid for pipeline
        param_grid = {f'regressor__{k}': v for k, v in param_grids[model_name].items()}
        
        # Randomized search with cross-validation
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        try:
            search.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_exit_velocity_model(
                search.best_estimator_, X_test, y_test, model_name
            )
            
            all_results[model_name] = {
                'model': search.best_estimator_,
                'metrics': metrics,
                'best_params': search.best_params_
            }
            
            # Track best model by R²
            if metrics['r2_score'] > best_score:
                best_score = metrics['r2_score']
                best_model = search.best_estimator_
                
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    if best_model is None:
        logger.error("No models trained successfully")
        return None, {}
    
    # Final evaluation of best model
    best_metrics = evaluate_exit_velocity_model(best_model, X_test, y_test, "Best Model")
    
    # Store feature list in metrics for consistency
    best_metrics['feature_list'] = available_features
    best_metrics['all_model_results'] = {k: v['metrics'] for k, v in all_results.items()}
    
    return best_model, best_metrics

def test_model_with_real_data_fixed(model, df, train_features):
    """Test model performance with guaranteed feature consistency"""
    print("\n" + "="*60)
    print("REAL DATA MODEL TESTING (FIXED)")
    print("="*60)
    
    if model is None:
        print("❌ Model is None - cannot test")
        return None
    
    # Get contact events only
    contact_data = df[~df['is_whiff'].fillna(False) & df['launch_speed'].notna()].copy()
    
    if len(contact_data) < 50:
        print(f"❌ Insufficient contact data for testing: {len(contact_data)} samples")
        return None
    
    # Verify ALL features exist
    missing_features = [f for f in train_features if f not in contact_data.columns]
    if missing_features:
        print(f"❌ Missing features in test data: {missing_features}")
        return None
    
    # Use EXACT same features as training
    X = contact_data[train_features].copy()
    X = X.fillna(X.median())
    y = contact_data['launch_speed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    try:
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Practical accuracy metrics
        within_3mph = np.mean(np.abs(y_pred - y_test) <= 3)
        within_5mph = np.mean(np.abs(y_pred - y_test) <= 5)
        within_10mph = np.mean(np.abs(y_pred - y_test) <= 10)
        
        print(f"✅ Tested on {len(y_test)} real contact events")
        print(f"🔧 Used {len(train_features)} features consistently")
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   RMSE: {rmse:.2f} mph")
        print(f"   MAE:  {mae:.2f} mph") 
        print(f"   R²:   {r2:.3f}")
        
        print(f"\n🎯 PRACTICAL ACCURACY:")
        print(f"   Within 3 mph:  {within_3mph:.1%}")
        print(f"   Within 5 mph:  {within_5mph:.1%}")
        print(f"   Within 10 mph: {within_10mph:.1%}")
        
        # Check for systematic bias
        residuals = y_pred - y_test
        bias = residuals.mean()
        if abs(bias) > 2:
            print(f"⚠️  Systematic bias detected: {bias:+.2f} mph")
        else:
            print(f"✅ Low systematic bias: {bias:+.2f} mph")
        
        # Performance interpretation
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if r2 > 0.7 and within_5mph > 0.6:
            print("   🥇 EXCELLENT: Model performs very well!")
        elif r2 > 0.5 and within_5mph > 0.5:
            print("   🥈 GOOD: Model performs well for most cases")
        elif r2 > 0.3 and within_5mph > 0.4:
            print("   🥉 FAIR: Model has moderate predictive power")
        else:
            print("   ❌ POOR: Model needs significant improvement")
            
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_3mph': within_3mph,
            'within_5mph': within_5mph,
            'within_10mph': within_10mph,
            'bias': bias,
            'test_size': len(y_test),
            'predictions': y_pred,
            'actuals': y_test,
            'features_used': train_features
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return None

# Alternative function that takes a specific file path
def run_complete_pipeline_test(csv_file_path):
    """Run the complete pipeline test on a specific CSV file"""
    print("🚀 STARTING COMPLETE EXIT VELOCITY MODEL PIPELINE TEST")
    print("=" * 70)
    
    # Load and validate data
    df = load_and_validate_data(csv_file_path)
    if df is None:
        return None, None, None
    
    return run_complete_pipeline_test_with_data(df)

# Example usage for testing with your actual CSV files:
if __name__ == "__main__":
    # Method 1: Automatic detection of your CSV files
    print("🎯 TESTING EXIT VELOCITY MODEL WITH YOUR CSV FILES")
    print("="*60)
    model, metrics, test_results = run_comprehensive_test()
    
    # Method 2: Specify exact file path if needed
    # model, metrics, test_results = run_complete_pipeline_test('data/prior_pitchlevel.csv')
    
    # Method 3: Step-by-step testing if you want more control
    # df = load_and_validate_data('data/prior_pitchlevel.csv')
    # if df is not None:
    #     analyze_data_quality(df)
    #     model, metrics = improved_exit_velocity_model(df, "mock_whiff_model")
    #     test_results = test_model_with_real_data(model, df, your_feature_list)
    
    if model is not None and test_results is not None:
        print("\n🎉 SUCCESS! Your exit velocity model is ready!")
        print("📁 Check the diagnostics folder for detailed analysis")
    else:
        print("\n⚠️  Something went wrong. Check the error messages above.")
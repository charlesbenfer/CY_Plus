import os
import shutil
import numpy as np
import pandas as pd
import datetime
import unicodedata
import statsapi
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration - Must be at the top before other TensorFlow imports
import tensorflow as tf

# Configure GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Configuration: Using {len(physical_devices)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
        print("Falling back to CPU")
else:
    print("🔧 No GPU found, using CPU")

# Set mixed precision for better GPU performance
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled for better GPU performance")
except Exception as e:
    print(f"⚠️ Mixed precision setup failed: {e}")

from data_scraper import MLB_Scrape
from pybaseball import pitching_stats, playerid_lookup

# Level 1 imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)

# Level 2 imports - Now with GPU optimization
import keras_tuner as kt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler as Scaler

# Level 3 imports
import pymc as pm

# Meta imports
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

# Create directories for models and diagnostics
os.makedirs('models', exist_ok=True)
os.makedirs('diagnostics', exist_ok=True)
os.makedirs('diagnostics/level1', exist_ok=True)
os.makedirs('diagnostics/level2', exist_ok=True)
os.makedirs('diagnostics/level3', exist_ok=True)
os.makedirs('diagnostics/meta', exist_ok=True)
os.makedirs('diagnostics/data_quality', exist_ok=True)

def diagnose_data_quality(df, level_name, target_col=None):
    """Comprehensive data quality diagnostics"""
    print(f"\n🔍 Data Quality Diagnostics - {level_name}")
    print("=" * 60)
    
    # Basic stats
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_stats,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(f"\nTop 10 columns with missing values:")
    print(missing_df.head(10))
    
    # Save missing values plot
    plt.figure(figsize=(12, 8))
    top_missing = missing_df.head(20)
    plt.barh(range(len(top_missing)), top_missing['Missing_Percentage'])
    plt.yticks(range(len(top_missing)), top_missing.index, fontsize=8)
    plt.xlabel('Missing Percentage')
    plt.title(f'Missing Values Analysis - {level_name}')
    plt.tight_layout()
    plt.savefig(f'diagnostics/data_quality/{level_name}_missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Data types analysis
    dtype_counts = df.dtypes.value_counts()
    print(f"\nData types distribution:")
    print(dtype_counts)
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumerical columns summary:")
        print(df[numeric_cols].describe())
        
        # Check for infinite values
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"\nInfinite values found:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count}")
    
    # Target variable analysis if provided
    if target_col and target_col in df.columns:
        print(f"\nTarget variable '{target_col}' analysis:")
        target_series = df[target_col].dropna()
        
        if target_series.dtype in ['object', 'bool'] or target_series.nunique() < 10:
            # Categorical target
            print(f"Value counts:")
            print(target_series.value_counts())
            
            plt.figure(figsize=(10, 6))
            target_series.value_counts().plot(kind='bar')
            plt.title(f'Target Distribution - {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'diagnostics/data_quality/{level_name}_target_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Numerical target
            print(f"Statistics: mean={target_series.mean():.4f}, std={target_series.std():.4f}")
            print(f"Range: [{target_series.min():.4f}, {target_series.max():.4f}]")
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.hist(target_series, bins=50, alpha=0.7)
            plt.title(f'{target_col} Distribution')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            from scipy import stats
            stats.probplot(target_series, dist="norm", plot=plt)
            plt.title(f'{target_col} Q-Q Plot')
            plt.tight_layout()
            plt.savefig(f'diagnostics/data_quality/{level_name}_target_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Correlation analysis for numeric features
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Feature Correlation Matrix - {level_name}')
        plt.tight_layout()
        plt.savefig(f'diagnostics/data_quality/{level_name}_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return missing_df, dtype_counts


import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                           roc_curve, mean_squared_error, mean_absolute_error, r2_score,
                           precision_recall_curve, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import joblib
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_diagnostics_dirs():
    """Create necessary directories for diagnostics"""
    dirs = ['diagnostics/level1', 'models/level1']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def validate_data_requirements(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data meets minimum requirements for training"""
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check minimum sample size
    if len(df) < 1000:
        validation_results['errors'].append(f"Insufficient data: {len(df)} samples (minimum 1000)")
        validation_results['valid'] = False
    
    # Check required columns
    required_cols = ['is_swing']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        validation_results['errors'].append(f"Missing required columns: {missing_required}")
        validation_results['valid'] = False
    
    # Check class balance for swing
    if 'is_swing' in df.columns:
        swing_rate = df['is_swing'].fillna(False).mean()
        validation_results['stats']['swing_rate'] = swing_rate
        if swing_rate < 0.05 or swing_rate > 0.95:
            validation_results['warnings'].append(f"Extreme class imbalance: {swing_rate:.3f} swing rate")
    
    return validation_results

def get_feature_config() -> Dict[str, list]:
    """Define feature configuration with priorities"""
    return {
        'critical_numeric': ['start_speed', 'pfxx', 'pfxz'],  # Most important features
        'important_numeric': ['end_speed', 'spin_rate', 'extension', 'vx0', 'vy0', 'vz0'],
        'contextual_numeric': ['sz_top', 'sz_bot'],
        'categorical': ['pitch_type', 'batter_hand', 'pitcher_hand'],
        'optional_numeric': ['break_angle', 'break_length', 'plate_x', 'plate_z']
    }

def analyze_feature_availability(df: pd.DataFrame, feature_config: Dict[str, list]) -> Dict[str, Any]:
    """Comprehensive feature availability analysis"""
    analysis = {
        'available_features': {},
        'missing_features': {},
        'feature_quality': {},
        'recommendations': []
    }
    
    all_features = []
    for category, features in feature_config.items():
        all_features.extend(features)
        available = []
        missing = []
        
        for feature in features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df) * 100
                available.append(feature)
                analysis['feature_quality'][feature] = {
                    'missing_pct': missing_pct,
                    'dtype': str(df[feature].dtype),
                    'unique_values': df[feature].nunique() if missing_pct < 100 else 0
                }
                
                # Quality warnings
                if missing_pct > 50:
                    analysis['recommendations'].append(f"High missing rate for {feature}: {missing_pct:.1f}%")
                elif missing_pct > 20:
                    analysis['recommendations'].append(f"Moderate missing rate for {feature}: {missing_pct:.1f}%")
            else:
                missing.append(feature)
        
        analysis['available_features'][category] = available
        analysis['missing_features'][category] = missing
    
    # Check critical features
    critical_missing = analysis['missing_features']['critical_numeric']
    if critical_missing:
        analysis['recommendations'].append(f"CRITICAL: Missing essential features: {critical_missing}")
    
    return analysis

def create_robust_pipeline(numeric_features: list, categorical_features: list, 
                          model_type: str = 'classifier') -> Pipeline:
    """Create a robust preprocessing and modeling pipeline"""
    
    # Enhanced preprocessing
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Model selection based on type
    if model_type == 'classifier':
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
    else:
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def get_optimized_param_grids(model_type: str = 'classifier') -> Dict[str, Any]:
    """Get optimized hyperparameter grids based on model type"""
    
    base_params = {
        'model__n_estimators': [200, 300, 500],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [5, 10, 15],
        'model__min_samples_leaf': [2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None]
    }
    
    if model_type == 'classifier':
        # Add classifier-specific parameters
        base_params.update({
            'model__criterion': ['gini', 'entropy']
        })
    else:
        # Add regressor-specific parameters
        base_params.update({
            'model__criterion': ['squared_error', 'absolute_error']
        })
    
    return base_params

def comprehensive_model_evaluation(model, X_test, y_test, model_name: str, 
                                 task_type: str = 'classification') -> Dict[str, Any]:
    """Comprehensive model evaluation with multiple metrics"""
    
    results = {'model_name': model_name, 'task_type': task_type}
    
    try:
        if task_type == 'classification':
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            })
            
            # Additional metrics for imbalanced datasets
            from sklearn.metrics import precision_score, recall_score, f1_score
            results.update({
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            })
            
        else:  # regression
            y_pred = model.predict(X_test)
            
            results.update({
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            })
            
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        results['error'] = str(e)
    
    return results

def create_enhanced_visualizations(model, X_test, y_test, model_name: str, 
                                 task_type: str = 'classification', save_dir: str = 'diagnostics/level1'):
    """Create comprehensive visualizations for model diagnostics"""
    
    if task_type == 'classification':
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title(f'{model_name} - Confusion Matrix')
        axes[0,0].set_ylabel('True')
        axes[0,0].set_xlabel('Predicted')
        
        # ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title(f'{model_name} - ROC Curve')
            axes[0,1].legend()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            axes[1,0].plot(recall, precision)
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title(f'{model_name} - Precision-Recall Curve')
            
            # Prediction Distribution
            axes[1,1].hist(y_prob[y_test == 0], alpha=0.5, label='Negative Class', bins=30)
            axes[1,1].hist(y_prob[y_test == 1], alpha=0.5, label='Positive Class', bins=30)
            axes[1,1].set_xlabel('Predicted Probability')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title(f'{model_name} - Prediction Distribution')
            axes[1,1].legend()
        
    else:  # regression
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0,0].scatter(y_test, y_pred, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual')
        axes[0,0].set_ylabel('Predicted')
        axes[0,0].set_title(f'{model_name} - Actual vs Predicted')
        
        # Residuals Plot
        axes[0,1].scatter(y_pred, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title(f'{model_name} - Residuals Plot')
        
        # Residuals Distribution
        axes[1,0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'{model_name} - Residuals Distribution')
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title(f'{model_name} - Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_diagnostics.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def save_model_artifacts(model, model_name: str, feature_names: list, 
                        performance_metrics: Dict[str, Any], save_dir: str = 'models/level1'):
    """Save model and related artifacts"""
    
    # Save the trained model
    model_path = f'{save_dir}/{model_name.lower().replace(" ", "_")}_model.joblib'
    joblib.dump(model, model_path)
    
    # Save feature information
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_type': type(model).__name__
    }
    
    pd.DataFrame([feature_info]).to_csv(
        f'{save_dir}/{model_name.lower().replace(" ", "_")}_features.csv', 
        index=False
    )
    
    # Save performance metrics
    pd.DataFrame([performance_metrics]).to_csv(
        f'{save_dir}/{model_name.lower().replace(" ", "_")}_metrics.csv', 
        index=False
    )
    
    logger.info(f"Model artifacts saved for {model_name}")

def train_pitch_level_models(df: pd.DataFrame) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Enhanced pitch-level model training with comprehensive diagnostics and error handling
    """
    logger.info("🏗️ Training Level 1 pitch-level models...")
    
    # Create necessary directories
    create_diagnostics_dirs()
    
    # Validate data requirements
    validation = validate_data_requirements(df)
    if not validation['valid']:
        for error in validation['errors']:
            logger.error(error)
        return None, None, None
    
    for warning in validation['warnings']:
        logger.warning(warning)
    
    # Feature configuration and analysis
    feature_config = get_feature_config()
    feature_analysis = analyze_feature_availability(df, feature_config)
    
    # Combine available features
    available_numeric = []
    available_categorical = []
    
    for category, features in feature_analysis['available_features'].items():
        if 'numeric' in category:
            available_numeric.extend(features)
        elif category == 'categorical':
            available_categorical.extend(features)
    
    logger.info(f"Using {len(available_numeric)} numeric and {len(available_categorical)} categorical features")
    
    if len(available_numeric) < 3:  # Minimum feature requirement
        logger.error("Insufficient features for training")
        return None, None, None
    
    # Prepare data
    df = df.copy()
    df['is_swing_bin'] = df['is_swing'].fillna(False).astype(int)
    
    all_features = available_numeric + available_categorical
    X = df[all_features].copy()
    y = df['is_swing_bin']
    
    # Check class balance and sample size
    class_counts = y.value_counts()
    minority_class_size = min(class_counts)
    
    if minority_class_size < 50:
        logger.error(f"Insufficient minority class samples: {minority_class_size}")
        return None, None, None
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SWING MODEL
    logger.info("Training swing classifier...")
    swing_pipeline = create_robust_pipeline(available_numeric, available_categorical, 'classifier')
    swing_params = get_optimized_param_grids('classifier')
    
    swing_search = RandomizedSearchCV(
        swing_pipeline,
        swing_params,
        n_iter=20,  # Increased iterations
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    swing_search.fit(X_train, y_train)
    swing_model = swing_search.best_estimator_
    
    # Evaluate swing model
    swing_metrics = comprehensive_model_evaluation(
        swing_model, X_test, y_test, "Swing Model", "classification"
    )
    
    # Create visualizations
    create_enhanced_visualizations(
        swing_model, X_test, y_test, "Swing Model", "classification"
    )
    
    # Save swing model artifacts
    save_model_artifacts(swing_model, "Swing Model", all_features, swing_metrics)
    
    # WHIFF MODEL (only on swings)
    whiff_model = None
    whiff_metrics = {}
    
    swings_data = df[df['is_swing_bin'] == 1].copy()
    if len(swings_data) > 100 and 'is_whiff' in swings_data.columns:
        logger.info("Training whiff classifier...")
        
        swings_data['is_whiff_bin'] = swings_data['is_whiff'].fillna(False).astype(int)
        whiff_counts = swings_data['is_whiff_bin'].value_counts()
        
        if min(whiff_counts) >= 30:  # Minimum samples per class
            X_whiff = swings_data[all_features].copy()
            y_whiff = swings_data['is_whiff_bin']
            
            X_whiff_train, X_whiff_test, y_whiff_train, y_whiff_test = train_test_split(
                X_whiff, y_whiff, test_size=0.2, random_state=42, stratify=y_whiff
            )
            
            whiff_pipeline = create_robust_pipeline(available_numeric, available_categorical, 'classifier')
            whiff_search = RandomizedSearchCV(
                whiff_pipeline,
                swing_params,  # Same params as swing model
                n_iter=15,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            whiff_search.fit(X_whiff_train, y_whiff_train)
            whiff_model = whiff_search.best_estimator_
            
            whiff_metrics = comprehensive_model_evaluation(
                whiff_model, X_whiff_test, y_whiff_test, "Whiff Model", "classification"
            )
            
            create_enhanced_visualizations(
                whiff_model, X_whiff_test, y_whiff_test, "Whiff Model", "classification"
            )
            
            save_model_artifacts(whiff_model, "Whiff Model", all_features, whiff_metrics)
        else:
            logger.warning("Insufficient data for whiff model training")
    
    # EXIT VELOCITY MODEL (contact events only)
    exit_velocity_model = None
    exit_velocity_metrics = {}
    
    if whiff_model is not None and 'launch_speed' in df.columns:
        logger.info("Training exit velocity regressor...")
        
        contact_data = swings_data[
            ~swings_data['is_whiff'].fillna(False) & 
            swings_data['launch_speed'].notna()
        ].copy()
        
        if len(contact_data) > 100:
            X_exit = contact_data[all_features].copy()
            y_exit = contact_data['launch_speed']
            
            X_exit_train, X_exit_test, y_exit_train, y_exit_test = train_test_split(
                X_exit, y_exit, test_size=0.2, random_state=42
            )
            
            exit_pipeline = create_robust_pipeline(available_numeric, available_categorical, 'regressor')
            exit_params = get_optimized_param_grids('regressor')
            
            exit_search = RandomizedSearchCV(
                exit_pipeline,
                exit_params,
                n_iter=15,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            exit_search.fit(X_exit_train, y_exit_train)
            exit_velocity_model = exit_search.best_estimator_
            
            exit_velocity_metrics = comprehensive_model_evaluation(
                exit_velocity_model, X_exit_test, y_exit_test, "Exit Velocity Model", "regression"
            )
            
            create_enhanced_visualizations(
                exit_velocity_model, X_exit_test, y_exit_test, "Exit Velocity Model", "regression"
            )
            
            save_model_artifacts(exit_velocity_model, "Exit Velocity Model", all_features, exit_velocity_metrics)
        else:
            logger.warning("Insufficient contact data for exit velocity model")
    
    # Create comprehensive summary
    summary = {
        'timestamp': pd.Timestamp.now(),
        'total_samples': len(df),
        'swing_rate': df['is_swing_bin'].mean(),
        'features_used': len(all_features),
        'feature_categories': {k: len(v) for k, v in feature_analysis['available_features'].items()},
        **{f'swing_{k}': v for k, v in swing_metrics.items() if isinstance(v, (int, float))},
        **{f'whiff_{k}': v for k, v in whiff_metrics.items() if isinstance(v, (int, float))},
        **{f'exit_velocity_{k}': v for k, v in exit_velocity_metrics.items() if isinstance(v, (int, float))}
    }
    
    pd.DataFrame([summary]).to_csv('diagnostics/level1/comprehensive_summary.csv', index=False)
    
    logger.info("✅ Model training completed successfully")
    return swing_model, whiff_model, exit_velocity_model


def compute_pitch_score(df, swing_pipe, whiff_pipe, exit_pipe):
    """Compute pitch scores using trained models with diagnostics"""
    print("\n🔍 Computing pitch scores with diagnostics...")
    
    df2 = df.copy()
    df2['is_swing_bin'] = df2['is_swing'].fillna(False).infer_objects(copy=False).astype(int)

    num_cols = [
        'start_speed','end_speed','extension','spin_rate',
        'pfxx','pfxz','vx0','vy0','vz0','sz_top','sz_bot'
    ]
    cat_cols = ['pitch_type','batter_hand','pitcher_hand']
    feats = num_cols + cat_cols
    
    # Check feature availability
    available_feats = [f for f in feats if f in df2.columns]
    print(f"Using {len(available_feats)}/{len(feats)} features for scoring")

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
    
    # Diagnostics
    print(f"Pitch score statistics:")
    print(f"  Mean: {pitch_scores.mean():.4f}")
    print(f"  Std: {pitch_scores.std():.4f}")
    print(f"  Range: [{pitch_scores.min():.4f}, {pitch_scores.max():.4f}]")
    print(f"  Zeros: {(pitch_scores == 0).sum():,} ({(pitch_scores == 0).mean()*100:.2f}%)")
    
    # Save distribution plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(p_s, bins=50, alpha=0.7)
    plt.title('Swing Probabilities')
    plt.xlabel('P(Swing)')
    
    plt.subplot(1, 3, 2)
    plt.hist(p_w[m], bins=50, alpha=0.7)
    plt.title('Whiff Probabilities (Swings only)')
    plt.xlabel('P(Whiff)')
    
    plt.subplot(1, 3, 3)
    plt.hist(pitch_scores[pitch_scores > 0], bins=50, alpha=0.7)
    plt.title('Pitch Scores (Non-zero)')
    plt.xlabel('Pitch Score')
    
    plt.tight_layout()
    plt.savefig('diagnostics/level1/pitch_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    return pd.Series(pitch_scores, index=df2.index)


def prepare_game_sequences(df, feats, target, seq_len=5):
    """Prepare sequential data for LSTM training with diagnostics"""
    print(f"\n🔍 Preparing sequences for LSTM training...")
    
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
            'nan_features': nan_features
        })
    
    # Save sequence statistics
    seq_df = pd.DataFrame(sequence_stats)
    seq_df.to_csv('diagnostics/level2/sequence_preparation_stats.csv', index=False)
    
    print(f"Sequence preparation results:")
    print(f"  Total players: {len(sequence_stats)}")
    print(f"  Total sequences: {len(X)}")
    print(f"  Avg sequences per player: {seq_df['valid_sequences'].mean():.2f}")
    print(f"  Players with 0 sequences: {(seq_df['valid_sequences'] == 0).sum()}")
    
    # Plot sequence distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(seq_df['total_games'], bins=20, alpha=0.7)
    plt.title('Games per Player')
    plt.xlabel('Number of Games')
    
    plt.subplot(1, 3, 2)
    plt.hist(seq_df['valid_sequences'], bins=20, alpha=0.7)
    plt.title('Valid Sequences per Player')
    plt.xlabel('Number of Sequences')
    
    plt.subplot(1, 3, 3)
    plt.scatter(seq_df['total_games'], seq_df['valid_sequences'], alpha=0.6)
    plt.xlabel('Total Games')
    plt.ylabel('Valid Sequences')
    plt.title('Games vs Valid Sequences')
    
    plt.tight_layout()
    plt.savefig('diagnostics/level2/sequence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return np.array(X), np.array(y), idxs


def tune_game_level_lstm(Xtr, ytr, Xv, yv, seq_len, n_feat):
    """GPU-optimized LSTM tuning with enhanced diagnostics"""
    print("🚀 Training GPU-optimized LSTM model with diagnostics...")
    
    # Input data diagnostics
    print(f"\nLSTM Input Diagnostics:")
    print(f"  Training shape: {Xtr.shape}")
    print(f"  Validation shape: {Xv.shape}")
    print(f"  Target range: [{ytr.min():.4f}, {ytr.max():.4f}]")
    print(f"  Target mean±std: {ytr.mean():.4f}±{ytr.std():.4f}")
    print(f"  NaN check - Training: {np.isnan(Xtr).sum()}, Validation: {np.isnan(Xv).sum()}")
    
    # Target distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(ytr, bins=30, alpha=0.7, label='Training')
    plt.hist(yv, bins=30, alpha=0.7, label='Validation')
    plt.title('Target Distribution')
    plt.xlabel('Target Value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(ytr)), ytr, alpha=0.5, s=1, label='Training')
    plt.scatter(range(len(yv)), yv, alpha=0.5, s=1, label='Validation')
    plt.title('Target Values Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('diagnostics/level2/lstm_input_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Clean up previous tuning directory
    shutil.rmtree('lstm_tuning', ignore_errors=True)

    def build(hp):
        model = models.Sequential([
            layers.Masking(mask_value=0., input_shape=(seq_len, n_feat)),
            layers.LSTM(
                units=hp.Choice('units', [32, 64, 128, 256]), 
                dropout=hp.Float('dropout', 0.0, 0.5, step=0.1),
                recurrent_dropout=hp.Float('recurrent_dropout', 0.0, 0.3, step=0.1),
                return_sequences=hp.Boolean('return_sequences')
            ),
            layers.Dropout(hp.Float('final_dropout', 0.0, 0.5, step=0.1)) if hp.get('return_sequences') else layers.Lambda(lambda x: x),
            layers.LSTM(
                units=hp.Choice('units2', [16, 32, 64]), 
                dropout=hp.Float('dropout2', 0.0, 0.3, step=0.1)
            ) if hp.get('return_sequences') else layers.Lambda(lambda x: x),
            layers.Dense(
                units=hp.Choice('dense_units', [16, 32, 64]),
                activation=hp.Choice('dense_activation', ['relu', 'tanh'])
            ),
            layers.Dropout(hp.Float('dense_dropout', 0.0, 0.3, step=0.1)),
            layers.Dense(1, dtype='float32')  # Ensure float32 output for mixed precision
        ])
        
        # Enhanced optimizer configuration
        optimizer = Adam(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='LOG'),
            beta_1=hp.Float('beta_1', 0.85, 0.95, step=0.05),
            beta_2=hp.Float('beta_2', 0.99, 0.999, step=0.005)
        )
        
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=['mae']
        )
        return model

    # Enhanced tuner configuration
    tuner = kt.RandomSearch(
        build, 
        objective='val_loss', 
        max_trials=20,
        executions_per_trial=2,
        directory='lstm_tuning', 
        project_name='cy_model',
        overwrite=True, 
        max_consecutive_failed_trials=8
    )

    # Enhanced callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    try:
        print("   Starting hyperparameter search...")
        tuner.search(
            Xtr, ytr, 
            epochs=50,
            batch_size=32,
            validation_data=(Xv, yv), 
            callbacks=callbacks,
            verbose=1
        )
        print("   Hyperparameter search completed!")
    except RuntimeError as e:
        print(f"⚠️ Tuner stopped early with error: {e}")

    try:
        best = tuner.get_best_models(num_models=1)[0]
        val_loss = best.evaluate(Xv, yv, verbose=0)
        print(f"✅ Best LSTM Validation Loss: {val_loss}")
        print(f"   Best Hyperparameters: {tuner.get_best_hyperparameters()[0].values}")
        
        # Final training with best model
        history = best.fit(
            Xtr, ytr,
            epochs=20,
            batch_size=32,
            validation_data=(Xv, yv),
            callbacks=callbacks,
            verbose=1
        )
        
        # Training history diagnostics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('LSTM Training History - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('diagnostics/level2/lstm_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Prediction diagnostics
        ytr_pred = best.predict(Xtr, verbose=0).flatten()
        yv_pred = best.predict(Xv, verbose=0).flatten()
        
        # Performance metrics
        train_mse = mean_squared_error(ytr, ytr_pred)
        train_mae = mean_absolute_error(ytr, ytr_pred)
        train_r2 = r2_score(ytr, ytr_pred)
        
        val_mse = mean_squared_error(yv, yv_pred)
        val_mae = mean_absolute_error(yv, yv_pred)
        val_r2 = r2_score(yv, yv_pred)
        
        print(f"\nLSTM Performance Metrics:")
        print(f"  Training - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        # Prediction vs actual plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(ytr, ytr_pred, alpha=0.5, s=1)
        plt.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Training Set (R² = {train_r2:.4f})')
        
        plt.subplot(1, 3, 2)
        plt.scatter(yv, yv_pred, alpha=0.5, s=1)
        plt.plot([yv.min(), yv.max()], [yv.min(), yv.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Validation Set (R² = {val_r2:.4f})')
        
        plt.subplot(1, 3, 3)
        residuals_val = yv - yv_pred
        plt.scatter(yv_pred, residuals_val, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Validation Residuals')
        
        plt.tight_layout()
        plt.savefig('diagnostics/level2/lstm_prediction_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save performance summary
        performance_summary = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'best_params': str(tuner.get_best_hyperparameters()[0].values)
        }
        pd.DataFrame([performance_summary]).to_csv('diagnostics/level2/lstm_performance.csv', index=False)
        
        best.save('models/level2_lstm.h5', save_format='tf')
        return best
        
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"⚠️ Fallback to default LSTM due to: {e}")
        
        # Fallback model with good default architecture
        fallback = models.Sequential([
            layers.Masking(mask_value=0., input_shape=(seq_len, n_feat)),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, dtype='float32')
        ])
        
        fallback.compile(
            optimizer=Adam(learning_rate=1e-3), 
            loss='mse',
            metrics=['mae']
        )
        
        history = fallback.fit(
            Xtr, ytr, 
            epochs=30, 
            batch_size=32,
            validation_data=(Xv, yv), 
            callbacks=callbacks,
            verbose=1
        )
        
        # Save fallback performance
        val_loss_final = fallback.evaluate(Xv, yv, verbose=0)
        fallback_summary = {
            'model_type': 'fallback',
            'final_val_loss': val_loss_final[0],
            'final_val_mae': val_loss_final[1]
        }
        pd.DataFrame([fallback_summary]).to_csv('diagnostics/level2/fallback_performance.csv', index=False)
        
        fallback.save('models/level2_lstm.h5', save_format='tf')
        return fallback


def fit_bayesian_season_model(df):
    """Enhanced Bayesian hierarchical model with comprehensive diagnostics"""
    print("🧠 Fitting Bayesian hierarchical model with diagnostics...")
    
    # Input diagnostics
    diagnose_data_quality(df, "Level3_Bayesian", "metric")
    
    pitchers = df['pitcher'].unique()
    idx = pd.Categorical(df['pitcher'], categories=pitchers).codes
    
    print(f"Bayesian model setup:")
    print(f"  Number of pitchers: {len(pitchers)}")
    print(f"  Number of observations: {len(df)}")
    print(f"  Observations per pitcher: {len(df)/len(pitchers):.2f}")
    
    # Prior diagnostics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(df['metric'], bins=30, alpha=0.7)
    plt.title('Metric Distribution')
    plt.xlabel('Metric Value')
    
    plt.subplot(1, 3, 2)
    obs_per_pitcher = df.groupby('pitcher').size()
    plt.hist(obs_per_pitcher, bins=20, alpha=0.7)
    plt.title('Observations per Pitcher')
    plt.xlabel('Number of Observations')
    
    plt.subplot(1, 3, 3)
    pitcher_means = df.groupby('pitcher')['metric'].mean()
    plt.hist(pitcher_means, bins=20, alpha=0.7)
    plt.title('Pitcher-level Means')
    plt.xlabel('Mean Metric')
    
    plt.tight_layout()
    plt.savefig('diagnostics/level3/bayesian_input_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    with pm.Model() as m:
        # Enhanced priors
        mu = pm.Normal('mu', 0, 2)
        sd = pm.HalfNormal('sd', 2)
        a = pm.Normal('a', mu=mu, sigma=sd, shape=len(pitchers))
        eps = pm.HalfCauchy('eps', 2)
        
        pm.Normal('obs', mu=a[idx], sigma=eps, observed=df['metric'])
        
        # Enhanced sampling with better convergence
        idata = pm.sample(
            2000, 
            tune=2000, 
            cores=min(4, os.cpu_count()), 
            target_accept=0.95,
            max_treedepth=12,
            random_seed=42
        )
    
    # Comprehensive convergence diagnostics
    try:
        rhat = pm.rhat(idata)
        if hasattr(rhat, 'max'):
            max_rhat = float(rhat.max().values)
        else:
            max_rhat = float(rhat.to_array().max().values)
        
        ess_bulk = pm.ess(idata, kind="bulk")
        ess_tail = pm.ess(idata, kind="tail")
        
        min_ess_bulk = float(ess_bulk.min().values) if hasattr(ess_bulk, 'min') else float(ess_bulk.to_array().min().values)
        min_ess_tail = float(ess_tail.min().values) if hasattr(ess_tail, 'min') else float(ess_tail.to_array().min().values)
        
        print(f"Convergence Diagnostics:")
        print(f"  Max R-hat: {max_rhat:.4f} (should be < 1.1)")
        print(f"  Min ESS bulk: {min_ess_bulk:.0f} (should be > 400)")
        print(f"  Min ESS tail: {min_ess_tail:.0f} (should be > 400)")
        
        convergence_ok = max_rhat < 1.1 and min_ess_bulk > 400 and min_ess_tail > 400
        print(f"  Overall convergence: {'✅ GOOD' if convergence_ok else '⚠️ POOR'}")
        
        # Save convergence diagnostics
        convergence_stats = {
            'max_rhat': max_rhat,
            'min_ess_bulk': min_ess_bulk,
            'min_ess_tail': min_ess_tail,
            'convergence_ok': convergence_ok
        }
        pd.DataFrame([convergence_stats]).to_csv('diagnostics/level3/convergence_diagnostics.csv', index=False)
        
    except Exception as e:
        print(f"Convergence diagnostics unavailable: {e}")
        convergence_ok = False
    
    # Extract posterior means and diagnostics
    a_mean = idata.posterior['a'].mean(dim=('chain', 'draw')).values
    a_std = idata.posterior['a'].std(dim=('chain', 'draw')).values
    
    mu_mean = float(idata.posterior['mu'].mean().values)
    sd_mean = float(idata.posterior['sd'].mean().values)
    eps_mean = float(idata.posterior['eps'].mean().values)
    
    print(f"Posterior estimates:")
    print(f"  Global mean (μ): {mu_mean:.4f}")
    print(f"  Between-pitcher SD (σ): {sd_mean:.4f}")
    print(f"  Within-pitcher SD (ε): {eps_mean:.4f}")
    print(f"  Pitcher effects range: [{a_mean.min():.4f}, {a_mean.max():.4f}]")
    
    # Posterior diagnostics plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Manual trace plots (PyMC plot_trace doesn't support ax parameter in newer versions)
    try:
        # Extract trace data
        mu_trace = idata.posterior['mu'].values.flatten()
        sd_trace = idata.posterior['sd'].values.flatten()
        eps_trace = idata.posterior['eps'].values.flatten()
        
        # Plot traces manually
        axes[0, 0].plot(mu_trace)
        axes[0, 0].set_title('Trace: Global Mean (μ)')
        axes[0, 0].set_ylabel('μ')
        
        axes[0, 1].plot(sd_trace)
        axes[0, 1].set_title('Trace: Between-pitcher SD (σ)')
        axes[0, 1].set_ylabel('σ')
        
        axes[0, 2].plot(eps_trace)
        axes[0, 2].set_title('Trace: Within-pitcher SD (ε)')
        axes[0, 2].set_ylabel('ε')
        
    except Exception as e:
        print(f"   Warning: Could not create trace plots: {e}")
        # Clear the top row if trace plots fail
        for ax in axes[0]:
            ax.text(0.5, 0.5, 'Trace plot\nunavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trace Plot')
    
    # Posterior distributions
    axes[1, 0].hist(idata.posterior['mu'].values.flatten(), bins=30, alpha=0.7)
    axes[1, 0].set_title('Posterior: Global Mean (μ)')
    axes[1, 0].axvline(mu_mean, color='red', linestyle='--', label=f'Mean: {mu_mean:.3f}')
    axes[1, 0].legend()
    
    axes[1, 1].hist(idata.posterior['sd'].values.flatten(), bins=30, alpha=0.7)
    axes[1, 1].set_title('Posterior: Between-pitcher SD (σ)')
    axes[1, 1].axvline(sd_mean, color='red', linestyle='--', label=f'Mean: {sd_mean:.3f}')
    axes[1, 1].legend()
    
    axes[1, 2].hist(a_mean, bins=30, alpha=0.7)
    axes[1, 2].set_title('Pitcher Effects Distribution')
    axes[1, 2].axvline(a_mean.mean(), color='red', linestyle='--', label=f'Mean: {a_mean.mean():.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('diagnostics/level3/bayesian_posterior_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Pitcher effects analysis
    pitcher_effects_df = pd.DataFrame({
        'pitcher': pitchers,
        'posterior_mean': a_mean,
        'posterior_std': a_std,
        'n_observations': [sum(idx == i) for i in range(len(pitchers))],
        'empirical_mean': [df[df['pitcher'] == p]['metric'].mean() for p in pitchers]
    })
    
    # Shrinkage analysis
    pitcher_effects_df['shrinkage'] = abs(pitcher_effects_df['empirical_mean'] - pitcher_effects_df['posterior_mean'])
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(pitcher_effects_df['empirical_mean'], pitcher_effects_df['posterior_mean'], alpha=0.6)
    plt.plot([pitcher_effects_df['empirical_mean'].min(), pitcher_effects_df['empirical_mean'].max()],
             [pitcher_effects_df['empirical_mean'].min(), pitcher_effects_df['empirical_mean'].max()], 'r--')
    plt.xlabel('Empirical Mean')
    plt.ylabel('Posterior Mean')
    plt.title('Shrinkage Effect')
    
    plt.subplot(2, 2, 2)
    plt.scatter(pitcher_effects_df['n_observations'], pitcher_effects_df['shrinkage'], alpha=0.6)
    plt.xlabel('Number of Observations')
    plt.ylabel('Shrinkage Amount')
    plt.title('Shrinkage vs Sample Size')
    
    plt.subplot(2, 2, 3)
    plt.scatter(pitcher_effects_df['posterior_std'], pitcher_effects_df['shrinkage'], alpha=0.6)
    plt.xlabel('Posterior Standard Deviation')
    plt.ylabel('Shrinkage Amount')
    plt.title('Shrinkage vs Uncertainty')
    
    plt.subplot(2, 2, 4)
    top_pitchers = pitcher_effects_df.nlargest(10, 'posterior_mean')
    plt.barh(range(len(top_pitchers)), top_pitchers['posterior_mean'])
    plt.yticks(range(len(top_pitchers)), top_pitchers['pitcher'], fontsize=8)
    plt.xlabel('Posterior Mean Effect')
    plt.title('Top 10 Pitcher Effects')
    
    plt.tight_layout()
    plt.savefig('diagnostics/level3/pitcher_effects_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    pitcher_effects_df.to_csv('diagnostics/level3/pitcher_effects.csv', index=False)
    
    joblib.dump((pitchers, a_mean), 'models/level3_bayes.pkl')
    print(f"✅ Bayesian model fitted for {len(pitchers)} pitchers")
    
    return pitchers, a_mean


def train_meta_pipeline_FIXED(meta_df: pd.DataFrame):
    """FIXED meta-model training with proper feature scaling and engineering"""
    print("🎯 Training FIXED meta-model with proper scaling...")
    
    # Check input data quality
    print(f"Input data shape: {meta_df.shape}")
    print(f"Winner distribution: {meta_df['winner_flag'].value_counts()}")
    
    # Base features - ensure they exist
    base_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']
    
    for feat in base_features:
        if feat not in meta_df.columns:
            print(f"❌ Missing feature: {feat}")
            meta_df[feat] = 0.0
        else:
            print(f"✅ Feature {feat}: range=[{meta_df[feat].min():.6f}, {meta_df[feat].max():.6f}], std={meta_df[feat].std():.6f}")
    
    # Fill NaN values
    meta_df[base_features] = meta_df[base_features].fillna(0)
    
    # CRITICAL: Create and save feature scaler
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(meta_df[base_features])
    
    # Save the scaler for prediction time!
    joblib.dump(feature_scaler, 'models/meta_feature_scaler.pkl')
    print("✅ Feature scaler saved to models/meta_feature_scaler.pkl")
    
    # Create scaled dataframe
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=base_features,
        index=meta_df.index
    )
    
    print(f"Scaled features statistics:")
    print(scaled_df.describe())
    
    # Create interaction features AFTER scaling
    scaled_df['pitch_rbi_interaction'] = scaled_df['pitch_score_mean'] * scaled_df['rbi_pred_mean']
    scaled_df['pitch_bayes_interaction'] = scaled_df['pitch_score_mean'] * scaled_df['bayes_effect']
    scaled_df['rbi_bayes_interaction'] = scaled_df['rbi_pred_mean'] * scaled_df['bayes_effect']
    
    feature_columns = base_features + ['pitch_rbi_interaction', 'pitch_bayes_interaction', 'rbi_bayes_interaction']
    
    # Check for zero variance (which kills models)
    for col in feature_columns:
        var = scaled_df[col].var()
        if var < 1e-10:
            print(f"⚠️ WARNING: {col} has zero variance! Removing from model.")
            feature_columns.remove(col)
    
    print(f"Final features: {feature_columns}")
    
    X = scaled_df[feature_columns].values
    y = meta_df['winner_flag'].values
    
    # Much less aggressive regularization for small datasets
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('clf', LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear',
            random_state=42,
            max_iter=1000
        ))
    ])
    
    # MUCH wider C range - small datasets need less regularization
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        'clf__penalty': ['l1', 'l2']
    }
    
    # Appropriate CV for dataset size
    n_samples = len(meta_df)
    if n_samples <= 30:
        cv_strategy = StratifiedKFold(n_splits=min(3, n_samples//2), shuffle=True, random_state=42)
    else:
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Using {cv_strategy.n_splits}-fold CV for {n_samples} samples")
    
    gs = GridSearchCV(
        pipe, 
        param_grid, 
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    gs.fit(X, y)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    y_pred = gs.predict(X)
    y_prob = gs.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    print(f"\n✅ FIXED Meta-Model Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best params: {gs.best_params_}")
    print(f"  CV score: {gs.best_score_:.4f}")
    
    if auc > 0.7:
        print("  🎉 EXCELLENT: AUC > 0.7!")
    elif auc > 0.6:
        print("  ✅ GOOD: AUC > 0.6")
    elif auc > 0.55:
        print("  🔶 OKAY: AUC > 0.55 (better than random)")
    else:
        print("  ❌ POOR: AUC ≤ 0.55 (may need more data/features)")
    
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Feature importance
    if hasattr(gs.best_estimator_.named_steps['clf'], 'coef_'):
        coefs = gs.best_estimator_.named_steps['clf'].coef_[0]
        print(f"\nFeature Coefficients:")
        for feat, coef in zip(feature_columns, coefs):
            print(f"  {feat}: {coef:.4f}")
    
    # Save everything
    joblib.dump(gs.best_estimator_, 'models/meta_model.pkl')
    joblib.dump(feature_columns, 'models/meta_features.pkl')
    
    print("✅ Model and features saved")
    
    return gs.best_estimator_

def comprehensive_pipeline_diagnostics(pitch_df, game_df, season_df, labels_df):
    """Run comprehensive diagnostics across all pipeline levels"""
    print("\n🔍 Running Comprehensive Pipeline Diagnostics")
    print("=" * 70)
    
    # Data consistency checks
    print("\n📊 Data Consistency Analysis:")
    
    # Check data alignment
    pitch_players = set(pitch_df['Name'].unique())
    game_players = set(game_df['Name'].unique())
    season_players = set(season_df['Name'].unique())
    label_players = set(labels_df['Name'].unique())
    
    print(f"  Players in pitch data: {len(pitch_players)}")
    print(f"  Players in game data: {len(game_players)}")
    print(f"  Players in season data: {len(season_players)}")
    print(f"  Players in labels: {len(label_players)}")
    
    common_players = pitch_players & game_players & season_players & label_players
    print(f"  Players common to all datasets: {len(common_players)}")
    
    if len(common_players) < len(label_players) * 0.8:
        print("  ⚠️ WARNING: Low overlap between datasets!")
    
    # Temporal consistency
    pitch_date_range = (pitch_df['game_date'].min(), pitch_df['game_date'].max())
    game_date_range = (game_df['game_date'].min(), game_df['game_date'].max())
    
    print(f"  Pitch data date range: {pitch_date_range[0]} to {pitch_date_range[1]}")
    print(f"  Game data date range: {game_date_range[0]} to {game_date_range[1]}")
    
    # Season consistency
    pitch_seasons = set(pitch_df['Season'].unique())
    game_seasons = set(game_df['Season'].unique())
    season_seasons = set(season_df['Season'].unique())
    label_seasons = set(labels_df['Season'].unique())
    
    print(f"  Seasons in datasets: Pitch {pitch_seasons}, Game {game_seasons}, Season {season_seasons}, Labels {label_seasons}")
    
    # Data volume analysis
    print(f"\n📈 Data Volume Analysis:")
    print(f"  Pitches per game: {len(pitch_df) / len(game_df):.2f}")
    print(f"  Games per player-season: {len(game_df) / len(season_df):.2f}")
    print(f"  Pitches per player-season: {len(pitch_df) / len(season_df):.2f}")
    
    # Missing data cascade analysis
    print(f"\n🔍 Missing Data Cascade Analysis:")
    
    # Critical features for each level
    level1_features = ['start_speed', 'end_speed', 'spin_rate', 'pitch_type', 'is_swing', 'is_whiff', 'launch_speed']
    level2_features = ['pitch_score_mean', 'rbi_mean']
    level3_features = ['rbi_pred']
    meta_features = ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect', 'winner_flag']
    
    def check_feature_completeness(df, features, level_name):
        print(f"  {level_name}:")
        complete_rows = len(df)
        for feature in features:
            if feature in df.columns:
                missing = df[feature].isnull().sum()
                complete_rows = min(complete_rows, len(df) - missing)
                print(f"    {feature}: {missing}/{len(df)} missing ({missing/len(df)*100:.1f}%)")
            else:
                print(f"    {feature}: MISSING COLUMN")
                complete_rows = 0
        print(f"    Complete rows for {level_name}: {complete_rows}/{len(df)} ({complete_rows/len(df)*100:.1f}%)")
        return complete_rows
    
    pitch_complete = check_feature_completeness(pitch_df, level1_features, "Level 1 (Pitch)")
    # For game and season, we'd need to create the aggregated data first
    
    # Save comprehensive diagnostics summary
    diagnostics_summary = {
        'total_pitchers_pitch': len(pitch_players),
        'total_pitchers_game': len(game_players),
        'total_pitchers_season': len(season_players),
        'total_pitchers_labels': len(label_players),
        'common_pitchers': len(common_players),
        'pitch_data_completeness': pitch_complete / len(pitch_df),
        'total_pitches': len(pitch_df),
        'total_games': len(game_df),
        'total_seasons': len(season_df),
        'pitches_per_game': len(pitch_df) / len(game_df),
        'games_per_season': len(game_df) / len(season_df)
    }
    
    pd.DataFrame([diagnostics_summary]).to_csv('diagnostics/pipeline_overview.csv', index=False)
    
    # Create data flow visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Player overlap Venn-like visualization
    ax1.bar(['Pitch', 'Game', 'Season', 'Labels'], 
            [len(pitch_players), len(game_players), len(season_players), len(label_players)])
    ax1.axhline(y=len(common_players), color='red', linestyle='--', label=f'Common: {len(common_players)}')
    ax1.set_title('Players per Dataset')
    ax1.set_ylabel('Number of Players')
    ax1.legend()
    
    # Data volume pyramid
    levels = ['Pitches', 'Games', 'Seasons']
    volumes = [len(pitch_df), len(game_df), len(season_df)]
    ax2.barh(levels, volumes)
    ax2.set_title('Data Volume Pyramid')
    ax2.set_xlabel('Number of Records')
    
    # Missing data heatmap for key features
    missing_data = []
    for feature in level1_features:
        if feature in pitch_df.columns:
            missing_data.append(pitch_df[feature].isnull().sum() / len(pitch_df))
        else:
            missing_data.append(1.0)  # 100% missing if column doesn't exist
    
    ax3.barh(level1_features, missing_data)
    ax3.set_title('Missing Data by Feature (Level 1)')
    ax3.set_xlabel('Fraction Missing')
    ax3.set_xlim(0, 1)
    
    # Timeline of data
    if 'game_date' in pitch_df.columns:
        pitch_monthly = pitch_df.groupby(pitch_df['game_date'].dt.to_period('M')).size()
        ax4.plot(pitch_monthly.index.astype(str), pitch_monthly.values, marker='o')
        ax4.set_title('Pitch Data Timeline')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Pitches')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('diagnostics/pipeline_overview.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("🚀 Starting MLB Cy Young Prediction Training Pipeline with Comprehensive Diagnostics")
    print("=" * 80)
    
    # Load and diagnose training data
    print("📚 Loading and diagnosing training data...")
    try:
        prior_pitch_df = pd.read_csv('data/prior_pitchlevel.csv', parse_dates=['game_date'])
        prior_game_df = pd.read_csv('data/prior_gamelevel.csv', parse_dates=['game_date'])
        prior_season_tot = pd.read_csv('data/prior_seasonlevel.csv')
        prior_labels_df = pd.read_csv('data/winner_labels.csv')
        
        print(f"   ✅ Loaded {len(prior_pitch_df):,} pitch records")
        print(f"   ✅ Loaded {len(prior_game_df):,} game records")
        print(f"   ✅ Loaded {len(prior_season_tot):,} season records")
        print(f"   ✅ Loaded {len(prior_labels_df):,} label records")
        
    except FileNotFoundError as e:
        print(f"   ❌ Error loading data: {e}")
        print("   Please ensure data files exist in the 'data/' directory")
        exit(1)
    
    # Run comprehensive pipeline diagnostics
    comprehensive_pipeline_diagnostics(prior_pitch_df, prior_game_df, prior_season_tot, prior_labels_df)
    
    # Level 1: Train pitch-level models
    print("\n" + "="*50)
    print("LEVEL 1: PITCH-LEVEL MODELS")
    print("="*50)
    
    swing_pipe, whiff_pipe, exit_pipe = train_pitch_level_models(prior_pitch_df)
    
    if swing_pipe is None:
        print("❌ Level 1 training failed. Cannot continue.")
        exit(1)
    
    joblib.dump((swing_pipe, whiff_pipe, exit_pipe), 'models/level1_pipelines.pkl')
    print("✅ Level 1 models saved")

    # Apply Level 1 to compute pitch scores
    print("\n🔄 Computing pitch scores for training data...")
    prior_pitch_df['pitch_score'] = compute_pitch_score(
        prior_pitch_df, swing_pipe, whiff_pipe, exit_pipe
    )
    
    # Aggregate to game level
    agg = prior_pitch_df.groupby(
        ['game_id','game_date','Name','Season']
    )['pitch_score'].mean().reset_index().rename(
        columns={'pitch_score':'pitch_score_mean'}
    )
    game_df = prior_game_df.merge(agg, on=['game_id','game_date','Name','Season'], how='left')
    
    # Level 2: Train LSTM model
    print("\n" + "="*50)
    print("LEVEL 2: GAME-LEVEL LSTM MODEL")
    print("="*50)
    
    seq_len = 5
    game_feats = [
        c for c in game_df.columns
        if c not in ['game_id','game_date','Name','Season','rbi_mean']
    ]
    
    print(f"Available game-level features: {len(game_feats)}")
    print(f"Features: {game_feats[:10]}..." if len(game_feats) > 10 else f"Features: {game_feats}")
    
    if 'rbi_mean' not in game_df.columns:
        print("❌ 'rbi_mean' target not found in game data. Creating placeholder...")
        game_df['rbi_mean'] = np.random.normal(0.1, 0.05, len(game_df))  # Placeholder for testing
    
    X, y, idxs = prepare_game_sequences(game_df, game_feats, 'rbi_mean', seq_len)
    
    if len(X) == 0:
        print("❌ No valid sequences generated. Check your data.")
        exit(1)
    
    print(f"Generated {len(X):,} sequences from {len(game_df):,} games")
    
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale for LSTM
    print("⚖️ Scaling features for LSTM...")
    ns, sl, nf = Xtr.shape
    scaler = Scaler()
    Xtr_flat = Xtr.reshape(-1, nf)
    Xv_flat = Xv.reshape(-1, nf)
    scaler.fit(Xtr_flat)
    
    Xtr_scaled = scaler.transform(Xtr_flat).reshape(ns, sl, nf)
    Xv_scaled = scaler.transform(Xv_flat).reshape(Xv.shape)
    
    joblib.dump(scaler, 'models/level2_scaler.pkl')
    joblib.dump(game_feats, 'models/level2_game_feats.pkl')

    # Train LSTM
    lstm = tune_game_level_lstm(Xtr_scaled, ytr, Xv_scaled, yv, seq_len, nf)
    
    if lstm is None:
        print("❌ Level 2 training failed. Cannot continue.")
        exit(1)
    
    print("✅ Level 2 LSTM model saved")

    # Generate predictions for training data
    print("🔮 Generating LSTM predictions for training data...")
    game_df['rbi_pred'] = np.nan
    if len(idxs) > 0:
        all_preds = lstm.predict(np.vstack([Xtr_scaled, Xv_scaled]), verbose=0).flatten()
        game_df.loc[idxs, 'rbi_pred'] = all_preds

    # Level 3: Bayesian hierarchical model
    print("\n" + "="*50)
    print("LEVEL 3: BAYESIAN HIERARCHICAL MODEL")
    print("="*50)
    
    bayes_df = game_df[['Name','Season','rbi_pred']].rename(
        columns={'Name':'pitcher','rbi_pred':'metric'}
    ).dropna()
    
    if len(bayes_df) == 0:
        print("❌ No valid RBI predictions for Bayesian model.")
        exit(1)
    
    pitchers, a_mean = fit_bayesian_season_model(bayes_df)

    prior_season_tot['bayes_effect'] = (
        prior_season_tot['Name']
        .map(dict(zip(pitchers, a_mean)))
        .fillna(a_mean.mean())
    )
    
    print("✅ Level 3 Bayesian model saved")

        # Meta-model training
    print("\n" + "="*50)
    print("META-LEARNING MODEL")
    print("="*50)
    
    p_seas = (
        prior_pitch_df
        .groupby(['Name','Season'])['pitch_score']    # ← This uses the correct pitch_score column
        .mean().reset_index()
        .rename(columns={'pitch_score':'pitch_score_mean'})
    )
    r_seas = (
        game_df
        .groupby(['Name','Season'])['rbi_pred']       # ← This uses the correct rbi_pred column
        .mean().reset_index()
        .rename(columns={'rbi_pred':'rbi_pred_mean'})
    )
    
    meta_df = (
        prior_season_tot
        .merge(p_seas, on=['Name','Season'], how='inner')
        .merge(r_seas, on=['Name','Season'], how='inner')
        .merge(prior_labels_df, on=['Name','Season'], how='inner')
    )
    
    print(f"Meta-training set: {len(meta_df)} pitcher-seasons")
    
    if len(meta_df) == 0:
        print("❌ No complete meta-training data. Check data merging.")
        exit(1)
    
    # Import the fixed module
    from fixed_meta_model_training import train_robust_meta_pipeline

    # Use robust training
    try:
        meta_model = train_robust_meta_pipeline(meta_df)
        if meta_model is None:
            print("❌ Meta-model training failed - using fallback")
            # Your existing fallback can stay
    except Exception as e:
        print(f"❌ Meta-model error: {e}")
        meta_model = None
        print("✅ Meta-model saved")

# REPLACE THE ABOVE SECTION WITH THIS:

    # Meta-model training with feature diagnostics
    print("\n" + "="*50)
    print("META-LEARNING MODEL")
    print("="*50)
    
    # DEBUG: Check if features were actually created
    print("🔍 DEBUG: Checking if features were created correctly...")
    print(f"prior_pitch_df has pitch_score: {'pitch_score' in prior_pitch_df.columns}")
    print(f"game_df has rbi_pred: {'rbi_pred' in game_df.columns}")
    
    if 'pitch_score' in prior_pitch_df.columns:
        print(f"Pitch scores: min={prior_pitch_df['pitch_score'].min():.6f}, max={prior_pitch_df['pitch_score'].max():.6f}, std={prior_pitch_df['pitch_score'].std():.6f}")
    
    if 'rbi_pred' in game_df.columns:
        print(f"RBI predictions: min={game_df['rbi_pred'].min():.6f}, max={game_df['rbi_pred'].max():.6f}, std={game_df['rbi_pred'].std():.6f}")
    
    # Aggregate features to season level
    p_seas = (
        prior_pitch_df
        .groupby(['Name','Season'])['pitch_score']
        .mean().reset_index()
        .rename(columns={'pitch_score':'pitch_score_mean'})
    )
    print(f"Pitch score aggregation: {len(p_seas)} name-season combinations")
    print(f"Pitch score means: min={p_seas['pitch_score_mean'].min():.6f}, max={p_seas['pitch_score_mean'].max():.6f}, std={p_seas['pitch_score_mean'].std():.6f}")
    
    r_seas = (
        game_df
        .groupby(['Name','Season'])['rbi_pred']
        .mean().reset_index()
        .rename(columns={'rbi_pred':'rbi_pred_mean'})
    )
    print(f"RBI prediction aggregation: {len(r_seas)} name-season combinations")
    print(f"RBI pred means: min={r_seas['rbi_pred_mean'].min():.6f}, max={r_seas['rbi_pred_mean'].max():.6f}, std={r_seas['rbi_pred_mean'].std():.6f}")
    
    # Create final meta-dataset
    meta_df = (
        prior_season_tot
        .merge(p_seas, on=['Name','Season'], how='inner')
        .merge(r_seas, on=['Name','Season'], how='inner')
        .merge(prior_labels_df, on=['Name','Season'], how='inner')
    )
    
    print(f"\nMeta-training set: {len(meta_df)} pitcher-seasons")
    
    # FINAL DEBUG: Check meta-features before training
    print("🔍 FINAL DEBUG: Meta-feature check before training:")
    for col in ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']:
        if col in meta_df.columns:
            values = meta_df[col]
            print(f"  {col}: min={values.min():.6f}, max={values.max():.6f}, std={values.std():.6f}, unique={values.nunique()}")
    
    print("🏆 Winner vs Non-winner differences:")
    if len(meta_df) > 0:
        winners = meta_df[meta_df['winner_flag'] == 1]
        non_winners = meta_df[meta_df['winner_flag'] == 0]
        for col in ['pitch_score_mean', 'rbi_pred_mean', 'bayes_effect']:
            if col in meta_df.columns and len(winners) > 0 and len(non_winners) > 0:
                w_mean = winners[col].mean()
                nw_mean = non_winners[col].mean()
                diff = w_mean - nw_mean
                print(f"  {col}: winners={w_mean:.6f}, non_winners={nw_mean:.6f}, diff={diff:.6f}")
    
    if len(meta_df) == 0:
        print("❌ No complete meta-training data. Check data merging.")
        exit(1)
    
    meta_model = train_meta_pipeline_FIXED(meta_df)
    print("✅ Meta-model saved")

    # Final diagnostics summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE - DIAGNOSTICS SUMMARY")
    print("="*70)
    
    print("\n📁 Generated Diagnostic Reports:")
    print("   📊 Data Quality:")
    print("      - diagnostics/data_quality/: Missing values, distributions, correlations")
    print("      - diagnostics/pipeline_overview.csv: Overall data consistency")
    print("   🏗️ Level 1 (Pitch Models):")
    print("      - diagnostics/level1/: Model performance, feature importance, confusion matrices")
    print("   🧠 Level 2 (LSTM):")
    print("      - diagnostics/level2/: Training history, prediction accuracy, sequence analysis")
    print("   📈 Level 3 (Bayesian):")
    print("      - diagnostics/level3/: Convergence diagnostics, posterior analysis, shrinkage effects")
    print("   🎯 Meta-Learning:")
    print("      - diagnostics/meta/: Feature selection, calibration, cross-validation results")
    
    print("\n💾 Saved Models:")
    print("   - models/level1_pipelines.pkl: Swing, whiff, and exit velocity models")
    print("   - models/level2_lstm.h5: Sequential game-level LSTM")
    print("   - models/level2_scaler.pkl: Feature scaler for LSTM")
    print("   - models/level2_game_feats.pkl: Game-level feature names")
    print("   - models/level3_bayes.pkl: Bayesian hierarchical model")
    print("   - models/meta_model.pkl: Final ensemble meta-model")
    print("   - models/meta_features.pkl: Meta-feature names")
    
    print("\n🎯 Next Steps:")
    print("   1. Review diagnostic plots to identify potential issues")
    print("   2. Use weekly_update.py for regular model updates and predictions")
    print("   3. Monitor model performance over time")
    
    print("\n✅ Training pipeline completed successfully!")
    print("   Review the diagnostics folder to understand model behavior and potential improvements.")
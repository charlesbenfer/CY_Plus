import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def diagnose_meta_model_issues():
    """Comprehensive diagnosis of meta-model issues"""
    print("🔍 DIAGNOSING META-MODEL ISSUES")
    print("=" * 50)
    
    # Try to load and examine the training data that was used
    try:
        # Load the models to understand what was trained
        meta_model = joblib.load('models/meta_model.pkl')
        meta_features = joblib.load('models/meta_features.pkl')
        print(f"✅ Loaded meta-model with features: {meta_features}")
    except FileNotFoundError:
        print("❌ Meta-model files not found. Please run training first.")
        return
    
    # Check if we can load any of the intermediate data from diagnostics
    try:
        # Look for meta-learning diagnostics
        meta_performance = pd.read_csv('diagnostics/meta/performance_summary.csv')
        print("\n📊 Meta-Model Performance Summary:")
        print(meta_performance)
        
        # Check feature importance if available
        try:
            feature_importance = pd.read_csv('diagnostics/meta/feature_importance.csv')
            print("\n📈 Feature Importance:")
            print(feature_importance)
        except FileNotFoundError:
            print("⚠️ Feature importance file not found")
            
    except FileNotFoundError:
        print("⚠️ Meta diagnostics files not found")
    
    # Analyze the class imbalance issue
    print("\n🎯 CLASS IMBALANCE ANALYSIS:")
    print("The meta-model shows 62 non-winners vs 4 winners (6% positive class)")
    print("This severe imbalance explains why:")
    print("  1. AUC = 0.5 (random performance)")
    print("  2. Model predicts all samples as negative class")
    print("  3. Precision/Recall for winners = 0.0")
    
    # Recommendations for fixing the issues
    print("\n🔧 RECOMMENDED FIXES:")
    print("\n1. CLASS IMBALANCE SOLUTIONS:")
    print("   a. Use class_weight='balanced' in LogisticRegression")
    print("   b. Use SMOTE or other oversampling techniques")
    print("   c. Use stratified sampling")
    print("   d. Adjust classification threshold")
    
    print("\n2. DATA QUALITY SOLUTIONS:")
    print("   a. Check if winner labels are correctly assigned")
    print("   b. Verify that features have meaningful variation")
    print("   c. Ensure features are properly scaled")
    print("   d. Check for data leakage")
    
    print("\n3. MODEL ARCHITECTURE SOLUTIONS:")
    print("   a. Try different algorithms (Random Forest, XGBoost)")
    print("   b. Use different scoring metrics (precision, recall, f1)")
    print("   c. Implement custom loss functions for imbalanced data")
    
    # Create visualization of the problem
    plt.figure(figsize=(15, 10))
    
    # Simulate the class distribution we observed
    classes = ['Non-Winners', 'Winners']
    counts = [62, 4]
    colors = ['lightcoral', 'lightblue']
    
    plt.subplot(2, 3, 1)
    plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors)
    plt.title('Class Distribution Problem\n(Severe Imbalance)')
    
    # Show what good vs bad AUC looks like
    plt.subplot(2, 3, 2)
    fpr_bad = [0, 0.5, 1]
    tpr_bad = [0, 0.5, 1]
    fpr_good = [0, 0.1, 0.3, 1]
    tpr_good = [0, 0.7, 0.9, 1]
    
    plt.plot(fpr_bad, tpr_bad, 'r--', label='Your Model (AUC=0.5)', linewidth=2)
    plt.plot(fpr_good, tpr_good, 'g-', label='Good Model (AUC=0.8)', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k:', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    
    # Show the confusion matrix pattern
    plt.subplot(2, 3, 3)
    cm = np.array([[62, 0], [4, 0]])  # Your model's pattern
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Pred: No', 'Pred: Yes'],
                yticklabels=['True: No', 'True: Yes'])
    plt.title('Your Model\'s Confusion Matrix\n(Never predicts winners)')
    
    # Feature scaling importance
    plt.subplot(2, 3, 4)
    # Simulate different feature scales that might cause issues
    feature_scales = [0.001, 0.1, 100, 10000]  # Representing different feature magnitudes
    feature_names_short = ['Pitch', 'RBI', 'Bayes', 'Interact']
    
    plt.bar(feature_names_short, feature_scales)
    plt.yscale('log')
    plt.title('Feature Scale Differences\n(May cause issues)')
    plt.ylabel('Magnitude (log scale)')
    plt.xticks(rotation=45)
    
    # Recommended class balancing techniques
    plt.subplot(2, 3, 5)
    techniques = ['Original', 'SMOTE', 'Class Weight', 'Threshold\nTuning']
    effectiveness = [0.5, 0.75, 0.7, 0.65]  # Simulated AUC improvements
    colors_tech = ['red', 'orange', 'yellow', 'lightgreen']
    
    bars = plt.bar(techniques, effectiveness, color=colors_tech)
    plt.ylabel('Expected AUC')
    plt.title('Class Imbalance Solutions')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Current')
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target')
    
    # Add value labels on bars
    for bar, val in zip(bars, effectiveness):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.legend()
    
    # Data quality checklist
    plt.subplot(2, 3, 6)
    plt.text(0.05, 0.95, 'DATA QUALITY CHECKLIST:', fontsize=12, fontweight='bold', 
             transform=plt.gca().transAxes, va='top')
    
    checklist = [
        '□ Cy Young labels correct?',
        '□ Features have variation?',
        '□ No data leakage?',
        '□ Proper time splits?',
        '□ Feature scaling applied?',
        '□ Missing values handled?',
        '□ Outliers identified?',
        '□ Cross-validation strategy?'
    ]
    
    for i, item in enumerate(checklist):
        plt.text(0.05, 0.85 - i*0.1, item, fontsize=10, 
                transform=plt.gca().transAxes, va='top')
    
    plt.axis('off')
    plt.title('Pre-Training Checklist')
    
    plt.tight_layout()
    plt.savefig('diagnostics/meta_model_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n💡 IMMEDIATE ACTION ITEMS:")
    print("1. 🔍 Check your winner_labels.csv file:")
    print("   - Are there really only 4 Cy Young winners in your dataset?")
    print("   - Should there be more winners (multiple years, leagues)?")
    print("   - Are the labels correctly assigned?")
    
    print("\n2. 🔧 Modify the meta-model training to handle class imbalance:")
    print("   - Add class_weight='balanced' to LogisticRegression")
    print("   - Use stratified cross-validation")
    print("   - Consider SMOTE for oversampling")
    
    print("\n3. 📊 Examine feature distributions:")
    print("   - Check if features actually differ between winners/non-winners")
    print("   - Look at the feature correlation plots in diagnostics/meta/")
    
    print(f"\n📁 Diagnosis plot saved to: diagnostics/meta_model_diagnosis.png")


def create_improved_meta_model():
    """Create an improved meta-model that handles class imbalance"""
    print("\n🔧 CREATING IMPROVED META-MODEL")
    print("=" * 40)
    
    # This is a template for how to fix the meta-model
    template_code = '''
# IMPROVED META-MODEL CODE (add this to your train_models.py)

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

def train_improved_meta_pipeline(meta_df: pd.DataFrame):
    """Enhanced meta-model with class imbalance handling"""
    
    # Check class distribution
    class_counts = meta_df['winner_flag'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    if class_counts.min() < 5:
        print("⚠️ Very few positive examples. Consider:")
        print("  1. Adding more years of data")
        print("  2. Including runner-up awards")
        print("  3. Using different target definition")
    
    # Feature engineering with better handling
    meta_features = meta_df[['pitch_score_mean','rbi_pred_mean','bayes_effect']].copy()
    
    # Add interactions
    meta_features['pitch_rbi_interaction'] = (
        meta_features['pitch_score_mean'] * meta_features['rbi_pred_mean']
    )
    meta_features['pitch_bayes_interaction'] = (
        meta_features['pitch_score_mean'] * meta_features['bayes_effect']
    )
    meta_features['rbi_bayes_interaction'] = (
        meta_features['rbi_pred_mean'] * meta_features['bayes_effect']
    )
    
    X = meta_features.values
    y = meta_df['winner_flag'].values
    
    # Handle class imbalance with multiple strategies
    
    # Strategy 1: Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Strategy 2: SMOTE (if you have imblearn installed)
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Improved pipeline
    from sklearn.ensemble import RandomForestClassifier  # Often better for imbalanced data
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k='all')),
        ('clf', LogisticRegression(
            class_weight='balanced',  # KEY FIX!
            solver='liblinear', 
            random_state=42,
            max_iter=1000
        ))
    ])
    
    # Use stratified CV to maintain class balance
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds due to small dataset
    
    param_grid = {
        'selector__k': [3, 4, 5, 6],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }
    
    gs = GridSearchCV(
        pipe, 
        param_grid, 
        cv=cv,  # Use stratified CV
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    gs.fit(X, y)
    
    # Evaluate with multiple metrics for imbalanced data
    y_pred = gs.predict(X)
    y_prob = gs.predict_proba(X)[:, 1]
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)
    
    print(f"Improved Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Best params: {gs.best_params_}")
    
    return gs.best_estimator_
'''
    
    print("📝 Template code for improved meta-model:")
    print(template_code)
    
    # Save the template to a file
    with open('improved_meta_model_template.py', 'w') as f:
        f.write(template_code)
    
    print(f"💾 Template saved to: improved_meta_model_template.py")


if __name__ == '__main__':
    diagnose_meta_model_issues()
    create_improved_meta_model()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Examine your winner_labels.csv file")
    print("2. Consider expanding your dataset (more years, both leagues)")
    print("3. Apply the improved meta-model template")
    print("4. Re-run training with the fixes")
    print("5. Check if AUC improves significantly (target: >0.7)")
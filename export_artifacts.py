"""
Artifact Export Module for Predictive Provenance System

This module handles exporting trained models, metrics, and reports to disk.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch

def _to_native(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj

def create_export_directories():
    """Create necessary directories for artifact export"""
    directories = ['models', 'reports', 'predictions', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def export_traditional_models(predictor, export_dir='models'):
    """Export traditional ML models (Random Forest, Gradient Boosting)"""
    os.makedirs(export_dir, exist_ok=True)
    
    exported_models = {}
    for name, model_info in predictor.models.items():
        model = model_info['model']
        score = model_info['score']
        
        # Save model
        model_path = os.path.join(export_dir, f'{name.lower().replace(" ", "_")}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        exported_models[name] = {
            'model_path': model_path,
            'score': score,
            'type': 'traditional'
        }
        
        print(f"✓ Exported {name} model to {model_path}")
    
    return exported_models

def export_neural_model(lstm_trainer, export_dir='models'):
    """Export LSTM neural network model"""
    os.makedirs(export_dir, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(export_dir, 'lstm_model.pth')
    torch.save({
        'model_state_dict': lstm_trainer.model.state_dict(),
        'model_architecture': {
            'input_size': lstm_trainer.model.input_size,
            'hidden_size': lstm_trainer.model.hidden_size,
            'num_classes': lstm_trainer.model.num_classes,
            'num_layers': lstm_trainer.model.num_layers
        }
    }, model_path)
    
    print(f"✓ Exported LSTM model to {model_path}")
    
    return {
        'model_path': model_path,
        'type': 'neural'
    }

def export_metrics(y_true, y_pred, labels, model_name, export_dir='reports'):
    """Export detailed metrics and confusion matrix"""
    os.makedirs(export_dir, exist_ok=True)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create metrics summary
    metrics = {
        'model_name': model_name,
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg'],
        'per_class_metrics': {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']},
        'confusion_matrix': cm.tolist(),
        'class_labels': labels
    }
    
    # Save metrics
    metrics_path = os.path.join(export_dir, f'{model_name.lower().replace(" ", "_")}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = os.path.join(export_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    
    print(f"✓ Exported {model_name} metrics to {metrics_path}")
    print(f"✓ Exported {model_name} confusion matrix to {cm_path}")
    
    return metrics_path, cm_path

def export_predictions(X_test, y_test, predictor, real_data_processor, export_dir='predictions'):
    """Export predictions for all models"""
    os.makedirs(export_dir, exist_ok=True)
    
    predictions_data = []
    
    for model_name, model_info in predictor.models.items():
        model = model_info['model']
        predictions = model.predict(X_test)
        
        # Convert predictions back to labels
        if hasattr(real_data_processor, 'label_encoder'):
            pred_labels = real_data_processor.label_encoder.inverse_transform(predictions)
            true_labels = real_data_processor.label_encoder.inverse_transform(y_test)
        else:
            pred_labels = predictions
            true_labels = y_test
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': pred_labels,
            'model': model_name,
            'correct': true_labels == pred_labels
        })
        
        # Save predictions
        pred_path = os.path.join(export_dir, f'{model_name.lower().replace(" ", "_")}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        
        predictions_data.append({
            'model': model_name,
            'predictions': pred_df,
            'file_path': pred_path
        })
        
        print(f"✓ Exported {model_name} predictions to {pred_path}")
    
    return predictions_data

def export_summary_report(results, export_dir='reports'):
    """Export comprehensive summary report"""
    os.makedirs(export_dir, exist_ok=True)
    
    sequences = results['sequences']
    labels = results['labels']
    models = results['models']
    patterns = results['patterns']
    
    # Create comprehensive report
    report = {
        'data_statistics': {
            'total_sequences': len(sequences),
            'average_sequence_length': float(np.mean([len(seq) for seq in sequences])),
            'behavior_distribution': dict(pd.Series(labels).value_counts()),
            'unique_behavior_types': len(set(labels))
        },
        'model_performance': {
            name: {
                'score': float(model_info['score']),
                'type': 'traditional'
            } for name, model_info in models.items()
        },
        'interaction_patterns': {
            'total_patterns': len(patterns),
            'top_patterns': [
                {'pattern': str(pattern), 'count': int(count)} 
                for pattern, count in patterns[:10]
            ]
        },
        'system_info': {
            'python_version': '3.x',
            'libraries_used': ['scikit-learn', 'torch', 'pandas', 'numpy', 'matplotlib', 'seaborn'],
            'data_source': 'projection-space-explorer',
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save report (ensure JSON-serializable types)
    report_path = os.path.join(export_dir, 'comprehensive_report.json')
    with open(report_path, 'w') as f:
        json.dump(_to_native(report), f, indent=2)
    
    # Create markdown summary
    markdown_content = f"""# Predictive Provenance System - Analysis Report

## Data Statistics
- **Total Sequences**: {len(sequences)}
- **Average Sequence Length**: {np.mean([len(seq) for seq in sequences]):.1f}
- **Behavior Types**: {len(set(labels))}

### Behavior Distribution
"""
    
    for behavior, count in pd.Series(labels).value_counts().items():
        percentage = count/len(labels)*100
        markdown_content += f"- **{behavior}**: {count} ({percentage:.1f}%)\n"
    
    markdown_content += f"""
## Model Performance
"""
    
    for name, model_info in models.items():
        markdown_content += f"- **{name}**: {model_info['score']:.3f}\n"
    
    markdown_content += f"""
## Top Interaction Patterns
"""
    
    for i, (pattern, count) in enumerate(patterns[:10]):
        markdown_content += f"{i+1}. **{pattern}**: {count} occurrences\n"
    
    markdown_content += f"""
## Export Information
- **Export Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Data Source**: projection-space-explorer repository
- **Models Exported**: {len(models)} traditional models + 1 neural model
"""
    
    # Save markdown report
    markdown_path = os.path.join(export_dir, 'analysis_report.md')
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"✓ Exported comprehensive report to {report_path}")
    print(f"✓ Exported markdown summary to {markdown_path}")
    
    return report_path, markdown_path

def export_all_artifacts(results, predictor, lstm_trainer, X_test, y_test, real_data_processor):
    """Export all artifacts in one function"""
    print("\n" + "=" * 60)
    print("EXPORTING ARTIFACTS")
    print("=" * 60)
    
    # Create directories
    directories = create_export_directories()
    print(f"✓ Created directories: {', '.join(directories)}")
    
    # Export traditional models
    print("\nExporting traditional models...")
    traditional_models = export_traditional_models(predictor)
    
    # Export neural model
    print("\nExporting neural model...")
    neural_model = export_neural_model(lstm_trainer)
    
    # Export predictions
    print("\nExporting predictions...")
    predictions = export_predictions(X_test, y_test, predictor, real_data_processor)
    
    # Export metrics for each model
    print("\nExporting metrics...")
    metrics_files = []
    for model_name, model_info in predictor.models.items():
        model = model_info['model']
        predictions = model.predict(X_test)
        
        if hasattr(real_data_processor, 'label_encoder'):
            pred_labels = real_data_processor.label_encoder.inverse_transform(predictions)
            true_labels = real_data_processor.label_encoder.inverse_transform(y_test)
        else:
            pred_labels = predictions
            true_labels = y_test
        
        unique_labels = sorted(list(set(true_labels) | set(pred_labels)))
        metrics_path, cm_path = export_metrics(true_labels, pred_labels, unique_labels, model_name)
        metrics_files.extend([metrics_path, cm_path])
    
    # Export summary report
    print("\nExporting summary report...")
    report_path, markdown_path = export_summary_report(results)
    
    print(f"\n✅ All artifacts exported successfully!")
    print(f"📁 Check the following directories:")
    for directory in directories:
        print(f"   • {directory}/")
    
    return {
        'traditional_models': traditional_models,
        'neural_model': neural_model,
        'predictions': predictions,
        'metrics_files': metrics_files,
        'summary_report': report_path,
        'markdown_report': markdown_path
    }

if __name__ == "__main__":
    print("Artifact Export Module - Ready for use")
    print("This module is imported by demo_notebook.py for artifact export")

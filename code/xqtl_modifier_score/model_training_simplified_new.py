import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

pbar = ProgressBar(dt=1)
pbar.register()

import time
from sklearn.model_selection import cross_val_score, BaseCrossValidator
from sklearn.model_selection import LeaveOneGroupOut

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, GPSampler

from tqdm import tqdm
tqdm.pandas()

import requests
import numpy as np
from os import walk
import dask

import pickle

dask.config.set({'temporary_directory': '/nfs/scratch'})

import json
import pysam
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn import metrics
import sklearn
from sklearn.linear_model import SGDClassifier
import yaml
import pickle
import torch
import random
from sklearn.calibration import CalibratedClassifierCV
import optunahub
import joblib
from catboost import CatBoostClassifier


class ConfigurationManager:
    """Manages loading and validation of configuration files."""
    
    def __init__(self, data_config_path, model_config_path, data_params_path=None):
        self.data_config = self._load_yaml(data_config_path)
        self.model_config = self._load_yaml(model_config_path)
        
        # Load legacy data_params.yaml if provided (for backward compatibility)
        if data_params_path and os.path.exists(data_params_path):
            self.data_params = self._load_yaml(data_params_path)
        else:
            self.data_params = {}
    
    def _load_yaml(self, path):
        """Load YAML configuration file with error handling."""
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {path}: {e}")
    
    def get_data_config(self):
        """Get data configuration with legacy fallback."""
        config = self.data_config.copy()
        
        # Merge with legacy data_params for backward compatibility
        if self.data_params:
            if 'train' in self.data_params and 'test' in self.data_params:
                config['thresholds'] = {
                    'train': self.data_params['train'],
                    'test': self.data_params['test']
                }
        
        return config
    
    def get_model_config(self):
        """Get model configuration."""
        return self.model_config
    
    def get_file_paths(self, cohort, chromosome):
        """Generate file paths based on configuration and parameters."""
        data_config = self.get_data_config()
        
        # Gene constraint file
        gene_file = data_config['gene_constraint_file']
        
        # MAF file
        maf_file = data_config['maf_file_pattern'].format(chromosome=chromosome)
        
        # Data directories
        base_dir = data_config['training_data']['base_dir'].format(cohort=cohort)
        output_dir = data_config['output']['base_dir'].format(cohort=cohort)
        
        return {
            'gene_constraint_file': gene_file,
            'maf_file': maf_file,
            'base_data_dir': base_dir,
            'output_dir': output_dir,
            'predictions_dir': os.path.join(output_dir, data_config['output']['predictions_dir']),
            'columns_dict_file': data_config['columns_dict_file']
        }


def set_random_seeds(config):
    """Set random seeds for reproducibility."""
    seed = config['training']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_variant_features(df):
    """Create variant-based features from variant_id."""
    df[['chr','pos','ref','alt']] = df['variant_id'].str.split(':', expand=True)
    df['length_diff'] = df['ref'].str.len() - df['alt'].str.len()
    df['is_SNP'] = df['length_diff'].apply(lambda x: 1 if x == 0 else 0)
    df['is_indel'] = df['length_diff'].apply(lambda x: 1 if x != 0 else 0)
    df['is_insertion'] = df['length_diff'].apply(lambda x: 1 if x > 0 else 0)
    df['is_deletion'] = df['length_diff'].apply(lambda x: 1 if x < 0 else 0)
    df.drop(columns=['chr','pos','ref','alt'], inplace=True)
    
    # Make label the last column
    cols = df.columns.tolist()
    cols.insert(len(cols)-1, cols.pop(cols.index('label')))
    df = df.loc[:, cols]
    return df


def load_gene_constraint_data(file_path, config):
    """Load and process gene constraint data."""
    gene_config = config['gene_constraint_columns']
    sheet_name = config['gene_constraint_sheet']
    
    df = pd.read_excel(file_path, sheet_name)
    df = df[[gene_config['gene_id'], gene_config['gene_lof']]]
    df = df.rename(columns={
        gene_config['gene_id']: 'gene_id', 
        gene_config['gene_lof']: 'gene_lof'
    })
    df['gene_lof'] = np.log2(df['gene_lof'])
    return df


def load_maf_data(file_path, config):
    """Load and process MAF data."""
    maf_config = config['maf_columns']
    df = dd.read_csv(file_path, sep='\t')
    df = df[[maf_config['variant_id'], maf_config['maf']]].compute()
    return df


def load_training_data(base_dir, cohort, chromosomes, npr, thresholds, file_pattern, is_test=False):
    """Load training or test data from parquet files."""
    files = []
    valid_chromosomes = []
    
    threshold_type = 'test' if is_test else 'train'
    dir_pattern = f"{'test' if is_test else 'train'}_NPR_{npr}_PIP_{thresholds[threshold_type]['positive_class_threshold']}_{thresholds[threshold_type]['negative_class_threshold']}"
    
    for chromosome in chromosomes:
        file_path = os.path.join(base_dir, dir_pattern, file_pattern.format(cohort=cohort, chromosome=chromosome))
        
        if os.path.exists(file_path):
            files.append(file_path)
            valid_chromosomes.append(chromosome)
        else:
            print(f"Warning: File {file_path} does not exist. Skipping chromosome {chromosome}.")
    
    if not files:
        raise ValueError(f"No {'test' if is_test else 'training'} files found for any chromosome. Cannot proceed.")
    
    df = dd.read_parquet(files, engine='pyarrow').compute()
    return df, valid_chromosomes


def create_feature_weights(columns, config):
    """Create feature weights dictionary based on configuration."""
    feature_weights = {}
    default_weight = config['feature_weights']['default_weight']
    high_priority_config = config['feature_weights']['high_priority_features']
    
    # Set default weight for all features
    for col in columns:
        feature_weights[col] = default_weight
    
    # Set high priority weights for specified feature patterns
    for col in columns:
        for pattern in high_priority_config['feature_patterns']:
            if pattern in col:
                feature_weights[col] = high_priority_config['weight']
                break
    
    return feature_weights


def calculate_class_weights(df):
    """Calculate class weights for training data."""
    class_0 = df[df['label'] == 0].shape[0]
    class_1 = df[df['label'] == 1].shape[0]
    total_pip = df[df['label'] == 1].pip.sum()
    pip_percent = class_0 / total_pip if total_pip > 0 else 1
    
    df['weight'] = np.where(df['label'] == 0, 1, df['pip'] * pip_percent)
    return df


def prepare_features(df, config, column_dict):
    """Prepare feature sets based on configuration."""
    # Get feature configuration
    feature_config = config['features']
    
    # Remove specified columns
    columns_to_remove = feature_config['columns_to_remove']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Create subset columns
    subset_keys = feature_config['subset_keys']
    subset_cols = []
    for key in subset_keys:
        if key in column_dict:
            subset_cols.extend(column_dict[key])
    
    # Keep only columns that exist in the dataframe
    subset_cols = [col for col in subset_cols if col in df.columns]
    
    # Add variant features
    variant_features = feature_config['variant_features']
    subset_cols.extend(variant_features)
    
    # Get columns for absolute value transformation
    abs_keys = feature_config['absolute_value_keys']
    columns_to_abs = []
    for key in abs_keys:
        if key in column_dict:
            columns_to_abs.extend([col for col in column_dict[key] if col in df.columns])
    
    return subset_cols, columns_to_abs


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train eQTL prediction model")
    parser.add_argument("cohort", type=str, help="Cohort/project name (e.g., Mic_mega_eQTL)")
    parser.add_argument("chromosome", type=str, help="Chromosome number (e.g., 2)")
    parser.add_argument("--gene_lof_file", type=str, required=True, 
                       help="Path to Excel file (e.g., 41588_2024_1820_MOESM4_ESM.xlsx)")
    parser.add_argument("--data_config", type=str, default="data_config.yaml", 
                       help="Path to data configuration YAML file")
    parser.add_argument("--model_config", type=str, default="model_config.yaml", 
                       help="Path to model configuration YAML file")
    parser.add_argument("--data_params", type=str, default="data_params.yaml", 
                       help="Path to legacy data_params.yaml (optional)")
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(args.data_config, args.model_config, args.data_params)
    data_config = config_manager.get_data_config()
    model_config = config_manager.get_model_config()
    
    # Set random seeds
    set_random_seeds(model_config)
    
    # Get file paths
    file_paths = config_manager.get_file_paths(args.cohort, args.chromosome)
    
    # Use provided gene_lof_file or fall back to config
    gene_lof_file = args.gene_lof_file or file_paths['gene_constraint_file']
    
    # Configuration parameters
    sampling_config = data_config['sampling']
    NPR_tr = sampling_config['npr_train']
    NPR_te = sampling_config['npr_test']
    
    chromosome_out = f'chr{args.chromosome}'
    chromosomes = [chromosome_out]
    train_chromosomes = chromosomes
    test_chromosomes = chromosomes
    
    # Create output directories
    write_dir = file_paths['output_dir']
    predictions_dir = file_paths['predictions_dir']
    
    for directory in [write_dir, predictions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load gene constraint data
    print("Loading gene constraint data...")
    gene_lof_df = load_gene_constraint_data(gene_lof_file, data_config)
    
    # Load MAF data
    print("Loading MAF data...")
    maf_df = load_maf_data(file_paths['maf_file'], data_config)
    
    # Load column dictionary
    with open(file_paths['columns_dict_file'], 'rb') as f:
        column_dict = pickle.load(f)
    
    # Load training data
    print("Loading training data...")
    train_df, valid_train_chromosomes = load_training_data(
        file_paths['base_data_dir'], 
        args.cohort, 
        train_chromosomes, 
        NPR_tr, 
        data_config['thresholds'], 
        data_config['training_data']['file_pattern']
    )
    
    # Load test data
    print("Loading test data...")
    test_df, _ = load_training_data(
        file_paths['base_data_dir'], 
        args.cohort, 
        test_chromosomes, 
        NPR_te, 
        data_config['thresholds'], 
        data_config['training_data']['file_pattern'],
        is_test=True
    )
    
    # Process training data
    print("Processing training data...")
    train_df = make_variant_features(train_df)
    train_df = train_df.merge(gene_lof_df, on='gene_id', how='left')
    train_df = train_df.merge(maf_df, on='variant_id', how='left')
    train_df['gene_lof'] = train_df['gene_lof'].fillna(train_df['gene_lof'].median())
    train_df['gnomad_MAF'] = train_df['gnomad_MAF'].fillna(train_df['gnomad_MAF'].median())
    train_df = calculate_class_weights(train_df)
    
    # Process test data
    print("Processing test data...")
    test_df = make_variant_features(test_df)
    test_df = test_df.merge(gene_lof_df, on='gene_id', how='left')
    test_df = test_df.merge(maf_df, on='variant_id', how='left')
    test_df['gene_lof'] = test_df['gene_lof'].fillna(test_df['gene_lof'].median())
    test_df['gnomad_MAF'] = test_df['gnomad_MAF'].fillna(test_df['gnomad_MAF'].median())
    test_df = calculate_class_weights(test_df)
    
    # Print class distributions
    print("Training data class distribution:")
    print(train_df['label'].value_counts())
    print("Test data class distribution:")
    print(test_df['label'].value_counts())
    
    # Prepare features
    print("Preparing features...")
    meta_data = data_config['metadata_columns']
    
    X_train = train_df.drop(columns=meta_data)
    Y_train = train_df['label']
    weight_train = train_df['weight']
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_train = X_train.fillna(0)
    
    X_test = test_df.drop(columns=meta_data)
    Y_test = test_df['label']
    weight_test = test_df['weight']
    X_test = X_test.replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0)
    
    # Remove gene_id if present
    if 'gene_id' in X_train.columns:
        X_train = X_train.drop(columns=['gene_id'])
        X_test = X_test.drop(columns=['gene_id'])
    
    # Prepare feature subsets
    subset_cols, columns_to_abs = prepare_features(X_train, data_config, column_dict)
    
    # Create subset dataframes
    X_train_subset = X_train[subset_cols].copy()
    X_test_subset = X_test[subset_cols].copy()
    
    # Apply absolute values to specified columns
    for col in columns_to_abs:
        if col in X_train_subset.columns:
            X_train_subset[col] = X_train_subset[col].abs()
            X_test_subset[col] = X_test_subset[col].abs()
    
    # Create feature weights
    feature_weights = create_feature_weights(X_train_subset.columns, model_config)
    
    print("Feature weights distribution:")
    print(f"Number of features with weight 10.0: {sum(value == 10.0 for value in feature_weights.values())}")
    print(f"Number of features with weight 1.0: {sum(value == 1.0 for value in feature_weights.values())}")
    
    # Train models based on configuration
    models = {}
    predictions = {}
    metrics_dict = {}
    
    model_variants = model_config['model_variants']
    catboost_params = model_config['catboost_params']
    
    for model_id, variant_config in model_variants.items():
        print(f"Training {model_id}: {variant_config['description']}")
        
        # Get parameters for this variant
        params = catboost_params[variant_config['params']].copy()
        
        # Create model
        model_kwargs = {
            **params,
            'name': variant_config['name']
        }
        
        # Add feature weights if specified
        if variant_config['use_feature_weights']:
            model_kwargs['feature_weights'] = feature_weights
        
        model = CatBoostClassifier(**model_kwargs)
        
        # Train model
        model.fit(X_train_subset, Y_train, sample_weight=weight_train)
        models[model_id] = model
        
        # Make predictions
        pred_proba = model.predict_proba(X_test_subset)[:, 1]
        predictions[model_id] = pred_proba
        
        # Calculate metrics
        metrics_dict[model_id] = {
            'AP': metrics.average_precision_score(Y_test, pred_proba),
            'AUC': metrics.roc_auc_score(Y_test, pred_proba)
        }
    
    # Print metrics for all models
    print("\nTest Set Metrics:")
    for model_id, model_metrics in metrics_dict.items():
        variant_config = model_variants[model_id]
        print(f"{model_id}. {variant_config['description']} - AP: {model_metrics['AP']:.4f}, AUC: {model_metrics['AUC']:.4f}")
    
    # Save models
    output_patterns = model_config['training']['output_patterns']
    for model_id, model in models.items():
        variant_name = model_variants[model_id]['name'].lower()
        filename = output_patterns['model_file'].format(
            variant_name=variant_name, 
            chromosome=chromosome_out, 
            npr=NPR_tr
        )
        joblib.dump(model, os.path.join(write_dir, filename))
    
    # Get feature importances
    feature_importances = {}
    feature_dfs = []
    
    for model_id, model in models.items():
        feature_importances[model_id] = {
            'importances': model.feature_importances_,
            'features': X_train_subset.columns
        }
        
        # Create feature importance dataframe
        model_df = pd.DataFrame({
            'feature': X_train_subset.columns,
            'importance': model.feature_importances_,
            'model': model_id
        })
        model_df = model_df.sort_values(by='importance', ascending=False)
        feature_dfs.append(model_df)
        
        # Print top 20 features
        print(f"\nTop 20 features for {model_id}:")
        for i, (feature, importance) in enumerate(zip(model_df['feature'][:20], model_df['importance'][:20])):
            print(f"{i + 1}. {feature}: {importance:.6f}")
    
    # Save feature importances
    all_features_df = pd.concat(feature_dfs, ignore_index=True)
    feature_importance_file = output_patterns['feature_importance_file'].format(
        chromosome=chromosome_out, npr=NPR_tr
    )
    all_features_df.to_csv(os.path.join(write_dir, feature_importance_file), index=False)
    
    # Create summary dictionary
    summary_dict = {
        'CatBoost': {
            model_id: {
                'AP_test': metrics_dict[model_id]['AP'],
                'AUC_test': metrics_dict[model_id]['AUC'],
                'params': catboost_params[model_variants[model_id]['params']],
                **(
                    {'feature_weights': 'chrombpnet_positive, tf_positive, and diff features set to 10.0, others to 1.0'} 
                    if model_variants[model_id]['use_feature_weights'] else {}
                )
            }
            for model_id in model_variants.keys()
        }
    }
    
    # Add class distribution info
    summary_dict['CatBoost'].update({
        'test_num_positive_labels': Y_test.value_counts().get(1, 0),
        'test_num_negative_labels': Y_test.value_counts().get(0, 0),
        'train_standard_num_positive_labels': Y_train.value_counts().get(1, 0),
        'train_standard_num_negative_labels': Y_train.value_counts().get(0, 0)
    })
    
    print('Writing results to file')
    
    # Save summary dictionary
    summary_file = output_patterns['summary_file'].format(chromosome=chromosome_out, npr=NPR_tr)
    with open(os.path.join(write_dir, summary_file), 'wb') as f:
        pickle.dump(summary_dict, f)
    
    # Add predictions to test dataframe
    for model_id, pred_proba in predictions.items():
        variant_name = model_variants[model_id]['name'].lower()
        test_df[f'{variant_name}_pred_prob'] = pred_proba
        test_df[f'{variant_name}_pred_label'] = models[model_id].predict(X_test_subset)
    
    test_df['actual_label'] = Y_test
    
    # Save predictions
    predictions_file = output_patterns['predictions_file'].format(chromosome=args.chromosome)
    test_df.to_csv(os.path.join(predictions_dir, predictions_file), sep='\t', index=False)
    
    # Save feature weights and subset columns for reference
    feature_weights_file = output_patterns['feature_weights_file'].format(
        chromosome=args.chromosome, npr=NPR_tr
    )
    with open(os.path.join(write_dir, feature_weights_file), 'wb') as f:
        pickle.dump(feature_weights, f)
    
    subset_columns_file = output_patterns['subset_columns_file'].format(
        chromosome=args.chromosome, npr=NPR_tr
    )
    with open(os.path.join(write_dir, subset_columns_file), 'wb') as f:
        pickle.dump({
            'subset_columns': subset_cols,
            'abs_columns': columns_to_abs
        }, f)
    
    print(f"\nTraining and evaluation complete for {len(model_variants)} models.")


if __name__ == "__main__":
    main()
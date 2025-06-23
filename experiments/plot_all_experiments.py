import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.rc('axes', grid=True)
# Define hyperparameters for each method
hyperparameters = {
    'ES': ['mu', 'lambda_', 'variant_mu_lambda', 'modified_ES'],
    'DE': ['NP', 'F', 'CR', 'max_generations', 'initial_lower', 'initial_upper'],
    'DEAW': ['NP', 'F', 'CR', 'max_generations', 'initial_lower', 'initial_upper'],
    'EDEAdam': ['pop_size', 'max_evals', 'exchange_interval', 'batch_size']
}

with open('experiments/results/history_all_final.json', 'r') as f:
    results = json.load(f)

def load_and_parse_data(method, results):
    """Load and parse data for a specific method from JSON file."""

    data_rows = []
    for key, seeds_data in results.items():
        # Rozdziel na 4 części od końca
        parts = key.split('_')
        dataset = parts[0]
        model = parts[1]
        current_method = parts[2]
        config_str = '_'.join(parts[3:])  # obsługuje podkreślniki w config

        if current_method != method:
            continue
        try:
            config = ast.literal_eval(config_str)
        except Exception as e:
            print(f"Nie można sparsować config_str: {config_str} dla klucza {key}: {e}")
            continue

        for seed, data in seeds_data.items():
            history = data['history']
            row = {
                'dataset': dataset,
                'model': model,
                'method': method,
                'seed': int(seed),
                'training_time': data['training_time'],
                'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history else np.nan,
                'test_accuracy': history['test_accuracy'][0] if 'test_accuracy' in history else np.nan,
                'history': history
            }
            # Add only relevant hyperparameters
            for param in hyperparameters[method]:
                row[param] = config.get(param, np.nan)
            data_rows.append(row)

    return pd.DataFrame(data_rows)

def generate_summary(df, method):
    """Generate summary statistics for a given method's DataFrame."""
    group_cols = ['dataset', 'model'] + hyperparameters[method]
    summary = df.groupby(group_cols).agg({
        'training_time': ['mean', 'std'],
        'final_val_accuracy': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
        'seed': 'count'
    }).reset_index()

    # Spłaszcz MultiIndex
    summary.columns = ['_'.join(col) if isinstance(col, tuple) and col[1] else col[0] for col in summary.columns.values]
    
    return summary

def print_tables(summary, method):
    """Print tables for a method, dataset, and model."""
    print(f"\nMethod: {method}")
    for dataset in summary['dataset'].unique():
        for model in summary['model'].unique():
            subset = summary[(summary['dataset'] == dataset) & (summary['model'] == model)]
            if not subset.empty:
                print(f"\nDataset: {dataset}, Model: {model}")
                display_cols = hyperparameters[method] + [
                    'training_time_mean', 'training_time_std',
                    'test_accuracy_mean', 'test_accuracy_std'
                ]
                print(subset[display_cols].to_string(index=False))

def plot_es_hyperparameters(summary, method):
    """Plot hyperparameter effects for ES method."""
    if method != 'ES':
        return
    for dataset in ['wine', 'iris', 'bcw']:
        for model in ['MLP', 'DeepMLP']:
            # Test accuracy vs. mu for lambda_=50
            subset_mu = summary[(summary['dataset'] == dataset) & 
                                (summary['model'] == model) & 
                                (summary['lambda_'] == 50)]
            if not subset_mu.empty:
                plt.figure(figsize=(8, 6))
                sns.lineplot(x='mu', y='test_accuracy_mean', hue='modified_ES', data=subset_mu)
                plt.title(f'ES on {dataset} - {model}: Test Accuracy vs. mu (lambda_=50)')
                plt.xlabel('mu')
                plt.ylabel('Average Test Accuracy')
                plt.legend(title='modified_ES')
                plt.savefig(f'experiments/plots/es_mu_effect/es_mu_effect_{dataset}_{model}.png')
                plt.close()
            
            # Test accuracy vs. lambda_ for mu=50
            subset_lambda = summary[(summary['dataset'] == dataset) & 
                                    (summary['model'] == model) & 
                                    (summary['mu'] == 50)]
            if not subset_lambda.empty:
                plt.figure(figsize=(8, 6))
                sns.lineplot(x='lambda_', y='test_accuracy_mean', hue='modified_ES', data=subset_lambda)
                plt.title(f'ES on {dataset} - {model}: Test Accuracy vs. lambda_ (mu=50)')
                plt.xlabel('lambda_')
                plt.ylabel('Average Test Accuracy')
                plt.legend(title='modified_ES')
                plt.savefig(f'experiments/plots/es_lambda_effect/es_lambda_effect_{dataset}_{model}.png')
                plt.close()


def plot_training_curves(df, summary, method):
    """Plot training curves for the best configuration of a method."""
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            subset = summary[(summary['dataset'] == dataset) & (summary['model'] == model)]
            if not subset.empty:
                best_row = subset.loc[subset['final_val_accuracy_mean'].idxmax()]
                best_config = {col: best_row[col] for col in hyperparameters[method]}
                config_rows = df
                for k, v in best_config.items():
                    config_rows = config_rows[config_rows[k] == v]
                histories = config_rows['history'].tolist()
                val_accuracies = [h['val_accuracy'] for h in histories]
                if len(set(len(v) for v in val_accuracies)) == 1:
                    avg_val_accuracy = np.mean(val_accuracies, axis=0)
                    generations = range(1, len(avg_val_accuracy) + 1)
                    plt.plot(generations, avg_val_accuracy, label=method)
                    plt.title(f'Training Curves on {dataset} - {model} ({method})')
                    plt.xlabel('Generation')
                    plt.ylabel('Validation Accuracy')
                    plt.legend()
                    plt.savefig(f'experiments/plots/training_curve/training_curve_{method}_{dataset}_{model}.png')
                    plt.close()


def analyze_method(method):
    """Perform full analysis for a specific method."""
    # Load data
    df = load_and_parse_data(method, results)
    if df.empty:
        print(f"No data found for method: {method}")
        return
    
    print('df')
    print(df)

    # Generate summary
    summary = generate_summary(df, method)
    
    print(f'summary {summary}')

    # Print tables
    print_tables(summary, method)
    
    print(f'plotyinh hyperparmeters')
    # Plot hyperparameter effects (for ES only)
    plot_es_hyperparameters(summary, method)
    
    # Plot training curves
    plot_training_curves(df, summary, method)

    # Best configuration analysis
    best_configs = []
    for dataset in summary['dataset'].unique():
        for model in summary['model'].unique():
            subset = summary[(summary['dataset'] == dataset) & (summary['model'] == model)]
            if not subset.empty:
                best_row = subset.loc[subset['final_val_accuracy_mean'].idxmax()]
                best_configs.append(best_row)

    print(f'best configs')
    print(best_configs)
    if best_configs:
        best_configs_df = pd.concat(best_configs, axis=1).T
        best_configs_df['method'] = method

        # Przygotuj etykiety hiperparametrów dla najlepszych konfiguracji
        param_cols = hyperparameters[method]
        def param_label(row):
            return ', '.join([f"{col}={row[col]}" for col in param_cols])

        best_configs_df['params'] = best_configs_df.apply(param_label, axis=1)

        # Wykres: Test Accuracy najlepszych konfiguracji (jeden wykres, każdy słupek to dataset+model)
                # Wykres: Test Accuracy najlepszych konfiguracji (jeden wykres, każdy słupek to dataset+model)
        # Wykres: Test Accuracy najlepszych konfiguracji (jeden wykres, każdy słupek to dataset+model)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=best_configs_df,
            x='params',
            y='test_accuracy_mean',
            hue='dataset',
            dodge=True,
            width=0.5,
            edgecolor='black'
        )
        plt.title(f'Best Test Accuracy per Dataset/Model ({method})')
        plt.xlabel('Best Hyperparameters')
        plt.ylabel('Test Accuracy [%]')
        plt.legend(title='Dataset', loc='best')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)  # OBRÓT etykiet na osi X o 90 stopni
        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', padding=2)
       
        plt.tight_layout()
        plt.savefig(f'experiments/plots/test_accuracy_best/test_accuracy_best_{method}.png')
        plt.close()

        # Wykres: Training Time najlepszych konfiguracji (jeden wykres, każdy słupek to dataset+model)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=best_configs_df,
            x='params',
            y='training_time_mean',
            hue='dataset',
            dodge=True,
            width=0.5,
            edgecolor='black'
        )
        plt.title(f'Best Training Time per Dataset/Model ({method})')
        plt.xlabel('Best Hyperparameters')
        plt.ylabel('Training Time [s]')
        plt.legend(title='Dataset', loc='best')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)  # OBRÓT etykiet na osi X o 90 stopni
        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', padding=2)
       
        plt.tight_layout()
        plt.savefig(f'experiments/plots/training_time_best/training_time_best_{method}.png')
        plt.close()

        return df
    else:
        print(f"Brak najlepszych konfiguracji dla metody: {method}")
        return df
# print(results)
# # Analyze each method separately
# for method in hyperparameters.keys():
#     print(f"\n=== Analyzing {method} ===")
#     analyze_method(method)

# print("\nAnalysis complete. Tables printed, plots saved as PNG files.")
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def process_data(results):
    # Przetworzenie danych do DataFrame
    data = []
    for key, value in results.items():
        print(key)
        dataset_name, model_name, variant, mu_str, lambda_str = key.split('_')
        mu = int(mu_str.replace('mu', ''))
        lambda_ = int(lambda_str.replace('lambda', ''))

        data_for_every_seed = {}
        for seed, model_history in value.items():
            history = model_history['history']
            training_time = model_history['training_time']

            print(training_time)
            history_list = []
            for i, gen in enumerate(history):
                # Wybieramy tylko samples z danych na temat historii trenowania

                history_list.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'variant': variant,
                    'mu': int(mu),
                    'lambda': int(lambda_),
                    'generation': gen['generation'],
                    'train_loss': gen['train_loss'],
                    'val_loss': gen['val_loss'],
                    'train_accuracy': gen['train_accuracy'],
                    'val_accuracy': gen['val_accuracy'],
                    'training_time': training_time,
                    'seed': seed
                })
            data_for_every_seed[seed] = history_list 

            data.append(data_for_every_seed)
                

    df = pd.DataFrame(data)

    return df

# Ustawienie stylu wykresów
sns.set(style="darkgrid")

# Wykresy trenowania na jednym dużym wykresie
def plot_training_history(df, samples=1):
    # Rozpakowanie zagnieżdżonych słowników
    unpacked_data = []
    for index, row in df.iterrows():
        for seed, history_list in row.items():
            if isinstance(history_list, list):  # Sprawdzenie, czy jest to lista historii
                for entry in history_list:
                    entry['seed'] = seed
                    unpacked_data.append(entry)
    unpacked_df = pd.DataFrame(unpacked_data)

    for seed in unpacked_df['seed'].unique()[:samples]:
        for dataset_name, dataset in unpacked_df[unpacked_df['seed'] == seed].groupby('dataset'):
            plt.figure(figsize=(15, 10))
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Analiza trenowania dla {dataset_name} (Seed: {seed})', fontsize=16)
            
            subset = dataset
            
            # Strata treningowa
            sns.lineplot(data=subset, x='generation', y='train_loss', hue='model', style='variant', ax=axes[0, 0])
            axes[0, 0].set_title(f'Strata treningowa (Seed: {seed})')
            axes[0, 0].set_xlabel('Generacja')
            axes[0, 0].set_ylabel('Strata')
            
            # Strata walidacyjna
            sns.lineplot(data=subset, x='generation', y='val_loss', hue='model', style='variant', ax=axes[0, 1])
            axes[0, 1].set_title(f'Strata walidacyjna (Seed: {seed})')
            axes[0, 1].set_xlabel('Generacja')
            axes[0, 1].set_ylabel('Strata')
            
            # Dokładność treningowa
            sns.lineplot(data=subset, x='generation', y='train_accuracy', hue='model', style='variant', ax=axes[1, 0])
            axes[1, 0].set_title(f'Dokładność treningowa (Seed: {seed})')
            axes[1, 0].set_xlabel('Generacja')
            axes[1, 0].set_ylabel('Dokładność (%)')
            
            # Dokładność walidacyjna
            sns.lineplot(data=subset, x='generation', y='val_accuracy', hue='model', style='variant', ax=axes[1, 1])
            axes[1, 1].set_title(f'Dokładność walidacyjna (Seed: {seed})')
            axes[1, 1].set_xlabel('Generacja')
            axes[1, 1].set_ylabel('Dokładność (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

# Wykresy dokładności na jednym dużym wykresie
def plot_accuracy_evolution(df, samples=1):
    # Rozpakowanie zagnieżdżonych słowników
    unpacked_data = []
    for index, row in df.iterrows():
        for seed, history_list in row.items():
            if isinstance(history_list, list):  # Sprawdzenie, czy jest to lista historii
                for entry in history_list:
                    entry['seed'] = seed
                    unpacked_data.append(entry)
    unpacked_df = pd.DataFrame(unpacked_data)

    for seed in unpacked_df['seed'].unique()[:samples]:
        for dataset_name, dataset in unpacked_df[unpacked_df['seed'] == seed].groupby('dataset'):
            plt.figure(figsize=(15, 5))
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f'Ewolucja accuracy dla {dataset_name} (Seed: {seed})', fontsize=16)
            
            subset = dataset
            
            # Dokładność treningowa
            sns.lineplot(data=subset, x='generation', y='train_accuracy', hue='model', style='variant', ax=axes[0])
            axes[0].set_title(f'Dokładność treningowa (Seed: {seed})')
            axes[0].set_xlabel('Generacja')
            axes[0].set_ylabel('Dokładność (%)')
            
            # Dokładność walidacyjna
            sns.lineplot(data=subset, x='generation', y='val_accuracy', hue='model', style='variant', ax=axes[1])
            axes[1].set_title(f'Dokładność walidacyjna (Seed: {seed})')
            axes[1].set_xlabel('Generacja')
            axes[1].set_ylabel('Dokładność (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

# Wykresy czasu trenowania
def plot_training_time(averages):
    # Rozpakowanie zagnieżdżonych słowników
    unpacked_data = []
    for index, row in averages.iterrows():
        for seed, history_list in row.items():
            if isinstance(history_list, list):  # Sprawdzenie, czy jest to lista historii
                for entry in history_list:
                    unpacked_data.append(entry)  # Metadane są już w entry
    unpacked_df = pd.DataFrame(unpacked_data)

    # Obliczenie średnich na płaskim DataFrame
    last_gen = unpacked_df.groupby(['dataset', 'model', 'variant', 'mu', 'lambda', 'seed']).last().reset_index()
    averages = last_gen.groupby(['dataset', 'model', 'variant', 'mu', 'lambda']).agg({
        'train_loss': 'mean',
        'val_loss': 'mean',
        'train_accuracy': 'mean',
        'val_accuracy': 'mean',
        'training_time': 'mean'
    }).reset_index()

    for dataset in averages['dataset'].unique():
        plt.figure(figsize=(12, 6))
        subset = averages[averages['dataset'] == dataset]
        sns.barplot(data=subset, x='model', y='training_time', hue='variant', dodge=True)
        plt.title(f'Czas trenowania dla {dataset}')
        plt.xlabel('Model')
        plt.ylabel('Czas trenowania (s)')
        plt.legend(title='Wariant ES')
        plt.show()
        

# Analiza wpływu zmiennych (mu i lambda) na dokładność walidacyjną
def plot_analysis_of_input(df):
    # Rozpakowanie zagnieżdżonych słowników
    unpacked_data = []
    for index, row in df.iterrows():
        for seed, history_list in row.items():
            if isinstance(history_list, list):  # Sprawdzenie, czy jest to lista historii
                for entry in history_list:
                    entry['seed'] = seed
                    unpacked_data.append(entry)
    unpacked_df = pd.DataFrame(unpacked_data)

    for dataset in unpacked_df['dataset'].unique():
        subset = unpacked_df[unpacked_df['dataset'] == dataset]
        
        # Wpływ mu
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset, x='mu', y='val_accuracy', hue='variant')
        plt.title(f'Wpływ mu na dokładność walidacyjną dla {dataset}')
        plt.xlabel('mu')
        plt.ylabel('Dokładność walidacyjna (%)')
        plt.show()
        
        # Wpływ lambda
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset, x='lambda', y='val_accuracy', hue='variant')
        plt.title(f'Wpływ lambda na dokładność walidacyjną dla {dataset}')
        plt.xlabel('lambda')
        plt.ylabel('Dokładność walidacyjna (%)')
        plt.show()

# Obliczenie średnich dla ostatniej generacji
def get_averages(df):
    # Rozpakowanie zagnieżdżonych słowników
    unpacked_data = []
    for index, row in df.iterrows():
        for seed, history_list in row.items():
            if isinstance(history_list, list):  # Sprawdzenie, czy jest to lista historii
                for entry in history_list:
                    entry['seed'] = seed
                    unpacked_data.append(entry)
    unpacked_df = pd.DataFrame(unpacked_data)

    last_gen = unpacked_df.groupby(['dataset', 'model', 'variant', 'mu', 'lambda', 'seed']).last().reset_index()
    averages = last_gen.groupby(['dataset', 'model', 'variant', 'mu', 'lambda']).agg({
        'train_loss': 'mean',
        'val_loss': 'mean',
        'train_accuracy': 'mean',
        'val_accuracy': 'mean',
        'training_time': 'mean'
    }).reset_index()
    return averages, last_gen


# # # Dodaj parser argumentów
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, default="experiments/results/history_all.json", help="Ścieżka do pliku z wynikami")
# args = parser.parse_args()

# # Wczytanie danych z pliku JSON
# with open(args.input, 'r') as f:
#     results = json.load(f)


# df = process_data(results)

# plot_training_history(df)

# averages, last_gen = get_averages(df)

# plot_training_time(averages)

# plot_analysis_of_input(df)

# #Wnioski (przykładowe, do dostosowania po analizie wykresów)
# print("### Wnioski z analizy:")
# print("- Wariant ES: Na podstawie wykresów można ocenić, który wariant (np. '+' lub ',') daje lepsze wyniki dla różnych zbiorów danych.")
# print("- Mu i Lambda: Wyższe wartości mu/lambda mogą zwiększać dokładność, ale kosztem dłuższego czasu trenowania.")
# print("- Modele: Głębsze sieci (DeepMLP) mogą osiągać wyższą dokładność w porównaniu do prostszych (MLP), ale wymagają więcej czasu.")




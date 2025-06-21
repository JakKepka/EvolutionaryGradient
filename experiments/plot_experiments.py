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
        history = value['history']
        training_time = value['training_time']

        print(training_time)
        for gen in history:
            data.append({
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
                'training_time': training_time
            })

    df = pd.DataFrame(data)

    return df

# Ustawienie stylu wykresów
sns.set(style="darkgrid")

# Wykresy trenowania na jednym dużym wykresie
def plot_training_history(df):
    for dataset in df['dataset'].unique():
        plt.figure(figsize=(15, 10))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analiza trenowania dla {dataset}', fontsize=16)
        
        subset = df[df['dataset'] == dataset]
        
        # Strata treningowa
        sns.lineplot(data=subset, x='generation', y='train_loss', hue='model', style='variant', ax=axes[0, 0])
        axes[0, 0].set_title('Strata treningowa')
        axes[0, 0].set_xlabel('Generacja')
        axes[0, 0].set_ylabel('Strata')
        
        # Strata walidacyjna
        sns.lineplot(data=subset, x='generation', y='val_loss', hue='model', style='variant', ax=axes[0, 1])
        axes[0, 1].set_title('Strata walidacyjna')
        axes[0, 1].set_xlabel('Generacja')
        axes[0, 1].set_ylabel('Strata')
        
        # Dokładność treningowa
        sns.lineplot(data=subset, x='generation', y='train_accuracy', hue='model', style='variant', ax=axes[1, 0])
        axes[1, 0].set_title('Dokładność treningowa')
        axes[1, 0].set_xlabel('Generacja')
        axes[1, 0].set_ylabel('Dokładność (%)')
        
        # Dokładność walidacyjna
        sns.lineplot(data=subset, x='generation', y='val_accuracy', hue='model', style='variant', ax=axes[1, 1])
        axes[1, 1].set_title('Dokładność walidacyjna')
        axes[1, 1].set_xlabel('Generacja')
        axes[1, 1].set_ylabel('Dokładność (%)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_training_time(averages):
    # Wykresy czasu trenowania
    for dataset in averages['dataset'].unique():
        plt.figure(figsize=(12, 6))
        subset = averages[averages['dataset'] == dataset]
        sns.barplot(data=subset, x='model', y='training_time', hue='variant', dodge=True)
        plt.title(f'Czas trenowania dla {dataset}')
        plt.xlabel('Model')
        plt.ylabel('Czas trenowania (s)')
        plt.legend(title='Wariant ES')
        plt.show()

def plot_analysis_of_input(df):
    # Analiza wpływu zmiennych (mu i lambda) na dokładność walidacyjną
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        
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

def get_averages(df):
    # Obliczenie średnich dla ostatniej generacji
    last_gen = df.groupby(['dataset', 'model', 'variant', 'mu', 'lambda']).last().reset_index()
    averages = last_gen.groupby(['dataset', 'model', 'variant', 'mu', 'lambda']).agg({
        'train_loss': 'mean',
        'val_loss': 'mean',
        'train_accuracy': 'mean',
        'val_accuracy': 'mean',
        'training_time': 'mean'
    }).reset_index()

    return averages, last_gen

# # Dodaj parser argumentów
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

# Wnioski (przykładowe, do dostosowania po analizie wykresów)
# print("### Wnioski z analizy:")
# print("- Wariant ES: Na podstawie wykresów można ocenić, który wariant (np. '+' lub ',') daje lepsze wyniki dla różnych zbiorów danych.")
# print("- Mu i Lambda: Wyższe wartości mu/lambda mogą zwiększać dokładność, ale kosztem dłuższego czasu trenowania.")
# print("- Modele: Głębsze sieci (DeepMLP) mogą osiągać wyższą dokładność w porównaniu do prostszych (MLP), ale wymagają więcej czasu.")




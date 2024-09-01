import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(data, target_column):
    sns.histplot(data[target_column], kde=True)
    plt.title(f'Distribution of {target_column}')
    plt.show()

def plot_correlation_matrix(data):
    # Filter the data to include only numeric columns
    numeric_data = data.select_dtypes(include=[float, int])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('../data/processed/processed_train.csv')
    plot_correlation_matrix(data)

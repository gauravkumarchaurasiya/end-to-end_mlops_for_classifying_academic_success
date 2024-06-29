import matplotlib.pyplot as plt

X_model_names = []
Y_model_accuracies = []

def update_metrics(names, accuracies):
    global X_model_names, Y_model_accuracies
    X_model_names = names
    Y_model_accuracies = accuracies

def plot_metrics():
    plt.figure(figsize=(10, 6))
    plt.bar(X_model_names, Y_model_accuracies, color='green')
    plt.ylim(0, 1)
    plt.title('Model Accuracies')
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Save the plot
    plot_path = 'src/visualization/metrics_plot.png'
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_metrics()

import matplotlib.pyplot as plt
# from metricsplot import model_accuracies

X_model_names = list()
Y_model_accuracies = list()

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.bar(X_model_names, Y_model_accuracies, color='green')
plt.ylim(0, 1)
plt.title('Model Accuracies')
plt.xlabel('Model Names')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Save the plot
plot_path = 'metrics/metrics_plot.png'
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

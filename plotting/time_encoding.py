import matplotlib.pyplot as plt
import numpy as np

def plot_classification_times():
    # Data
    models = ['VotingEnsemble', 'BaggedNN', 'RandomForest']
    encoded_times = [0.000041, 0.000032, 0.000011]
    not_encoded_times = [0.000035, 0.000033, 0.000003]

    # Colors
    light_red = "#FFE4E4"
    dark_red = "#BD8686"
    light_blue = "#E2FFFF"
    dark_blue = "#91C4C4"

    # Bar chart settings
    x = np.arange(len(models))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, encoded_times, width, label='Encoded',
                   color=light_blue, edgecolor=dark_blue)
    bars2 = ax.bar(x + width/2, not_encoded_times, width, label='Not Encoded',
                   color=light_red, edgecolor=dark_red)

    # Labels and title
    ax.set_ylabel('Average Classification Time (seconds)')
    ax.set_title('Average Time per Record: Encoded vs Not Encoded')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add value labels on top of the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

# Call the function
plot_classification_times()

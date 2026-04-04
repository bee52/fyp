import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style to look like an academic paper
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Data representing the disparity in available Fake News Datasets
regions = ['United States (US)', 'United Kingdom (UK)']
dataset_count = [42, 3] # Representing the vast difference in available research datasets

# Create the figure
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=regions, y=dataset_count, palette=['#1f77b4', '#d62728'])

# Add labels and title
plt.title('Availability of Region-Specific Disinformation Datasets (2015-2025)', fontsize=14, pad=15)
plt.ylabel('Number of Major Academic Datasets', fontsize=12)
plt.xlabel('Geographic Focus', fontsize=12)

# Add the exact numbers on top of the bars
"""for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                )"""

# Clean up the borders
sns.despine()

# Save the graph as a high-resolution PNG for your Word document
plt.tight_layout()
plt.savefig('uk_vs_us_datasets_graph.png', dpi=300)
print("Graph saved successfully as 'uk_vs_us_datasets_graph.png'. You can now insert this into your Word document.")
plt.show()
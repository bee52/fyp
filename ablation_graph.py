import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set academic visual style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# The Data: Simulating the F1-Scores for the ablation study
categories = ['Overall Performance', 'Standard News', 'Sarcastic/Adversarial']
roberta_only = [0.85, 0.92, 0.61] # RoBERTa struggles with sarcasm
dual_branch = [0.93, 0.94, 0.89]  # Dual-Branch fixes the sarcasm blindspot

x = np.arange(len(categories))
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(9, 6))

# Create the grouped bars
rects1 = ax.bar(x - width/2, roberta_only, width, label='Branch A Only (RoBERTa)', color='#7f8c8d')
rects2 = ax.bar(x + width/2, dual_branch, width, label='Dual-Branch (Hybrid)', color='#c0392b')

# Add labels, title, and formatting
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Expected Ablation Study Results: F1-Score Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax.set_ylim(0, 1.1)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Clean up borders
sns.despine()
plt.tight_layout()

# Save the high-res image for the Word document
plt.savefig('ablation_study_projection.png', dpi=300)
print("Success! Graph saved as 'ablation_study_projection.png'")
plt.show()
# presentation_image.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the style for a professional presentation
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create figure
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# Data from your screenshots
class_names = ['Negative', 'Neutral', 'Positive']

# Training progress data
epochs = [1, 2, 3, 4]
train_acc = [0.8594, 0.9071, 0.9403, 0.9642]
train_loss = [0.5450, 0.4316, 0.0360, 0.1308]

# Confusion matrix from your screenshot
conf_matrix = np.array([
    [876, 109, 50],
    [90, 154, 106],
    [45, 125, 2844]
])

# Classification metrics from your screenshot
precision = [0.87, 0.40, 0.95]
recall = [0.85, 0.44, 0.94]
f1_score = [0.86, 0.42, 0.95]
support = [1035, 350, 3014]

# 1. Training Progress
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_acc, 'o-', color='blue', label='Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training Progress')
ax1.set_xticks(epochs)
ax1.set_ylim(0.8, 1.0)
ax1.grid(True)

# Add loss on secondary y-axis
ax1_twin = ax1.twinx()
ax1_twin.plot(epochs, train_loss, 'o-', color='red', label='Loss')
ax1_twin.set_ylabel('Loss')
ax1_twin.set_ylim(0, 0.6)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# 2. Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax2)
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# 3. Per-Class Metrics
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(class_names))
width = 0.25

ax3.bar(x - width, precision, width, label='Precision')
ax3.bar(x, recall, width, label='Recall')
ax3.bar(x + width, f1_score, width, label='F1 Score')

ax3.set_xticks(x)
ax3.set_xticklabels(class_names)
ax3.set_ylim(0, 1.0)
ax3.set_title('Per-Class Performance Metrics')
ax3.set_ylabel('Score')
ax3.legend()

# 4. Class Distribution
ax4 = fig.add_subplot(gs[1, 1])
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax4.pie(support, labels=class_names, autopct='%1.1f%%', colors=colors, 
        startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
ax4.set_title('Class Distribution in Test Dataset')

# Add overall accuracy as text annotation
fig.text(0.5, 0.02, f'Overall Model Accuracy: 88.07%', 
         ha='center', fontsize=16, fontweight='bold')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('sentiment_analysis_presentation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Presentation image saved as 'sentiment_analysis_presentation.png'")
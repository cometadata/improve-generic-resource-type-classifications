import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load the accuracy data
with open('output/icl/datacite_2024_classified_qwen3_8b_accuracy.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame for easier manipulation
rows = []
for true_label, predictions in data.items():
    for pred_label, count in predictions.items():
        rows.append({'true': true_label, 'predicted': pred_label, 'count': count})

df = pd.DataFrame(rows)

# Create confusion matrix
confusion_matrix = df.pivot(index='true', columns='predicted', values='count').fillna(0)

# Normalize by row (true label totals)
confusion_matrix_norm = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

# Create heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(confusion_matrix_norm, 
            annot=False, 
            cmap='Blues', 
            cbar_kws={'label': 'Normalized count'})
plt.title('Confusion matrix (row-normalized)')
plt.xlabel('Predicted')
plt.ylabel('True label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('output/icl/datacite_2024_classified_qwen3_8b_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
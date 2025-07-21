import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python create_heatmap.py <input_json_file>")
    sys.exit(1)

input_file = sys.argv[1]
input_dir = os.path.dirname(input_file)
base_name = os.path.splitext(os.path.basename(input_file))[0]

# Load the accuracy data
with open(input_file, 'r') as f:
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
output_path = os.path.join(input_dir, f'{base_name}_confusion_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {output_path}")
plt.show()

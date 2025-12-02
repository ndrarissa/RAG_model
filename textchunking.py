# Load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

data = pd.read_csv('MedQuAD_Dataset_RAG_Scenario.csv')

# Text chunking

data['chunk'] = data.apply(
    lambda row: f"{row['question']} {row['answer']} {row['focus_area']} {row['source']}",
    axis=1
)
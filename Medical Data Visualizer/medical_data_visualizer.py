import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, 
                     id_vars='cardio', 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 # 7
    df_cat = pd.DataFrame({'total':df_cat.groupby(['cardio', 'variable'])['value'].value_counts()})\
                                     .rename(columns={'cardio':'Cardio','variable':'Variable', 'value':'Value'})\
                                     .reset_index()  

    # 8
    fig = sns.catplot(x='variable', y='total', col='cardio', data=df_cat, kind='bar', hue='value', height=5, aspect=1)

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', ax=ax, cbar_kws={'shrink': .8}, linewidths=0.5)

    # 16
    fig.savefig('heatmap.png')
    return fig

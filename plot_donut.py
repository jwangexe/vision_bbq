import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects

# Load data into DataFrame
df = pd.read_csv("./bias_class_counts.csv")
df = df[df["bias_class"] != "Nationality"]

# Replace underscores with spaces and title case
df['bias_class'] = df['bias_class'].str.replace('_', ' ').str.title()
df.loc[df["bias_class"] == "Ses", 'bias_class'] = "SES"

# Calculate synthetic ratio and category percentage
df['syn_ratio'] = df['num_syn'] / df['num_total']
df['category_percent'] = df['num_total'] / df['num_total'].sum() * 100

all_data = df[df['bias_class'] == 'All']
df = df[df['bias_class'] != 'All']  # Remove summary row

# Create custom red-to-blue colormap
colors = ["#3498db", "#e74c3c"]  # Blue to red
cmap = LinearSegmentedColormap.from_list("rb_cmap", colors)

# Create figure with optimized layout
fig, ax = plt.subplots(figsize=(13, 12))

# Set main title
plt.title("Image Distribution By Bias Category", 
          fontsize=20, pad=20, weight='bold')

# Create outer donut chart
wedges, texts = ax.pie(
    df['num_total'],
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5),
    startangle=90,
    colors=[cmap(ratio) for ratio in df['syn_ratio']],
    labels=[f"{row['bias_class']}\n{row['syn_ratio']*100:.0f}%" 
            for _, row in df.iterrows()],
    labeldistance=0.75,  # Increased distance from center
    textprops=dict(
        color='white',
        fontsize=11,
        weight='bold',
        ha='center',
        va='center',
        path_effects=[patheffects.withStroke(linewidth=3, foreground='black')]
    )
)

# Add center circle to create donut appearance
centre_circle = plt.Circle((0, 0), 0.3, color='white', fc='white', linewidth=0)
ax.add_artist(centre_circle)

# Add center sub-ring for 'all' category
#all_data = df[df['bias_class'] == 'All']
if not all_data.empty:
    all_value = all_data['num_total'].values[0]
    all_color = cmap(all_data['syn_ratio'].values[0])
    ax.pie(
        [all_value],
        radius=0.5,
        colors=[all_color],
        wedgeprops=dict(width=0.2, edgecolor='white', linewidth=1.5),
        startangle=90,
        labels=[f'All\n{all_data["syn_ratio"].values[0]*100:.0f}%'],
        labeldistance=0.75,
        textprops=dict(
            color='white',
            fontsize=11,
            weight='bold',
            ha='center',
            va='center',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black')]
        )
    )

# # Create custom legend moved to the left with small margin
# legend_elements = [
#     plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], 
#                markersize=12, label='100% Searched'),
#     plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], 
#                markersize=12, label='100% AI-Generated'),
#     plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#9b59b6', 
#                markersize=12, label='50% AI / 50% Search')
# ]

# ax.legend(
#     handles=legend_elements,
#     loc='center left',
#     bbox_to_anchor=(-0.25, 0.5),  # Slightly adjusted position to move left
#     fontsize=11,
#     title="Color Meaning:",
#     title_fontsize=12,
#     frameon=False
# )

# Add colorbar with clear labeling
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label('AI-Generated Ratio', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
cbar.ax.text(0.5, -0.01, "More Searched", ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
cbar.ax.text(0.5, 1.01, "More AI-Generated", ha='center', va='bottom', fontsize=9, transform=cbar.ax.transAxes)

# Add data source footnote
plt.figtext(0.5, 0.01, "Data source: vBBQ | Segment size proportional to total images in category", 
            ha='center', fontsize=10, color='#777777')

# Adjust layout to prevent clipping
fig.tight_layout(pad=0.1)

# Save the figure with minimal padding
plt.savefig('bias_donut_chart_with_center_ring.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()

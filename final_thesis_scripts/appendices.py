## PLOTTING ROP ON EACH WELL WITH SECTION THRESHOLD --------------------------------------------
# Thresholds per well with section names
section_thresholds = {
    'A-3': [(37, '26"'), (523, '17 1/2"'), (1650, '12 1/4"'), (2641, '8 1/2"')],
    'A-4': [(49, '26"'), (531, '17 1/2"'), (1650, '12 1/4"'), (2726, '8 1/2"')],
    'A-5': [(49, '26"'), (536, '17 1/2"'), (1619, '12 1/4"'), (2638, '8 1/2"')],
}

# Filter data for selected wells
selected_wells = ['A-3', 'A-4', 'A-5']
df_filtered = df[df['WELL_ID'].isin(selected_wells)]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 15), sharey=True, squeeze=False)

for i, well in enumerate(selected_wells):
    ax = axes[0, i]
    well_df = df_filtered[df_filtered['WELL_ID'] == well]

    # Plot ROP vs depth
    ax.plot(well_df['ROP'], well_df['MDEPTH'], color='b')
    ax.set_xlabel('ROP (m/hr)')
    ax.set_title(f'Well {well}', fontweight='bold')
    ax.set_ylim(0, 2800)
    ax.grid(True)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_major_locator(MultipleLocator(300))

    # Plot section thresholds and labels
    for depth, label in section_thresholds[well]:
        ax.axhline(y=depth, color='red', linestyle='--', linewidth=2)
        ax.text(
            0.99, depth+45, label,
            transform=ax.get_yaxis_transform(),
            ha='right', va='bottom',
            fontsize=11, color='red', fontweight='bold'
        )

# Shared y-axis label
fig.supylabel('Measured Depth (m)', fontsize=12)
plt.tight_layout()
plt.show()

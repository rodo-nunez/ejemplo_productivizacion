
# Section ---------------------------------------- 

def plot_groupby_target(data, groupby_col, colors=None):
    group_data = data.groupby([groupby_col, 'target'], observed=True)['customer_id'].count().unstack(fill_value=0)

    ind = np.arange(len(group_data))
    width = 0.25

    if colors is None:
        colors = ['g', 'violet', 'blue', 'orange', 'red', 'purple', 'cyan', 'brown']

    plt.figure(figsize=(12, 7))

    for i, target_val in enumerate(group_data.columns):
        label = 'No abandonó' if target_val == 0 else 'Abandonó'
        plt.bar(ind + i * width, group_data[target_val], width, color=colors[i % len(colors)], label=label)

    plt.xlabel('Categoría')
    plt.ylabel('Cantidad de clientes')
    plt.title(f'{groupby_col.capitalize()} vs Abandono')
    plt.xticks(ind + width, group_data.index.tolist())
    plt.legend()
    plt.show()
    
    
plot_groupby_target(data, 'gender')
print(data.groupby(['gender', 'target'], observed=True)['customer_id'].count().unstack(fill_value=0))

plot_groupby_target(data, 'contract_type')
print(data.groupby(['contract_type', 'target'], observed=True)['customer_id'].count().unstack(fill_value=0))

plot_groupby_target(data, 'payment_method')
print(data.groupby(['payment_method', 'target'], observed=True)['customer_id'].count().unstack(fill_value=0))

plot_groupby_target(data, 'internet_service')
print(data.groupby(['internet_service', 'contract_type'], observed=True)['customer_id'].count().unstack(fill_value=0))

plot_groupby_target(data, 'senior_citizen')
print(data.groupby(['senior_citizen', 'target'], observed=True)['customer_id'].count().unstack(fill_value=0))

col_stats = ['monthly_charges', 'total_charges', 'target', 'senior_citizen']
print(data[col_stats].corr())
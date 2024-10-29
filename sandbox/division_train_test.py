
# División entre train y test ---------------------------------------- 

valid_set = data[data['begin_date'] >= reference_date]
train_test_set = data[data['begin_date'] < reference_date]

train_set, test_set = train_test_split(train_test_set, test_size=0.25, random_state=12345, stratify=train_test_set['target'])

columns_to_drop = ['customer_id', 'begin_date', 'end_date']

train_features = train_set.drop(columns=columns_to_drop + ['target'])
train_target = train_set['target']

valid_features = valid_set.drop(columns=columns_to_drop + ['target'])
valid_target = valid_set['target']

test_features = test_set.drop(columns=columns_to_drop + ['target'])
test_target = test_set['target']


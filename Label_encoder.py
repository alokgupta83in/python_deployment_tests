

#### Trainig time
def process_categorical_data_train(df_train, null_dict, unknown_dict):
    df = df_train.copy()
    le_dict = defaultdict(LabelEncoder)
    #df = df_raw[['SUBZIP', 'SUBGROUPCODE', 'RELCODE', 'GROUPCODE', 'COBBATCHID', 'WORKINGSTATUS', 'WFID', 'SUBGROUPNAME', 'SUBCITY', 'FUNDINGTYPE', 'BUSINESS_ID', 'SUBGENDER', 'SUBFIRSTNAME', 'FEEDBACKCD', 'OICARRIERCODE', 'LOB', 'SUBSTATE', 'CONTRACTNUMBER', 'DIVISIONCODE', 'AUTHNO', 'UHGPrimacy']].copy()
    
    for col in df.columns:
        df[col] = df[col].fillna(null_dict[col])
    
    # add unknown data so we can get those classes in label encoder
    df.loc[df.shape[0]] = list(unknown_dict.values())
    df.loc[df.shape[0]] = list(null_dict.values())
    
    # Fit the label encoder with unknown and Null data classes which are not part of legacy data classes .
    df = df.apply(lambda x: le_dict[x.name].fit_transform(x.astype(str)))
    
    # drop the unknown data row
    df = df.drop([df.shape[0]-1])
    df = df.drop([df.shape[0]-1])
    
    # save the label encoder dict in pkl format
    with open('../../model_objects/label_encoder_dictionary.pkl', 'wb') as f:
        pickle.dump(le_dict, f, pickle.HIGHEST_PROTOCOL)
    
    return df



for key in cat_cols:
    null_dict[key] = "NULL_" + key
    unknown_dict[key] = "UNKNOWN_" + key


df_categorical = reuse_model.process_categorical_data_train(df_categorical, null_dict, unknown_dict)



##### testing time

def process_data_test(df_raw, null_dict, unknown_dict):
    df = df_raw.copy()
    
    # Load the encoder saved during training time.
    le_dict = load_classifier_scaler('label_encoder_dictionary')
    
    for col in df.columns:
        df[col] = df[col].fillna(null_dict[col])
    
    # Change unseen values to unknown classes    
    for col in df.columns:
        le_classes = set(le_dict[col].classes_)
        col_classes = set(df[col].value_counts().index)
        new_classes = list(col_classes.difference(le_classes))
        if new_classes:
            df[col] = df[col].replace(new_classes, unknown_dict[col])
    
    # Encode the data using existing label encoder.
    df = df.apply(lambda x: le_dict[x.name].transform(x.astype(str)))
    
    return df




null_dict = {}
unknown_dict = {}

for key in categorical_list:
    null_dict[key] = "NULL_" + key
    unknown_dict[key] = "UNKNOWN_" + key


df_categorical = reuse_model.process_data_test(df_categorical, null_dict, unknown_dict)


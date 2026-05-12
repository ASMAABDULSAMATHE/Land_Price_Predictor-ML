#Model training
def train_model(X_train, y_train):


    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)

    #Converting model features to list
    features_list=model.feature_names_in_.tolist()
    print(features_list)

    return model
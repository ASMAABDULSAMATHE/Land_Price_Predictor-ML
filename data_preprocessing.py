#Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess_data():

    df=pd.read_csv("land_prices.csv")

    #Calculating price per cent
    df["price_per_cent"]=df["price_lakhs"]/df["land_area_cents"]
    print(df.columns)

    #Feature Engineering-One hot encoding
    numeric_features=['land_area_cents', 'distance_to_school_km',
        'distance_to_airport_km', 'distance_to_railway_station_km',
        'distance_to_hospital_km', 'distance_to_medical_college_km',
        'distance_to_bus_stop_km', 'distance_to_market_km']
    df_processed=pd.get_dummies(df,columns=['location_name','taluk','village','land_type'],drop_first=True)
    df_processed.head()

    #Extracting One hot encode features
    dummy_features=[]
    for column_name in df_processed.columns:
        if column_name.startswith('land_type') or column_name.startswith('village') or column_name.startswith('location_name') or column_name.startswith('taluk'):
            dummy_features.append(column_name)
    print(dummy_features)

    #Final Features with numeric and encoded categoric features
    features=numeric_features+dummy_features
    print(features)

    X=df_processed[features]
    y=df_processed['price_per_cent']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

    return X_train,X_test,y_train,y_test,df_processed
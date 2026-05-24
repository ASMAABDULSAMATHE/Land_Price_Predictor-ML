import pandas as pd
from data_preprocessing import preprocess_data
from train_model import train_model
from predict import evaluate
from visualise import plot_results,plot_feature_importance
from test import test_model
import joblib

def main():

    #1.Preprocessing
    X_train, X_test, y_train, y_test, df = preprocess_data()

    #2.Training
    model = train_model(X_train, y_train)

    #3.Prediction on test set
    y_pred, df_comparison = evaluate(model, X_test, y_test)

    #4.Visualisation
    plot_results(df, y_test, y_pred)

    #Feature Importance
    plot_feature_importance(model, X_train)

    #5.Save model 
    joblib.dump(model, "models/rf_model.pkl")

    print("Pipeline completed successfully!")


    #Model Testing
    new_land_property={
        'land_area_cents': [10.5],
        'distance_to_school_km': [2.0],
        'distance_to_airport_km': [25.0],
        'distance_to_railway_station_km': [7.0],
        'distance_to_hospital_km': [3.0],
        'distance_to_medical_college_km': [3.0],
        'distance_to_bus_stop_km': [0.5],
        'distance_to_market_km': [1.5],
        'location_name_Beypore': [0],
        'location_name_Chevayur': [0],
        'location_name_Elathur': [0],
        'location_name_Feroke': [0],
        'location_name_Kakkodi': [0],
        'location_name_Karaparamba': [0],
        'location_name_Koduvally': [1],
        'location_name_Koyilandy': [0],
        'location_name_Kunnamangalam': [0],
        'location_name_Mavoor': [0],
        'location_name_Medical College Area': [0],
        'location_name_Olavanna': [0],
        'location_name_Pantheerankavu': [0],
        'location_name_Peruvayal': [0],
        'location_name_Ramanattukara': [0],
        'location_name_Thamarassery': [0],
        'location_name_Vatakara':[0],
        'taluk_Kozhikode': [1],
        'taluk_Thamarassery': [0],
        'taluk_Vatakara': [0],

        'village_Beypore': [0],
        'village_Chevayur': [0],
        'village_Elathur': [0],
        'village_Feroke': [0],
        'village_Kakkodi': [0],
        'village_Karaparamba': [0],
        'village_Koduvally': [0],
        'village_Koyilandy': [0],
        'village_Kozhikode': [1],
        'village_Kunnamangalam': [0],
        'village_Mavoor': [0],
        'village_Olavanna': [0],
        'village_Pantheerankavu': [0],
        'village_Peruvayal': [0],
        'village_Ramanattukara': [0],
        'village_Thamarassery': [0],
        'village_Vatakara': [0],

        'land_type_Commercial': [0],
        'land_type_Residential': [1]
    }
    test_model(model,new_land_property)

if __name__ == "__main__":
    main()
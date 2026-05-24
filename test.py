import pandas as pd
def test_model(model,new_land_property):
    #Model testing
    new_land_df=pd.DataFrame(new_land_property)
    new_land_df = new_land_df[model.feature_names_in_]
    new_land_predicted_value=model.predict(new_land_df)
    print("Predicted price:",new_land_predicted_value)

    return new_land_predicted_value
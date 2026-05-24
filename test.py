import pandas as pd
def test_model(model,new_land_property):
    new_land_df=pd.DataFrame(new_land_property)
    new_land_predicted_value=model.predict(new_land_df)
    print("Predicted price:",new_land_predicted_value)

    return new_land_predicted_value
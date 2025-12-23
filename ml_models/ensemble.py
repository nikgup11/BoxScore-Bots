
print("--- Importing Random Forest ---")
import random_forest 
rf_pred = random_forest.y_pred_pts[0] # Extract the float value


print("--- Importing Linear Regression ---")
import linear_regression
lr_pred = linear_regression.get_model_and_prediction('gilgesh01_Shai_Gilgeous-Alexander_last3.csv')

# 3. Create the Ensemble (Soft Voting)
# We average the two predictions
ensemble_pred = (rf_pred + lr_pred) / 2

print("\n==============================")
print("   ENSEMBLE PREDICTION")
print("==============================")
print(f"Random Forest:    {rf_pred:.2f}")
print(f"Linear Reg (NN):  {lr_pred:.2f}")
print("------------------------------")
print(f"FINAL PREDICTION: {ensemble_pred:.2f} pts")
print("==============================")

# Optional: Add simple logic for betting lines
line = 32  # Example betting line
if ensemble_pred > line:
    print(f"Recommendation: OVER {line}")
else:
    print(f"Recommendation: UNDER {line}")
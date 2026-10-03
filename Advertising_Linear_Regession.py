import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Display Border
# ------------------------------------------------------------

def Border():
    print("-" * 70)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def LoadData(filename):
    Data = pd.read_csv(filename)
    return Data

# ------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------

def PrepareData(Data):
    Features = [
        "TV",
        "radio",
        "newspaper"
    ]

    X = Data[Features]
    Y = Data["sales"]

    return X, Y

# ------------------------------------------------------------
# Split Dataset
# ------------------------------------------------------------

def SplitData(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------
# Train Linear Regression Model
# ------------------------------------------------------------

def TrainModel(X_train, Y_train):
    Model = LinearRegression()
    Model.fit(X_train, Y_train)
    return Model

# ------------------------------------------------------------
# Display Predictions
# ------------------------------------------------------------

def DisplayPredictions(Model, X_test, Y_test):
    Y_pred = Model.predict(X_test)

    Result = pd.DataFrame({
        "Expected Sales": Y_test.values,
        "Predicted Sales": Y_pred
    })

    Border()
    print("Predicted and Expected Sales")
    Border()
    print(Result.to_string(index=False))

    return Y_pred

# ------------------------------------------------------------
# Display Model Details
# ------------------------------------------------------------

def DisplayModelDetails(Model):
    Border()
    print("Linear Regression Model")
    Border()

    print("Intercept :", Model.intercept_)
    print("TV Coefficient :", Model.coef_[0])
    print("Radio Coefficient :", Model.coef_[1])
    print("Newspaper Coefficient :", Model.coef_[2])

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    Border()
    print("Advertising Sales Prediction using Linear Regression")
    Border()

    Data = LoadData("Advertising.csv")

    print("Dataset Loaded Successfully")
    print("Dataset Shape :", Data.shape)

    X, Y = PrepareData(Data)

    print("Data Prepared Successfully")
    print("Features :", list(X.columns))
    print("Target : sales")

    X_train, X_test, Y_train, Y_test = SplitData(X, Y)

    Border()
    print("Training Data")
    Border()
    print("Training Samples :", len(X_train))

    Border()
    print("Testing Data")
    Border()
    print("Testing Samples :", len(X_test))

    Model = TrainModel(X_train, Y_train)

    Border()
    print("Linear Regression Model Trained Successfully")
    Border()

    DisplayModelDetails(Model)
    DisplayPredictions(Model, X_test, Y_test)

    Border()
    print("Assignment 46 Completed Successfully")
    Border()

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

def get_model():
    # Import data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Cleaning dataset
    train_df['Levy'] = train_df['Levy'].str.replace('-','0')
    train_df['Levy'] = train_df['Levy'].astype(int)
    train_df = train_df[train_df['Levy'] !='0']
    train_df['Mileage'] = train_df['Mileage'].str.replace('km','')
    train_df['Mileage'] = train_df['Mileage'].astype(int)
    train_df.drop(['ID', "Doors"], axis=1, inplace=True)
    train_df = train_df[train_df['Price']<=400000]
    train_df.drop_duplicates(inplace=True)

    # Make everything numeric
    categories = train_df.select_dtypes(include='O')
    encoder = LabelEncoder()
    encode = list(categories)
    train_df[encode] = train_df[encode].apply(lambda col: encoder.fit_transform(col))
    train_df[encode]

    # Set up model
    target = train_df.Price
    features = train_df.drop('Price', axis=1)
    scaler = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 40)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model = ExtraTreesRegressor(n_estimators = 70, max_depth = 20)
    model.fit(x_train, y_train)
    #y_predict = model.predict(x_train)

    # print('train score:', model.score(x_train, y_train))
    # print('test score:', model.score(x_test, y_test))

    #testing
    my_test = {'Manufacturer': [16], 'Model': [629], 'Prod year.': [2012],
            'Category':[4], 'Leather interior': [1], 'Fuel type': [2],
            'Engine volume': [46], 'Mileage': [143000], 'Cylinders': [4],
            'Gear box type': [0], 'Drive wheels': [0], 'Wheel': [0], 
            'Color': [14], 'Airbags': [12], 'Price': [0]}
    my_test = pd.DataFrame(my_test)

    new_predict = model.predict(my_test)
    print("PRICE! ", new_predict[0])
    return model


model = get_model()

my_test = {'Manufacturer': [16], 'Model': [629], 'Prod year.': [2012],
        'Category':[4], 'Leather interior': [1], 'Fuel type': [2],
        'Engine volume': [46], 'Mileage': [143000], 'Cylinders': [4],
        'Gear box type': [0], 'Drive wheels': [0], 'Wheel': [0], 
        'Color': [14], 'Airbags': [12], 'Price': [0]}
my_test = pd.DataFrame(my_test)

new_predict = model.predict(my_test)

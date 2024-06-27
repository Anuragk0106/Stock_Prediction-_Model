from flask import Flask, render_template, request
from stock_prediction_model import StockPredictionModel


app = Flask(__name__)

# Load TensorFlow model
model = StockPredictionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_choice = request.form['stock']
    if stock_choice == '1':
        train_data_file = 'CSV_Files/Gas Data Last Year.csv'
        test_data_file = 'CSV_Files/Gas Data Last Month.csv'
    elif stock_choice == '2':
        train_data_file = 'CSV_Files/Gold Data Last Year.csv'
        test_data_file = 'CSV_Files/Gold Data Last Month.csv'
    elif stock_choice == '3':
        train_data_file = 'CSV_Files/Oil Data Last Year.csv'
        test_data_file = 'CSV_Files/Oil Data Last Month.csv'
    elif stock_choice == '4':
        train_data_file = 'CSV_Files/Silver Data Last Year.csv'
        test_data_file = 'CSV_Files/Silver Data Last Month.csv'
    else:
        return "Invalid choice"

    accuracy = model.train_and_predict(train_data_file, test_data_file)

    return render_template('results.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)

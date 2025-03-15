# Car Price Prediction Model

This project implements a **Car Price Prediction Model** using linear regression with gradient descent. The dataset used is `Cleaned_CarPrice_Assignment.csv`, and the model predicts car prices based on various features after applying **Z-score normalization**.

## Features
- The dataset undergoes preprocessing, including dropping unnecessary columns and one-hot encoding categorical variables.
- Feature normalization is applied using **Z-score normalization**.
- **Gradient Descent** is implemented from scratch to optimize the cost function.
- The model is trained for **10,000 iterations** with a learning rate (`alpha`) of **0.2**.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib word2number
```

## Project Structure
- `Cleaned_CarPrice_Assignment.csv` - The dataset used for training.
- `main.py` - The script containing data preprocessing, training, and prediction.

## How It Works
1. **Load Data**: Reads the CSV file and extracts features (`X_train`) and target values (`y_train`).
2. **Preprocessing**:
   - Drops irrelevant columns.
   - Encodes categorical variables using one-hot encoding.
   - Converts the dataset into a NumPy array.
3. **Normalization**: Applies **Z-score normalization** to standardize feature values.
4. **Gradient Descent**:
   - Computes the cost function.
   - Computes the gradient for optimization.
   - Updates weights iteratively to minimize error.
5. **Prediction**: Uses the trained model to predict car prices.

## How to Run
Run the following command in your terminal:
```bash
python car_price_prediction_model.py
```
The script will train the model and print optimized values of `w` and `b`, along with a sample price prediction.

## Expected Output values of w , b and cost after 10000 iterations
```
The value of w found is : [ 3.46400527e+02 -8.17381202e+02  1.67836357e+03  9.77381746e+02
  8.39782965e+02  4.83486671e+03 -1.78119213e+03  2.63467245e+03
 -8.88508456e+02  1.32110507e+03  2.15361432e+00  3.01582779e+02
 -9.66956698e+02 -2.92973256e+03 -2.58824068e+03 -2.21923688e+03
  8.08151563e+01  1.31609565e+03 -1.80695713e+02  3.05488863e+02
  1.34372622e+03  1.68278565e+03 -6.85438774e+02  1.33621337e+03
  2.30760993e+02  3.91447356e+01 -2.15361432e+00 -1.85750132e+02
  3.17913178e+01 -3.86316253e+02  4.73739785e+01] and the value of b foudn is 13276.71057073171

Cost: 3235224.57104417
```

## Customization
- Modify `alpha` (learning rate) and `iterations` to experiment with different optimization speeds.
- Add/remove features in `df.drop()` for better model tuning.

## Contact
For any queries, feel free to open an issue or contribute to this project!

Happy Coding! 🚀


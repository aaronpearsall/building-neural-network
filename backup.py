def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weights_1, bias)

target = 1

mse = np.square(prediction - target)

print(f'The prediction result is: {prediction}; Error: {mse}')

derivative = 2 * (prediction - target)

print(f'The derivative is: {derivative}')

weights_1 = weights_1 - derivative

prediction = make_prediction(input_vector, weights_1, bias)

error = np.square(prediction - target)

print(f'Prediction: {prediction}; Error: {error}')
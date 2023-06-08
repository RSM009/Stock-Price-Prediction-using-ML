def calculate_max_pain(options_data):
    strike_prices = options_data.keys()
    max_pain = min(strike_prices, key=lambda x: abs(sum(options_data[x])))
    return max_pain


options_data = {
    100: [10, -5, 20],  # Strike price 100, open interest: [10, -5, 20]
    110: [-2, 8, 15],   # Strike price 110, open interest: [-2, 8, 15]
    120: [5, 10, -10]   # Strike price 120, open interest: [5, 10, -10]
}

max_pain_price = calculate_max_pain(options_data)
print("Max Pain Price:", max_pain_price)





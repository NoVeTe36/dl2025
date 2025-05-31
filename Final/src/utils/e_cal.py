def factorial(n):
    if n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

def calculate_e(iterations):
    e_approx = 0
    for i in range(iterations):
        e_approx += 1 / factorial(i)
    return e_approx

def return_e():
    iterations = 100
    e_value = calculate_e(iterations)
    return round(e_value, 6)
function solve_quad(a, b, c)
    x_plus = (-b + sqrt(b^2 - 4 * a * c))/2 * a
    x_minus = (-b - sqrt(b^2 - 4 * a * c))/2 * a
    imag(x_plus) == 0 && imag(x_minus) == 0 || error("Expected only real values.")
    n = max(x_plus, x_minus)
    isinteger(n) || error("Expected an integer value, got $(n)")
    Int(n)
end

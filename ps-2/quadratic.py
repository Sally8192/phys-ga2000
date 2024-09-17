import numpy as np

def quadratic_ab(a, b, c, method_b=False):
    """
    Compute the roots of the quadratic equation ax^2 + bx + c = 0 using
    either the standard or alternative formula.
    """
    discriminant = b**2 - 4*a*c
    sqrt_discriminant = np.sqrt(discriminant)
    
    if method_b:
        # Alternative formula
        x1 = (2 * c) / (-b - sqrt_discriminant)
        x2 = (2 * c) / (-b + sqrt_discriminant)
    else:
        # Standard formula
        x1 = (-b + sqrt_discriminant) / (2 * a)
        x2 = (-b - sqrt_discriminant) / (2 * a)
    
    return x1, x2

def quadratic(a,b,c):
    if b>0:
        r1 = (2*c)/(-b - np.sqrt(b**2 - 4*a*c))
        r2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    else:
        r1 = (2*c)/(-b + np.sqrt(b**2 - 4*a*c))
        r2 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    x1 = np.max((r1,r2))
    x2 = np.min((r1,r2))
    return x1,x2

# Print results for part (a) and (b)
root1, root2 = quadratic_ab(0.001, 1000, 0.001)
print(f"(a) Standard formula solutions: x1 = {root1}, x2 = {root2}")

root1, root2 = quadratic_ab(0.001, 1000, 0.001, method_b=True)
print(f"(b) Alternative formula solutions: x1 = {root1}, x2 = {root2}")

# Print results for part (c)
root1, root2 = quadratic(0.001, 1000, 0.001)
print(f"(c) New method solutions: x1 = {root1}, x2 = {root2}")

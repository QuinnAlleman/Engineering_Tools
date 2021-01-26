'''
Author: Quinn Alleman
github: https://github.com/QuinnAlleman

This module is a compilation of various functions, methods, and algorithms that I have used throughout my school and work experience.

Table of Contents:
- Approximate Relative Error: Finding the approximate relative error of a iterative algorithm/method.
- Bisection Method: Useful for finding a solution to an implicit function.
- False Position Method: Depending on the function and the boundary location, it will converge at a different rate than the bisection method.
    Not as robust, but will often converge faster if the conditions are right.
- Secant Method: Finding the root of a function without having to provide a a range or the derivative like the Newton-Raphson method.

To-Add:
- Gauss Elimination
- LU Decomposition
- Multi-Dimensional Optimization
- Constrained Optimization with linear Programming
- Fast Quadratic Interpolation for x points.
- Fourier Approximation
- Least Squares Regression
- Integration Methods
    - Trapezoidal Rule
    - Simpson's Rule
- Differentiation
- ODEs
    - Runga Kutta
    - Eigenvalue
    - Boundary Value


To-Do:
- Add in functionality to pass in function or value as RHS of equation in root finding methods.
    - This may include adding in new function to handle creating lambda function for each method. Would avoid having to reuse same code.
'''

# Approximate Relative Error - Uses the previous and current guess of an iterative function to find the apprximate error along with accounting for divide by zero errors.
def approx_relative_error(previous_estimate, current_estimate):
    if current_estimate == 0:
        return float('inf')
    return abs((current_estimate-previous_estimate)/current_estimate)

# Bisection Method - Finding the root of a function
def bisection_method(input_func, x_lower, x_upper, y=0, max_error = 1e-5, max_iterations = 100, print_out = False):
    '''
    Find the approximate solution to f(x) = y  for x in an interval of x_lower -> x_upper.
    This function is helpful for solving implicit functions. This is where x may show up several times and can't be solved for explicitly.
    Wikipedia Article: https://en.wikipedia.org/wiki/Bisection_method

    Input Parameters
    ================
    input_func: Type = function
        The function you are finding the solution for f(x) = y
    x_lower, x_upper = numbers
        The interval you are expecting a solution to lay for f(x) = y
    
    Optional Input Parameters
    =========================
    y: number or single input lmabda function.
        This is the RHS for the function. If there is no arguments passed, it will default to y=0 and find the root of f(x)=0.
    max_error: number
        The method will stop looping when it gets below this error. Default is 1e-5.
    max_iterations: number
        This will default to 100 iterations. The root finding will stop after this many iterations.
    print_out: boolean
        This will print out to the console information on the solution.

    Returns
    =======
    x_solution: number
        This value is the approximated solution to the function. It will return the solution once either the error tolerance or maximum iterations has been satisfied.

    Examples
    ========
    >>> my_function = lambda x: 2*x**4-20
    >>> bisection_method(my_function, 0, 100)  # In this case, y=0 and the iterations are set to default.
    '''

    # Initial Variables Setup
    # Id the user input the wrong order of lower and upper, this should fix it.
    solve_func = lambda x: input_func(x) - y
    x_a = min([x_lower, x_upper])
    x_b = max([x_lower, x_upper])

    # Check to make sure there is a solution guaranteed. If not, raise an error.
    if (solve_func(x_a)*solve_func(x_b) > 0):
        raise ValueError('Solution is not guaranteed in the defined bounds. Try changing the bounds.')

    for current_iteration in range(max_iterations): # Loop through the iterations.
        # There are 3 cases for each iteration.
        #   1. x_c that is solved for is the exact solution.
        #   2. The solution lies between x_c and x_b.
        #   3. The solution lies between x_a and x_c.
        # There are also 2 cases which will cause the loop to end.
        #   1. The current error is below the max error tolerance.
        #   2. The current iteration is the last loop, thus an approximate solution was not found in the defined max iterations.
        if current_iteration == 0:
            x_p = x_a
        else:
            x_p = x_c
        x_c = (x_a+x_b)/2
        if (solve_func(x_c) == 0): # Solution Found. Very unlikely.
            if print_out:
                print('Exact answer was found!\nAnswer: %.6e\nIterations Run: %d'%(x_c, current_iteration))
            return x_c
        elif (solve_func(x_a) > 0) == (solve_func(x_c) > 0): # If x_a and x_c are the same sign. The solution lies in x_c -> x_b
            x_a = x_c
        else: # The solution lies in between x_a -> x_c
            x_b = x_c
        current_error = approx_relative_error(x_p, x_c)
        if current_error < max_error: # solution found within acceptable error tolerance.
            if print_out:
                print('Approximate answer was found!\nApproximation: %.6e\nIterations Run: %d\nApproximate Error: %.6e'%(x_c, current_iteration, current_error))
            return x_c
    if print_out:
        print('Did not find an answer in %f iterations below a %.3e approximate error tolerance.' % (max_iterations, max_error))
    raise ValueError('Max iterations run in bisection method. Answer was not approximated.')

# False Position Method - Finding the root of a function (Usually more efficient than Bisection Method)
def false_position_method(input_func, x_lower, x_upper, y=0, max_error = 1e-5, max_iterations = 100, print_out = False):
    '''
    Find the approximate solution to f(x) = y  for x in an interval of x_lower -> x_upper.
    This function is helpful for solving implicit functions. This is where x may show up several times and can't be solved for explicitly.
    This method will estimate the location of the zero based on a linear interpolation.
    Wikipedia Article: https://en.wikipedia.org/wiki/Regula_falsi

    Input Parameters
    ================
    input_func: Type = function
        The function you are finding the solution for f(x) = y
    x_lower, x_upper = numbers
        The interval you are expecting a solution to lay for f(x) = y
    
    Optional Input Parameters
    =========================
    y: number
        This is the RHS for the function. If there is no arguments passed, it will default to y=0 and find the root of f(x)=0.
    max_error: number
        The method will stop looping when it gets below this error.
    max_iterations: number
        This will default to 100 iterations. The root finding will stop after this many iterations.
    print_out: boolean
        This will print out to the console information on the solution.

    Returns
    =======
    x_solution: number
        This value is the approximated solution to the function. It will return the solution once either the error tolerance or maximum iterations has been satisfied.

    Examples
    ========
    >>> my_function = lambda x: 2*x**4-20
    >>> false_position_method(my_function, 0, 100)  # In this case, y=0 and the iterations are set to default.
    
    '''

    # Initial Variables Setup
    # Id the user input the wrong order of lower and upper, this should fix it.
    solve_func = lambda x: input_func(x) - y
    x_a = min([x_lower, x_upper])
    x_b = max([x_lower, x_upper])

    # Check to make sure there is a solution guaranteed. If not, raise an error.
    if (solve_func(x_a)*solve_func(x_b) > 0):
        raise ValueError('Solution is not guaranteed in the defined bounds. Try changing the bounds.')

    for current_iteration in range(max_iterations): # Loop through the iterations.
        # There are 3 cases for each iteration.
        #   1. x_c that is solved for is the exact solution.
        #   2. The solution lies between x_c and x_b.
        #   3. The solution lies between x_a and x_c.
        # There are also 2 cases which will cause the loop to end.
        #   1. The current error is below the max error tolerance.
        #   2. The current iteration is the last loop, thus an approximate solution was not found in the defined max iterations.
        if current_iteration == 0:
            x_p = x_a
        else:
            x_p = x_c
        x_c = x_b-(solve_func(x_b)*(x_a-x_b))/(solve_func(x_a)-solve_func(x_b))
        if (solve_func(x_c) == 0): # Solution Found. Very unlikely.
            if print_out:
                print('Exact answer was found!\nAnswer: %.6e\nIterations Run: %d'%(x_c, current_iteration))
            return x_c
        elif (solve_func(x_a) > 0) == (solve_func(x_c) > 0): # If x_a and x_c are the same sign. The solution lies in x_c -> x_b
            x_a = x_c
        else: # The solution lies in between x_a -> x_c
            x_b = x_c
        current_error = approx_relative_error(x_p, x_c)
        if current_error < max_error: # solution found within acceptable error tolerance.
            if print_out:
                print('Approximate answer was found!\nApproximation: %.6e\nIterations Run: %d\nApproximate Error: %.6e'%(x_c, current_iteration, current_error))
            return x_c
    if print_out:
        print('Did not find an answer in %f iterations below a %.3e approximate error tolerance.' % (max_iterations, max_error))
    raise ValueError('Max iterations run in false-position method. Answer was not approximated.')

# Secant Method - Finding the root of a function without having to provide a a range or the derivative like the Newton-Raphson method.
def secant_method(input_func, x_0, x_1, y=0, max_error = 1e-5, max_iterations = 100, print_out = False):
    '''
    Find the approximate solution to f(x) = y  for x with two initial guesses (x_0 and x_1).
    This function is helpful for solving implicit functions. This is where x may show up several times and can't be solved for explicitly.
    This method will estimate the location of the zero based on a linear interpolation of two points.
    Wikipedia Article: https://en.wikipedia.org/wiki/Secant_method

    Input Parameters
    ================
    input_func: Type = function
        The function you are finding the solution for f(x) = y
    x_0, x_1 = numbers
        Two initial guesses for where you might expect a solution to lie. These do NOT need to bracket the solution.
    
    Optional Input Parameters
    =========================
    y: number
        This is the RHS for the function. If there is no arguments passed, it will default to y=0 and find the root of f(x)=0.
    max_error: number
        The method will stop looping when it gets below this error.
    max_iterations: number
        This will default to 100 iterations. The root finding will stop after this many iterations.
    print_out: boolean
        This will print out to the console information on the solution.

    Returns
    =======
    x_solution: number
        This value is the approximated solution to the function. It will return the solution once either the error tolerance or maximum iterations has been satisfied.

    Examples
    ========
    >>> my_function = lambda x: 2*x**4-20
    >>> secant_method(my_function, 0, 1)  # In this case, y=0 and the iterations are set to default.
    
    '''

    # Initial Variables Setup
    solve_func = lambda x: input_func(x) - y

    for current_iteration in range(max_iterations): # Loop through the iterations.
        # There are 2 cases for each iteration.
        #   1. x_0 is the exact solution.
        #   2. The solution is interpolated with the two previous points.
        # There are also 2 cases which will cause the loop to end.
        #   1. The current error is below the max error tolerance.
        #   2. The current iteration is the last loop, thus an approximate solution was not found in the defined max iterations.
        if solve_func(x_0) == 0:
            if print_out:
                print('Exact answer was found!\nAnswer: %.6e\nIterations Run: %d'%(x_0, current_iteration))
            return x_0
        current_error = approx_relative_error(x_0, x_1)
        if current_error < max_error: # Solution found within acceptable error tolerance.
            if print_out:
                print('Approximate answer was found!\nApproximation: %.6e\nIterations Run: %d\nApproximate Error: %.6e'%(x_1, current_iteration, current_error))
            return x_1
        x_2 = x_1 - (solve_func(x_1)*(x_0-x_1)/(solve_func(x_0)-solve_func(x_1)))
        x_0, x_1 = x_1, x_2
        
    if print_out:
        print('Did not find an answer in %f iterations below a %.3e approximate error tolerance.' % (max_iterations, max_error))
    raise ValueError('Max iterations run in false-position method. Answer was not approximated.')


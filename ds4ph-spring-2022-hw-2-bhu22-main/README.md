# Homework 2 for DS4PH 2022

In a single jupyter notebook file named `hw2.ipynb` create the following:

1. A sorting hat program that, when run, takes in a series of simple Y or N answers and returns your Hogwart's house.
2. Write an object `mydata` that takes in a vector of numbers and has the methods: `mean` and `std` (for standard deviation) returning the mean and standard deviation respectively.
3. Create a lag function as follows. It takes in a numpy vector, say `a = numpy.array( [1, 6, 8, 4] )` and an integer > 0. It then returns the list lagged by the integer value. So, `lag(a, 1)` will return `[nan, 1, 6, 8, 4]` and `lag(a, 2)` will return `[nan, nan, 1, 6, 8]`. That is, it gets rid of `n` values at the end and appends that many `nan`s at the beginning. Use `numpy`'s `nan` for the missing values. It should return a numpy array.
4. Using your code from 3, write a function that returns a function that lags by a specific amount. So, for example, if we name the function `lagGenerator` then `lagGenerator(2)` creates a function that lags by 2. 


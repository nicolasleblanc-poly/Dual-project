This program allows to find the value of a parameter zeta for which a function will be convex 
for values larger than the zeta value found. The code works by first finding an approximate
value of zeta such that all of the eigenvalues of the initial matrix or operator are positive.
This approximate value is found using two jacobi-Davidson algorithms: Ritz and harmonic Ritz.
The first allows to find the extremal eigenvalue and the second allows to find the eigenvalue
closest to 0. In both cases, the eigenvalue found can be either positive or negative. After the
approximate zeta value is found, the Padé approximate part of the program begins and allows to 
find the last root of the surrogate function. After this value, all of the larger zeta values 
will give all positive eigenvalues, which ensures the function is convex.

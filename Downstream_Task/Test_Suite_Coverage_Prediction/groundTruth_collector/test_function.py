def test_function():
    from math import factorial
    n = 1894;p = 1
    a = [50]
    even=0
    odd=0
    for i in a:
        if i%2==0:
            even+=1
        else:
            odd+=1
    def comb(a,b):
        return factorial(a)//(factorial(a-b)*factorial(b))
    n_even=0
    n_odd=0
    for i in range(even+1):
        n_even+=comb(even,i)
    if p==1:
        for i in range(1,odd+1,2):
            n_odd+=comb(odd,i)
    else:
        for i in range(0,odd+1,2):
            n_odd+=comb(odd,i)
    print(n_even*n_odd)
test_function()
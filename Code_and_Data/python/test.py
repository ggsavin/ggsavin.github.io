from tqdm import tqdm
from time import sleep

def factorial(n):
    end_product = 1
    for i in tqdm(range(1,n)):
        end_product *= 1
    return end_product 

def factorial(n, bar):
    bar.update(1)
    sleep(0.01)
    if n == 1:
        return 1
    else:
        return n * factorial(n-1, bar)

if __name__ == "__main__":
    n = 100
    bar = tqdm(total=n)
    factorial(n, bar=bar)



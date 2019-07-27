import math

def gugudan(dansu):
    # for i in range(2,10):
    #     for j in range(1, 10):
    #         print("{} * {} = {}".format(i,j,i*j))
    for i in range(1, 10):
        print("{} * {} = {}".format(dansu, i, dansu*i))
def is_prime_number(num):
    if num == 2:
        return True
    else:
        result = True
        for i in range(2, num):
            if (num % i) == 0:
                result = False
                break
        return result

def prime_number():
    for i in range(1,1001):
        if is_prime_number(i):
            print(i)

def star(num):
    for i in range(1, num+1):
        print('*' * i)
    print("          ")
    for i in range(0, num+1):
        if i == 0:
            dot = num
        if dot == 0 or dot == 1:
            break
        print(" " * i + "*" * dot)
        dot -= 2


if __name__ == '__main__':
    # dansu = int(input())
    # gugudan(dansu)
    # prime_number()
    # a = int(input())
    # star(a)
    prime_number()

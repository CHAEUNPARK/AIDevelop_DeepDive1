data = [1, 27, 9, 100, 31, 80, 11]
def Swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b
def quick_sort(left, right):
    global data
    pivot = left
    i = left + 1
    j = right

    while i < j:
        while data[pivot] < data[i]:
            i += 1
        while i < j and data[j] < data[pivot]:
            j -= 1

        data[i] , data[j] = Swap(data[i], data[j])
    data[i], data[pivot] = Swap(data[i], data[pivot])
    pivot = i
    quick_sort(left, pivot-1)
    quick_sort(pivot+1, right)


if __name__ == '__main__':
    quick_sort(0,len(data)-1)
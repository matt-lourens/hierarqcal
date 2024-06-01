def bitonic_sort(array):
    def bitonic_merge(array, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (array[i] > array[i + k]) == direction:
                    array[i], array[i + k] = array[i + k], array[i]
            bitonic_merge(array, low, k, direction)
            bitonic_merge(array, low + k, k, direction)

    def bitonic_sort_recursive(array, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_recursive(array, low, k, True)
            bitonic_sort_recursive(array, low + k, k, False)
            bitonic_merge(array, low, cnt, direction)

    bitonic_sort_recursive(array, 0, len(array), True)
    return array

# Implementing Bitonic Sort using HierarQcal
def bitonic_sort_hierarq(array):
    n = len(array)
    h_bitonic_sort = Qinit(n, state=array)
    for i in range(1, n):
        for j in range(i, 0, -1):
            asc = (i + j) % 2 == 0
            h_bitonic_sort += (
                Qcycle(stride=2 ** j, step=2 ** j, offset=0, mapping=h_comparator, boundary="open")
                + Qmask("*1")
            )
    return h_bitonic_sort()

# Demonstrate Bitonic Sort
array = [8, 5, 2, 1, 3, 4, 6, 7, 7, 7]
sorted_array = bitonic_sort(array.copy())
sorted_array_hierarq = bitonic_sort_hierarq(array.copy())

print(f"Starting array: {array}\nSorted array (Python): {sorted_array}")
print(f"Sorted array (HierarQcal): {sorted_array_hierarq}")

# Plot the circuit
plot_circuit(hierq)

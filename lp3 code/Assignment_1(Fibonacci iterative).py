def fibonacci_iterative(n): 
    if n <= 0:
        return [] 
    elif n == 1:
        return [0]

    sequence = [0, 1] 
    for _ in range(2, n):
        sequence.append(sequence[-1] + sequence[-2]) 
    return sequence

print(fibonacci_iterative(12))
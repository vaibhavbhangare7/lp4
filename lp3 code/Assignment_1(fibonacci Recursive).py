def fibonacci_Recursive(n): 
    sequence = []
    for i in range(n): 
        if i == 0:
            sequence.append(0) 
        elif i == 1:
            sequence.append(1) 
        else:
            sequence.append(sequence[i - 1] + sequence[i - 2]) 
    return sequence

print(fibonacci_Recursive(12))
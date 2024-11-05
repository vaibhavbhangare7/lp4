def is_safe(board, row, col):
    # Check this column on upper side 
    for i in range(row):
        if board[i][col] == 1: 
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)): 
        if j < 0:
            break
        if board[i][j] == 1: 
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, len(board))): 
        if j >= len(board):
            break
        if board[i][j] == 1: 
            return False

    return True

def solve_n_queens(board, row): 
    if row >= len(board):
        print_board(board)
        return True # Change this to return all solutions instead of stopping at first

    for col in range(len(board)): 
        if is_safe(board, row, col):
            board[row][col] = 1 # Place queen 
            solve_n_queens(board, row + 1) # Recur to place the rest 
            board[row][col] = 0 # Backtrack

    return False

def print_board(board): 
    for row in board:
        print(" ".join('Q' if x == 1 else '.' for x in row)) 
    print()

 

N = int(input("Enter the size of the board:")) 
board = [[0 for _ in range(N)] for _ in range(N)]
# Place the first queen at (0, 0) 
board[0][0] = 1
# Solve for the remaining queens 
solve_n_queens(board, 1)

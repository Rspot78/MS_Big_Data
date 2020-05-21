import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

def display_value(maze, V, states_target = None, states_restart = None, scale = 1, colors = ['r','b','g'], filename = None):
    maze_ = maze.astype(float)
    n1, n2 = maze_.shape
    if V is not None:
        maze_.data = V
        value_min = np.min(maze_.data)
        value_max = np.max(maze_.data)
        if value_min < value_max:
            maze_.data = 0.8 * (maze_.data - value_min) / (value_max - value_min) + 0.2
    maze_ = np.array(maze_.todense())
    plt.figure(figsize=(scale * 5 * n2 / max(n1, n2), scale * 5 * n1 / max(n1, n2)))
    plt.imshow(maze_, cmap = 'gray')
    if states_target is not None:
        for k, target in enumerate(states_target):
            plt.scatter(target[1], target[0], s=200, c=colors[k%len(colors)], marker = '*')
    if states_restart is not None:
        for k, states_ in enumerate(states_restart):
            for state in states_:
                plt.scatter(state[1], state[0], s=100, c=colors[k%len(colors)])    
    plt.axis('off')
    if filename is not None:
        plt.savefig(filename + '.pdf')
    else:
        plt.show()
        
def display_policy(maze, moves, scale = 1, filename = None):
    maze_ = maze.astype(float)
    n1, n2 = maze_.shape
    maze_ = np.array(maze_.todense())
    plt.figure(figsize=(scale * 5 * n2 / max(n1, n2), scale * 5 * n1 / max(n1, n2)))
    plt.imshow(maze_, cmap = 'gray')
    for state in moves:
        action = moves[state]
        plt.arrow(state[1], state[0] , action[1], action[0], color='r', width = 0.15, length_includes_head=True)
    plt.axis('off')
    if filename is not None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')
    else:
        plt.show()

def display_maze(maze, states_target = None, states_restart = None, scale = 1, colors = ['r','b','g'], filename = None):
    display_value(maze, None, states_target, states_restart, scale, colors, filename)
        
def display_diff_policy(maze_map, moves, moves_prev, scale = 1, filename = None):
    maze_ = maze_map.astype(float)
    n1, n2 = maze_.shape
    maze_ = np.array(maze_.todense())
    plt.figure(figsize=(scale * 5 * n2 / max(n1, n2), scale * 5 * n1 / max(n1, n2)))
    plt.imshow(maze_, cmap = 'gray')
    for state in moves:
        c = 'r'
        action = moves[state]
        if state not in moves_prev or action != moves_prev[state]:
            c = 'b'
        plt.arrow(state[1], state[0] , action[1], action[0], color=c, width = 0.15, length_includes_head=True)
    plt.axis('off')
    if filename is not None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')
    else:
        plt.show()

def normalize_sparse(matrix):
    """Normalize a sparse non-negative matrix so that each row sums to 1 (or zero is the row is null)"""
    n = matrix.shape[1]
    sums = matrix.dot(np.ones(n))
    sums_inv_diag = sparse.csr_matrix(sparse.diags(sums))
    sums_inv_diag.data = 1 / sums_inv_diag.data
    return sums_inv_diag.dot(matrix)

def get_moves(policy, model):
    states = model.states
    moves = {}
    for i in range(len(states)):
        indices = policy[i].indices
        if len(indices):
            j = indices[0]
            moves[states[i]] = tuple(np.array(states[j]) - np.array(states[i]))
    if hasattr(model, 'states_target'):
        for state in model.states_target:
            if state in moves:
                moves.pop(state)
    if hasattr(model, 'states_terminal'):
        for state in model.states_terminal:
            if state in moves:
                moves.pop(state)
    return moves


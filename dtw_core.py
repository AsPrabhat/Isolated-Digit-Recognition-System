# dtw_core.py
import numpy as np
import matplotlib.pyplot as plt
import os

def DTW(sequence1, sequence2):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        sequence1 (list or np.array): The first sequence (M elements).
        sequence2 (list or np.array): The second sequence (N elements).
        Each element can be a scalar or a vector (for MFCCs).
        
    Returns:
        tuple: (opt_distance, optimal_path, DTW_cumulate_Matrix)
            - opt_distance (float): The DTW distance between the two sequences.
            - optimal_path (list of tuples): The optimal warping path [(i0,j0), (i1,j1), ...].
            - DTW_cumulate_Matrix (np.array): The accumulated cost matrix.
    """
    M = len(sequence1)
    N = len(sequence2)

    s1 = np.array(sequence1)
    s2 = np.array(sequence2)

    DTW_cumulate_Matrix = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            local_cost = np.linalg.norm(s1[i] - s2[j])
            if i == 0 and j == 0:
                DTW_cumulate_Matrix[i, j] = local_cost
            elif i == 0:
                DTW_cumulate_Matrix[i, j] = local_cost + DTW_cumulate_Matrix[i, j-1]
            elif j == 0:
                DTW_cumulate_Matrix[i, j] = local_cost + DTW_cumulate_Matrix[i-1, j]
            else:
                DTW_cumulate_Matrix[i, j] = local_cost + min(DTW_cumulate_Matrix[i-1, j],
                                                              DTW_cumulate_Matrix[i, j-1],
                                                              DTW_cumulate_Matrix[i-1, j-1])
    
    opt_distance = DTW_cumulate_Matrix[M-1, N-1]

    i = M - 1
    j = N - 1
    optimal_path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Check diagonal first, as it's often preferred
            if DTW_cumulate_Matrix[i-1, j-1] <= DTW_cumulate_Matrix[i-1, j] and \
               DTW_cumulate_Matrix[i-1, j-1] <= DTW_cumulate_Matrix[i, j-1]:
                i -= 1
                j -= 1
            # Then check vertical (from 'below' in matrix terms, or 'up' if origin='lower')
            elif DTW_cumulate_Matrix[i-1, j] <= DTW_cumulate_Matrix[i-1, j-1] and \
                 DTW_cumulate_Matrix[i-1, j] <= DTW_cumulate_Matrix[i, j-1]:
                i -= 1
            # Else, must be horizontal
            else:
                j -= 1
        optimal_path.append((i, j))

    optimal_path = optimal_path[::-1]
    return opt_distance, optimal_path, DTW_cumulate_Matrix


def plotDTWpath(sequence1, sequence2, title_prefix="", save_path=None):
    """
    Computes and plots the DTW path on local cost and accumulated cost matrices.
    Saves the plot if save_path is provided.

    Args:
        sequence1 (list or np.array): The first sequence.
        sequence2 (list or np.array): The second sequence.
        title_prefix (str): String to prepend to plot titles.
        save_path (str, optional): Path to save the figure. If None, shows the plot.

    Returns:
        str: Path to the saved figure if save_path is provided, else None.
    """
    s1 = np.array(sequence1)
    s2 = np.array(sequence2)
    M = len(s1)
    N = len(s2)

    distance_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            distance_matrix[i, j] = np.linalg.norm(s1[i] - s2[j])

    optimal_distance, optimal_path, DTW_accumulated_matrix = DTW(s1, s2)
    optimal_path_array = np.array(optimal_path)

    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(distance_matrix, origin='lower', cmap='viridis', interpolation='nearest')
    plt.plot(optimal_path_array[:, 1], optimal_path_array[:, 0], color='red', marker='o', markersize=3, linewidth=1.5, label='Optimal Path')
    plt.title(f'{title_prefix}Local Cost Matrix\nOptimal DTW Distance: {optimal_distance:.2f}')
    plt.xlabel('Sequence B Index')
    plt.ylabel('Sequence A Index')
    plt.colorbar(label='Local Cost (Distance)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(DTW_accumulated_matrix, origin='lower', cmap='magma', interpolation='nearest')
    plt.plot(optimal_path_array[:, 1], optimal_path_array[:, 0], color='lime', marker='o', markersize=3, linewidth=1.5, label='Optimal Path')
    plt.title(f'{title_prefix}Accumulated Cost Matrix')
    plt.xlabel('Sequence B Index')
    plt.ylabel('Sequence A Index')
    plt.colorbar(label='Accumulated Cost')
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plots_dir = os.path.dirname(save_path)
        if plots_dir and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None

if __name__ == '__main__':
    print("Running DTW Core Example...")
    sequenceA = [4, 3, 7, 0, 2, 6, 5]
    sequenceB = [3, 7, 1, 6, 1, 5, 4, 4]
    
    dist, path, acc_cost_matrix = DTW(sequenceA, sequenceB)
    print(f"Optimal DTW Distance: {dist}")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_save_path = "plots/dtw_core_example_plot.png"
    plotDTWpath(sequenceA, sequenceB, title_prefix="Part (a) Example: ", save_path=plot_save_path)
    print(f"DTW Core Example Plot saved to {plot_save_path}")
    print("DTW Core Example Finished.")
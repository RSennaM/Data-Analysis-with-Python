import numpy as np

def calculate(list):
    if len(list)==9:
    
        matrix = np.array(list).reshape(3, 3)

        mean_values = [np.mean(matrix, axis=0).tolist(), np.mean(matrix, axis=1).tolist(), np.mean(matrix.flatten())]
        variance_values = [np.var(matrix, axis=0).tolist(), np.var(matrix, axis=1).tolist(), np.var(matrix.flatten())]
        standard_deviation_values = [np.std(matrix, axis=0).tolist(), np.std(matrix, axis=1).tolist(), np.std(matrix.flatten())]
        max_values = [np.max(matrix, axis=0).tolist(), np.max(matrix, axis=1).tolist(), np.max(matrix.flatten())]
        min_values = [np.min(matrix, axis=0).tolist(), np.min(matrix, axis=1).tolist(), np.min(matrix.flatten())]
        sum_values = [np.sum(matrix, axis=0).tolist(), np.sum(matrix, axis=1).tolist(), np.sum(matrix.flatten())]
    
        calculations = {
            'mean': mean_values,
            'variance': variance_values,
            'standard deviation': standard_deviation_values,
            'max': max_values,
            'min': min_values,
            'sum': sum_values
        }

    else:
        raise ValueError('List must contain nine numbers.')

    return calculations

import itertools
from typing import List, Tuple, Dict
from sklearn.metrics import adjusted_rand_score

def pairwise_f1_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute the pairwise F1 score for clustering evaluation.
    
    Args:
        true_labels (List[int]): Ground truth cluster labels.
        pred_labels (List[int]): Predicted cluster labels.
    
    Returns:
        float: Pairwise F1 score.
    """
    # Generate all unique unordered pairs of indices
    pairs = list(itertools.combinations(range(len(true_labels)), 2))
    
    # Initialize counts
    tp = fp = fn = 0
    
    for i, j in pairs:
        # True same-cluster?
        true_same = (true_labels[i] == true_labels[j])
        # Predicted same-cluster?
        pred_same = (pred_labels[i] == pred_labels[j])
        
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        # True negatives (not same in both) are not used in F1
    # print(tp, fp, fn)
    # Handle edge cases
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def pairwise_f1_score_per_speaker(true_labels: List[int], pred_labels: List[int]) -> Dict[int, float]:
    """
    Compute the pairwise F1 score for each speaker (one-vs-rest style) in clustering evaluation.
    
    Args:
        true_labels (List[int]): Ground truth cluster labels.
        pred_labels (List[int]): Predicted cluster labels.
    
    Returns:
        Dict[int, float]: Mapping from speaker index to their pairwise F1 score.
    """
    n = len(true_labels)
    scores = {}

    for i in range(n):
        tp = fp = fn = 0
        for j in range(n):
            if i == j:
                continue

            # True and predicted same-cluster relationships between i and j
            true_same = (true_labels[i] == true_labels[j])
            pred_same = (pred_labels[i] == pred_labels[j])

            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1

        # Compute F1 for this speaker
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        scores[i] = f1

    return scores

if __name__ == "__main__":
    # Example usage
    examples: List[Tuple[List[int], List[int]]] = [
        ([0, 0, 1, 1], [0, 0, 2, 2]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 1, 2], [0, 0, 1, 1]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 0, 0, 1, 2], [1, 1, 0, 0, 2, 2]),
        ([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1])
    ]

    # Compute and display results
    results = [(true, pred, pairwise_f1_score(true, pred), adjusted_rand_score(true, pred)) for true, pred in examples]
    for true, pred, f1, ari in results:
        print(f"True: {true}, Pred: {pred}, F1: {f1}, ARI: {ari}")

    # Compute per-speaker F1 scores
    for true, pred in examples:
        per_speaker_f1 = pairwise_f1_score_per_speaker(true, pred)
        print(f"True: {true}, Pred: {pred}, Per-Speaker F1: {per_speaker_f1}")
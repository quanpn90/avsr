import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import AgglomerativeClustering
import json
import glob
from cluster.eval import pairwise_f1_score, pairwise_f1_score_per_speaker
from talking_detector.segmentation import segment_by_asd
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import traceback

MAX_SPEAKERS = 8
MAX_CONVERSATIONS = 4



def validate_speaker_segments(speaker_segments: Dict[str, List[Tuple[float, float]]]) -> None:
    """
    Validate speaker segments against constraints.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their time segments
        
    Raises:
        ValueError: If constraints are violated
    """
    if len(speaker_segments) > MAX_SPEAKERS:
        raise ValueError(f"Maximum number of speakers is {MAX_SPEAKERS}")
    
    # Validate time segments
    for spk_id, segments in speaker_segments.items():
        if not segments:
            raise ValueError(f"Speaker {spk_id} has no time segments")
        # if len(segments) > MAX_TURNS_PER_SPEAKER:
            # raise ValueError(f"Speaker {spk_id} has {len(segments)} turns, maximum is {MAX_TURNS_PER_SPEAKER}")
        for start, end in segments:
            if start >= end:
                raise ValueError(f"Invalid time segment for speaker {spk_id}: start ({start}) >= end ({end})")

def calculate_overlap_duration(segments1: List[Tuple[float, float]], 
                             segments2: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate total overlap and non-overlap duration between two speakers' segments.
    
    Args:
        segments1: List of (start, end) tuples for first speaker
        segments2: List of (start, end) tuples for second speaker
        
    Returns:
        Tuple of (total_overlap_duration, total_non_overlap_duration)
    """
    total_overlap = 0.0
    total_non_overlap = 0.0
    
    # Calculate total duration of each speaker's segments
    total_duration1 = sum(end - start for start, end in segments1)
    total_duration2 = sum(end - start for start, end in segments2)
    
    # Calculate overlaps
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
    
    # Calculate non-overlap
    total_non_overlap = total_duration1 + total_duration2 - 2 * total_overlap
    
    return total_overlap, total_non_overlap

def calculate_conversation_scores(speaker_segments: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
    """
    Calculate conversation likelihood scores between all pairs of speakers.
    Higher score means speakers are more likely to be in the same conversation.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their time segments
        
    Returns:
        NxN numpy array of conversation scores
    """
    n_speakers = len(speaker_segments)
    scores = np.zeros((n_speakers, n_speakers))
    speaker_ids = list(speaker_segments.keys())
    
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk1 = speaker_ids[i]
            spk2 = speaker_ids[j]
            
            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments[spk1],
                speaker_segments[spk2]
            )
            
            # Calculate conversation likelihood score
            # Higher score when there's less overlap (more likely to be in same conversation)
            if overlap + non_overlap > 0:
                # Normalize overlap by total duration to get overlap ratio
                total_duration = overlap + non_overlap
                overlap_ratio = overlap / total_duration
                # Convert to conversation likelihood (1 - overlap_ratio)
                score = 1 - overlap_ratio
            else:
                score = 0
                
            scores[i, j] = score
            scores[j, i] = score  # Symmetric
    
    return scores

def cluster_speakers(scores: np.ndarray, 
                    speaker_ids: List[str],
                    threshold: float = 0.7,
                    n_clusters: int = None) -> Dict[int, List[str]]:
    """
    Cluster speakers based on their conversation scores.
    
    Args:
        scores: NxN numpy array of conversation scores
        speaker_ids: List of speaker IDs
        threshold: Minimum score to consider speakers in same conversation
        n_clusters: Number of clusters (if None, will be determined automatically)
        
    Returns:
        Dictionary mapping cluster IDs to lists of speaker IDs
    """
    if n_clusters is not None and n_clusters > MAX_CONVERSATIONS:
        raise ValueError(f"Maximum number of conversations is {MAX_CONVERSATIONS}")
    
    # Convert scores to distance matrix (1 - score)
    distances = 1 - scores
    
    # Perform hierarchical clustering
    if n_clusters is None:
        # Use threshold to determine number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            metric='precomputed',
            linkage='complete'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, MAX_CONVERSATIONS),
            metric='precomputed',
            linkage='complete'
        )
    
    cluster_labels = clustering.fit_predict(distances)
    
    # Group speakers by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(speaker_ids[i])
    
    # Speaker to cluster mapping
    spk_to_cluster = {spk_id: label.item() for spk_id, label in zip(speaker_ids, cluster_labels)}
    return spk_to_cluster

def get_speaker_activity_segments(pycrop_asd_path: List[str], uem_start: int, uem_end: int) -> List[Tuple[float, float]]:
    pycrop_asd_path = sorted(pycrop_asd_path)
    all_frames = dict({})
    for asd_path in pycrop_asd_path:
        with open(asd_path, "r") as f:
            asd_data = json.load(f)
            all_frames.update(asd_data)
    sorted_frames = sorted(all_frames.items(), key=lambda x: int(x[0]))
    
    # activity_segments = []
    # current_segment = []
    # for frame_id, score in sorted_frames:
    #     if score > 2.5:
    #         current_segment.append(frame_id)
    #     else:
    #         if current_segment:
    #             activity_segments.append(current_segment)
    #             current_segment = []
    # if current_segment:
    #     activity_segments.append(current_segment)
    
    activity_segments = segment_by_asd(all_frames)

    activity_segments = [
        (int(segment[0])/25, int(segment[-1])/25)
        for segment in activity_segments
    ]
    
    aligned_activity_segments = []
    for segment in activity_segments:
        if segment[1] < uem_start:
            continue
        if segment[0] > uem_end:
            break
        aligned_activity_segments.append([
            # max(segment[0] - uem_start, 0),
            # min(segment[1] - uem_start, uem_end - uem_start)
            segment[0] - uem_start,
            segment[1] - uem_start
        ])
    
    return aligned_activity_segments

def get_clustering_f1_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> float:
    """
    Calculate F1 score for clustering results.
    
    Args:
        conversation_clusters_label: Dictionary mapping speaker IDs to cluster IDs
        true_clusters_label: Dictionary mapping speaker IDs to true cluster IDs
        
    Returns:
        F1 score
    """
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    return pairwise_f1_score(true_clusters, conversation_clusters)

def get_speaker_clustering_f1_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> Dict[str, float]:
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    f1_scores = pairwise_f1_score_per_speaker(true_clusters, conversation_clusters)
    spk_to_f1_score = {spk: round(f1_scores[i], 4) for i, spk in enumerate(spk_list)}
    return spk_to_f1_score

def get_clustering_ari_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> float:
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    return adjusted_rand_score(true_clusters, conversation_clusters)
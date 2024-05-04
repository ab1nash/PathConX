import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from collections import defaultdict

from joblib import Parallel, delayed


def count_all_paths_with_mp(e2re, max_path_len, head2tails):
    '''
    A parallel version of count_all_paths

    :param e2re: entity to relation dictionary
    :param max_path_len: maximum length of paths
    :param head2tails: [(head, {tail1, tail2 ...}), ...]
    '''
    n_cores, pool, range_list = get_params_for_mp(len(head2tails))
    results = pool.map(count_all_paths, zip([e2re] * n_cores,
                                            [max_path_len] * n_cores,
                                            [head2tails[i[0]:i[1]] for i in range_list],
                                            range(n_cores)))
    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)

    return res

def count_all_paths_with_joblib(e2re, max_path_len, head2tails):
    '''
    A parallel version of count_all_paths using joblib

    :param e2re: entity to relation dictionary
    :param max_path_len: maximum length of paths
    :param head2tails: [(head, {tail1, tail2 ...}), ...]
    '''
    # Define a function to process each subset of head2tails
    def process_subset(e2re, max_path_len, subset, pid):
        res = {}
        for i, (head, tails) in enumerate(subset):
            # Call count_all_paths for each head-tail pair
            paths = count_all_paths((e2re, max_path_len, [(head, tails)], pid))
            res.update(paths)
            print(f'pid {pid}: {i + 1}/{len(subset)}', end='\r', flush=True)  # Update progress
        return res

    # Split head2tails into smaller subsets for parallel processing
    n_jobs = 8  # Use all available CPU cores
    results = Parallel(n_jobs=n_jobs)(delayed(process_subset)(e2re, max_path_len, subset, pid)
                                      for pid, subset in enumerate(np.array_split(head2tails, n_jobs)))

    # Merge results from all subsets into a single dictionary
    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)

    return res



def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
        # print(':count_all_paths(): pid %d:  %d / %d' % (pid, i, len(head2tails)))
    # print('pid %d  done' % pid)
    return ht2paths


def bfs(head, tails, e2re, max_path_len):
    # [time gap info]? 
    # put length-1 paths into all_paths
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            entities_in_path = set([head] + [i[1] for i in path])
            for edge in e2re[last_entity_in_path]:
                # append (relation, entity) to the path if the new entity does not appear in this path before
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
            p += 1

    ht2paths = defaultdict(set)
    for path in all_paths:
        tail = path[-1][1]
        if tail in tails:  # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path]))

    return ht2paths


def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                n_paths += 1
    return path2id, id2path, id2length


def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)

    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]


def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths]
            sampled_path_ids_for_triplet = np.random.choice(
                path_ids_for_triplet, size=path_samples, replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)
    return res


def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape

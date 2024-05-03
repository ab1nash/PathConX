import os
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
# 
from data_loader_interface import DataLoaderIface

# todo
from utils import count_all_paths_with_mp, count_paths, get_path_dict_and_length, one_hot_path_id, sample_paths


class DataLoaderKG(DataLoaderIface):

    def __init__(self) -> None:
        super().__init__()
        self.entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
        self.entity2edges = []  # each row in self.entity2edges is the sampled edges connecting to this entity
        self.edge2entities = []  # each row in self.edge2entities is the two entities connected by this edge
        self.edge2relation = []  # each row in self.edge2relation is the relation type of this edge

        self.e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity

        self.args = None
        self.entity_dict = None 
        self.relation_dict = None


    def read_entities(self, file_name):
        d = {}
        file = open(file_name)
        for line in file:
            index, name = line.strip().split('\t')
            d[name] = int(index)
        file.close()

        return d


    def read_relations(self, file_name):
        bow = []
        count_vec = CountVectorizer()

        d = {}
        file = open(file_name)
        for line in file:
            index, name = line.strip().split('\t')
            d[name] = int(index)

            if self.args.feature_type == 'bow' and not os.path.exists('../data/' + self.args.dataset + '/bow.npy'):
                tokens = re.findall('[a-z]{2,}', name)
                bow.append(' '.join(tokens))
        file.close()

        if self.args.feature_type == 'bow' and not os.path.exists('../data/' + self.args.dataset + '/bow.npy'):
            bow = count_vec.fit_transform(bow)
            np.save('../data/' + self.args.dataset + '/bow.npy', bow.toarray())

        return d

    def read_quiplets(self, file_name):
        pass
    
    def read_triplets(self, file_name):
        data = []

        file = open(file_name)
        for line in file:
            head, relation, tail = line.strip().split('\t')

            head_idx = self.entity_dict[head]
            relation_idx = self.relation_dict[relation]
            tail_idx = self.entity_dict[tail]

            data.append((head_idx, tail_idx, relation_idx))
        file.close()

        return data


    def build_kg(self, train_data):
        for edge_idx, triplet in enumerate(train_data):
            head_idx, tail_idx, relation_idx = triplet

            if self.args.use_context:
                self.entity2edge_set[head_idx].add(edge_idx)
                self.entity2edge_set[tail_idx].add(edge_idx)
                self.edge2entities.append([head_idx, tail_idx])
                self.edge2relation.append(relation_idx)

            if self.args.use_path:
                self.e2re[head_idx].add((relation_idx, tail_idx))
                self.e2re[tail_idx].add((relation_idx, head_idx))

        # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
        # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
        # self.entity2edge_set[isolated_node] = {null_edge}
        # self.entity2edge_set[null_entity] = {null_edge}
        # self.edge2entities[null_edge] = [null_entity, null_entity]
        # self.edge2relation[null_edge] = null_relation
        # The feature of null_relation is a zero vector. See _build_model() of model.py for details

        if self.args.use_context:
            null_entity = len(self.entity_dict)
            null_relation = len(self.relation_dict)
            null_edge = len(self.edge2entities)
            self.edge2entities.append([null_entity, null_entity])
            self.edge2relation.append(null_relation)

            for i in range(len(self.entity_dict) + 1):
                if i not in self.entity2edge_set:
                    self.entity2edge_set[i] = {null_edge}
                sampled_neighbors = np.random.choice(list(self.entity2edge_set[i]), size=self.args.neighbor_samples,
                                                    replace=len(self.entity2edge_set[i]) < self.args.neighbor_samples)
                self.entity2edges.append(sampled_neighbors)


    def get_h2t(self, train_triplets, valid_triplets, test_triplets):
        head2tails = defaultdict(set)
        for head, tail, relation in train_triplets + valid_triplets + test_triplets:
            head2tails[head].add(tail)
        return head2tails


    def get_paths(self, train_triplets, valid_triplets, test_triplets):
        directory = '../data/' + self.args.dataset + '/cache/'
        length = str(self.args.max_path_len)

        if not os.path.exists(directory):
            os.mkdir(directory)

        if os.path.exists(directory + 'train_paths_' + length + '.pkl'):
            print('loading paths from files ...')
            train_paths = pickle.load(open(directory + 'train_paths_' + length + '.pkl', 'rb'))
            valid_paths = pickle.load(open(directory + 'valid_paths_' + length + '.pkl', 'rb'))
            test_paths = pickle.load(open(directory + 'test_paths_' + length + '.pkl', 'rb'))

        else:
            print('counting paths from head to tail ...')
            head2tails = self.get_h2t(train_triplets, valid_triplets, test_triplets)
            ht2paths = count_all_paths_with_mp(self.e2re, self.args.max_path_len, [(k, v) for k, v in head2tails.items()])
            train_set = set(train_triplets)
            train_paths = count_paths(train_triplets, ht2paths, train_set)
            valid_paths = count_paths(valid_triplets, ht2paths, train_set)
            test_paths = count_paths(test_triplets, ht2paths, train_set)

            print('dumping paths to files ...')
            pickle.dump(train_paths, open(directory + 'train_paths_' + length + '.pkl', 'wb'))
            pickle.dump(valid_paths, open(directory + 'valid_paths_' + length + '.pkl', 'wb'))
            pickle.dump(test_paths, open(directory + 'test_paths_' + length + '.pkl', 'wb'))

        # if using rnn and no path is found for the triplet, put an empty path into paths
        if self.args.path_type == 'rnn':
            for paths in train_paths + valid_paths + test_paths:
                if len(paths) == 0:
                    paths.append([])

        return train_paths, valid_paths, test_paths


    def load_data(self, model_args):
        # global self.args, entity_dict, relation_dict

        self.args = model_args
        directory = '../data/' + self.args.dataset + '/'

        print('reading entity dict and relation dict ...')
        self.entity_dict = self.read_entities(directory + 'entities.dict')
        self.relation_dict = self.read_relations(directory + 'relations.dict')

        print('reading train, validation, and test data ...')
        train_triplets = self.read_triplets(directory + 'train.txt')
        valid_triplets = self.read_triplets(directory + 'valid.txt')
        test_triplets = self.read_triplets(directory + 'test.txt')

        print('processing the static knowledge graph ...')
        self.build_kg(train_triplets)

        triplets = [train_triplets, valid_triplets, test_triplets]

        if self.args.use_context:
            neighbor_params = [np.array(self.entity2edges), np.array(self.edge2entities), np.array(self.edge2relation)]
        else:
            neighbor_params = None

        if self.args.use_path:
            train_paths, valid_paths, test_paths = self.get_paths(train_triplets, valid_triplets, test_triplets)
            path2id, id2path, id2length = get_path_dict_and_length(
                train_paths, valid_paths, test_paths, len(self.relation_dict), self.args.max_path_len)

            if self.args.path_type == 'embedding':
                print('transforming paths to one hot IDs ...')
                paths = one_hot_path_id(train_paths, valid_paths, test_paths, path2id)
                path_params = [len(path2id)]
            elif self.args.path_type == 'rnn':
                paths = sample_paths(train_paths, valid_paths, test_paths, path2id, self.args.path_samples)
                path_params = [id2path, id2length]
            else:
                raise ValueError('unknown path type')
        else:
            paths = [None] * 3
            path_params = None

        return triplets, paths, len(self.relation_dict), neighbor_params, path_params

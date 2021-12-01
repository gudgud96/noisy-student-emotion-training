import os
import numpy as np
import pickle
from torch.utils import data
import torch
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import TimeMasking, FrequencyMasking
import random

spec_log_max = 9.6

class AudioFolder(data.Dataset):
    def __init__(self, root, root_hpcp, subset, tr_val='train', split=0, is_test_mode=False):
        self.trval = tr_val
        self.root = root
        self.root_hpcp = "E://mtg-jamendo-hpcp//"
        fn = '../../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)
        self.is_test_mode = is_test_mode

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.dictionary[index]['path'][:-3]+'npy')
        fn_hpcp = os.path.join(self.root_hpcp, self.dictionary[index]['path'][:-3]+'npy')
        
        # speed bottleneck is load time
        try:
            melspec = np.load(fn)
            hpcp = np.load(fn_hpcp)
        except Exception:
            return None, None, None, None

        if not self.is_test_mode:
            melspec = melspec[:, :1600].T
        else:
            melspec = melspec.T
        melspec = np.log1p(melspec)
        melspec = melspec / spec_log_max
        melspec = torch.tensor(melspec)
        new_melspec = []
        for i in range(0, melspec.shape[0], 80):
            if melspec[i:i+80, :].shape[0] == 80:
                new_melspec.append(melspec[i:i+80, :])
        new_melspec = torch.stack(new_melspec, dim=0)

        if not self.is_test_mode:
            hpcp = hpcp[:4000, :]
        hpcp = torch.tensor(hpcp)
        new_hpcp = []
        for i in range(0, hpcp.shape[0], 200):
            if hpcp[i:i+200, :].shape[0] == 200:
                new_hpcp.append(hpcp[i:i+200, :])
        new_hpcp = torch.stack(new_hpcp, dim=0)
             
        tags = self.dictionary[index]['tags']
        tags = torch.tensor(tags.astype('float32'))
        tags = torch.stack([tags for _ in range(min(new_hpcp.shape[0], new_melspec.shape[0]))], dim=0)
        
        if new_hpcp.shape[0] != new_melspec.shape[0]:
            if new_hpcp.shape[0] < new_melspec.shape[0]:
                new_melspec = new_melspec[:new_hpcp.shape[0]]
            else:
                new_hpcp = new_hpcp[:new_melspec.shape[0]]

        return new_melspec, new_hpcp, tags, None

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

        if "all" in fn:     # need to filter out labelled data
            fn_labelled = '../../data/splits/split-%d/%s_%s_dict.pickle' % (0, "moodtheme", "train")    # hardcode for now
            with open(fn_labelled, 'rb') as pf:
                dictionary_labelled = pickle.load(pf)
            labelled_paths = [dictionary_labelled[key]['path'] for key in dictionary_labelled]

            labelled_keys = []
            for key in self.dictionary:
                if self.dictionary[key]['path'] in labelled_paths:
                    labelled_keys.append(key)
            
            for key in labelled_keys:
                del self.dictionary[key]
            
            # rearrange index by sorted consecutive integers
            new_dictionary = {}
            cur_keys = sorted(list(self.dictionary.keys()))
            for i in range(len(cur_keys)):
                new_dictionary[i] = self.dictionary[cur_keys[i]]
            self.dictionary = new_dictionary

    def __len__(self):
        return len(self.dictionary)


class AudioFolderFull(data.Dataset):
    def __init__(self, root, root_hpcp, subset, tr_val='train', split=0, is_test_mode=False):
        self.trval = tr_val
        self.root = root
        self.root_hpcp = "C://mtg-jamendo-hpcp-20//"
        fn = '../../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)
        self.is_test_mode = is_test_mode

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.dictionary[index]['path'][:-3]+'npy')
        fn_hpcp = os.path.join(self.root_hpcp, self.dictionary[index]['path'][:-3]+'npy')
        
        try:
            melspec = np.load(fn)
            hpcp = np.load(fn_hpcp)
        except Exception:
             return None, None, None, None

        if not self.is_test_mode:
            melspec = melspec[:, :2000].T
        else:
            melspec = melspec.T
        melspec = np.log1p(melspec)
        melspec = melspec / spec_log_max
        melspec = torch.tensor(melspec)

        if not self.is_test_mode:
            hpcp = hpcp[:500, :]
        hpcp = torch.tensor(hpcp)
        tags = self.dictionary[index]['tags']

        return melspec, hpcp, tags.astype('float32'), None

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

        if "all" in fn:     # need to filter out labelled data
            fn_labelled = '../../data/splits/split-%d/%s_%s_dict.pickle' % (0, "moodtheme", "train")    # hardcode for now
            with open(fn_labelled, 'rb') as pf:
                dictionary_labelled = pickle.load(pf)
            labelled_paths = [dictionary_labelled[key]['path'] for key in dictionary_labelled]

            labelled_keys = []
            for key in self.dictionary:
                if self.dictionary[key]['path'] in labelled_paths:
                    labelled_keys.append(key)
            
            for key in labelled_keys:
                del self.dictionary[key]
            
            # rearrange index by sorted consecutive integers
            new_dictionary = {}
            cur_keys = sorted(list(self.dictionary.keys()))
            for i in range(len(cur_keys)):
                new_dictionary[i] = self.dictionary[cur_keys[i]]
            self.dictionary = new_dictionary

    def __len__(self):
        return len(self.dictionary)


class AudioFolderAugment(AudioFolder):
    def __init__(self, root, root_hpcp, subset, tr_val='train', split=0, prob=1, is_test_mode=False):
        self.audio_folder = AudioFolder(root, root_hpcp, subset, tr_val, split, is_test_mode=is_test_mode)
        self.prob = prob

    def __getitem__(self, index):
        melspec, hpcp, tag, _ = self.audio_folder[index]

        if melspec is not None:
            # do augmentation            
            new_melspec = []
            for l in range(len(melspec)):
                cur_melspec = melspec[l]
                time_mask = TimeMasking(random.randint(20, 60))
                cur_melspec = time_mask(cur_melspec)
                freq_mask = FrequencyMasking(random.randint(20, 60))
                cur_melspec = freq_mask(cur_melspec)
        
                # gaussian noise
                noise = torch.normal(0, 1, cur_melspec.size())
                cur_melspec += 0.01 * noise
                new_melspec.append(cur_melspec)
            new_melspec = torch.stack(new_melspec, dim=0)
        
        return melspec, hpcp, tag

    def __len__(self):
        return len(self.audio_folder)


class AudioFolderAugmentFull(AudioFolder):
    def __init__(self, root, root_hpcp, subset, tr_val='train', split=0, prob=1, is_test_mode=False):
        self.audio_folder = AudioFolderFull(root, root_hpcp, subset, tr_val, split, is_test_mode=is_test_mode)
        self.prob = prob

    def __getitem__(self, index):
        melspec, hpcp, tag, _ = self.audio_folder[index]

        if melspec is not None:
            # do augmentation
            time_mask = TimeMasking(random.randint(20, 60))
            melspec = time_mask(melspec)
            freq_mask = FrequencyMasking(random.randint(20, 60))
            melspec = freq_mask(melspec)
        
            # gaussian noise
            noise = torch.normal(0, 1, melspec.size())
            melspec += 0.01 * noise
        
        return melspec, hpcp, tag

    def __len__(self):
        return len(self.audio_folder)


def my_collate_hpcp(batch):
    f1 = torch.cat([k[0] for k in batch if k[0] is not None])
    f2 = torch.cat([k[1] for k in batch if k[1] is not None])
    f3 = torch.cat([k[2] for k in batch if k[2] is not None], dim=0)
    return torch.transpose(f1, 1, 2), torch.transpose(f2, 1, 2), f3, None


def my_collate_hpcp_full(batch):
    f1 = [k[0] for k in batch if k[0] is not None]
    f2 = [k[1] for k in batch if k[1] is not None]
    f3 = torch.tensor([k[2] for k in batch if k[2] is not None])
    f1 = pad_sequence(f1, batch_first=True)
    f2 = pad_sequence(f2, batch_first=True)
    return torch.transpose(f1, 1, 2), torch.transpose(f2, 1, 2), f3, None


def get_audio_loader(root, root_hpcp, subset, batch_size, tr_val='train', split=0, num_workers=0, shuffle=True, 
                     is_test_mode=False):
    collate_fn = my_collate_hpcp
    data_loader = data.DataLoader(dataset=AudioFolder(root, root_hpcp, subset, tr_val, split, is_test_mode=is_test_mode),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers)
    return data_loader


def get_audio_loader_full(root, root_hpcp, subset, batch_size, tr_val='train', split=0, num_workers=0, shuffle=True,
                          is_test_mode=False):
    collate_fn = my_collate_hpcp_full
    data_loader = data.DataLoader(dataset=AudioFolderFull(root, root_hpcp, subset, tr_val, split, is_test_mode=is_test_mode),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers)
    return data_loader


def get_audio_loader_augment(root, root_hpcp, subset, batch_size, tr_val='train', split=0, num_workers=0, prob=1, shuffle=True,
                             is_test_mode=False):
    collate_fn = my_collate_hpcp
    data_loader = data.DataLoader(dataset=AudioFolderAugment(root, root_hpcp, subset, tr_val, split, prob, is_test_mode=is_test_mode),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers)
    return data_loader


def get_audio_loader_augment_full(root, root_hpcp, subset, batch_size, tr_val='train', split=0, num_workers=0, prob=1, shuffle=True,
                                 is_test_mode=False):
    collate_fn = my_collate_hpcp_full
    data_loader = data.DataLoader(dataset=AudioFolderAugmentFull(root, root_hpcp, subset, tr_val, split, prob, is_test_mode=is_test_mode),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers)
    return data_loader
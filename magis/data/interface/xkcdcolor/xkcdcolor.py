"""
XKCD Color Corpus Interface
(formermly known as Munroe Color Corpus)
Original (http://blog.xkcd.com/2010/05/03/color-survey-results/) was hand curated according to
McMahan & Stone. A Bayesian Model of Grounded Color Semantics. TACL. 2015.

Dataset abstract interface defined in eidos, so it is required.
"""

import os
HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import ujson as json
except:
    import json

from tqdm import tqdm
import numpy as np
import eidos
from eidos import GenericIndex as Gendex


class Dataset(eidos.Dataset):
    def __init__(self, coordinator=None):
        self.name = "xkcd"
        if coordinator:
            filer = coordinator.filer
            self.mgr = eidos.Manager.from_file(filer.xkcd_manager, 
                                               coordinator.filer,
                                               updated_category="xkcd_cache", 
                                               prefix=self.name)
            self.index = self.mgr.index
            self.loaded = eidos.utils.fix_keys(self.mgr.all_data)
        else:    
            with open(os.path.join(HERE, 'corpusindex.json')) as fp:
                self.index = json.load(fp)  
            self.loaded = {}
        
        self._name2index = {k:i for i,k in enumerate(self.index.keys())}
        self._index2name = {i:k for k,i in self._name2index.items()}
        self._active_df = None

    ############ properties 

    @property
    def training_size(self):
        if self._active_df is None:
            return 0
        return self._active_df.n.train

    @property
    def development_size(self):
        if self._active_df is None:
            return 0
        return self._active_df.n.train

    @property
    def testing_size(self):
        if self._active_df is None:
            return 0
        return self._active_df.n.train

    @property
    def number_words(self):
        return len(self._name2index)

    @property
    def description(self):
        a = "{} color names".format(self.number_words)
        b = "{} train; {} dev; {} test".format(self.training_size, 
                                               self.development_size, 
                                               self.testing_size)
        return "[Dataset][MunroeCorpus][{}][{}]".format(a, b)

    ############ methods

    def name2index(self, name):
        return self._name2index[name]

    def index2name(self, index):
        return self._index2name[index]


    def load(self, form='raw'):
        if not hasattr(self, 'mgr'):
            self.load_all()
            self.convert_all()
        self.make_datasets(form)
        self.forevers = eidos.GenericIndex({split:self.generate_forever(split) 
                                            for split in ('train', 'dev', 'test')})

    def generate_once(self, split='train'):
        ''' randomly iterate over all data points once '''
        ## get indices from names
        n2i = self._name2index
        ## total count for split
        n = self._active_df.n[split]
        ## the actual data
        matrix = self._active_df.mats[split]
        ## compute the true label orderings to match the rows in the matrix
        name_info = sorted(self._active_df.sparsemap[split].items(), key=lambda x: x[1][0])
        name_idx = np.array([n2i[r_name] for name, (_, m) in name_info 
                                         for r_name in [name]*m])
        ## get some random numbers
        indices = np.random.choice(np.arange(n), size=n, replace=False)
        ## now iterate over them and yield it out
        for idx in indices:
            yield matrix[idx], name_idx[idx]
            
            
    def generate_forever(self, split='train'):
        while True:
            for x, y in self.generate_once(split):
                yield x, y


    #######################  not in interface


    def load_all(self):
        for name, splits in tqdm(self.index.items(), desc='loading data'):
            splits.pop('name', '') # should have taken this out anyway =D
            item = {}
            for split, filename in splits.items():
                with open(os.path.join(HERE,filename)) as fp:
                     datum = json.load(fp)
                     item['raw', split] = np.array([d['raw'] for d in datum])
                     item['scaled', split] = np.array([d['scaled'] for d in datum])

            self.loaded[name] = item

    def convert_all(self):
        def convert_one(datum):
            datum_rads = datum * np.pi / 180.0
            datum_conv = 180.0 / np.pi * np.arctan2(np.sin(datum_rads), 
                                                     np.cos(datum_rads))
            return datum_conv

        for name, datums in tqdm(self.loaded.items(), 'converting data'):
            litmus = datums['scaled', 'dev'][:,0]
            litmus_conv = convert_one(litmus)
            
            if litmus.std() > litmus_conv.std():
                for (form,split), datum in datums.items():
                    scale = 1.
                    if form == 'raw':
                        scale = 360.
                    datum[:,0] = convert_one(datum[:,0]*scale) / scale


    def make_datasets(self, form='raw'):
        splits = ('train', 'dev', 'test')
        self.current_form = form

        mats = Gendex({k:np.empty((0,3)) for k in splits})
        sparsemap = Gendex({k:{} for k in splits})
        n = Gendex({k:0 for k in splits})

        for name, datums in self.loaded.items():
            for split in splits:
                D = datums[form, split]
                sparsemap[split][name] = (n[split], len(D))
                n[split] += len(D)
                mats[split] = np.concatenate((mats[split], D), axis=0)
        self._active_df = Gendex({'n': n, 'sparsemap': sparsemap, 'mats': mats})



    def cache(self, filer):
        name = "xkcd"
        mname = "{}_manager".format(name)
        cname = "{}_cache".format(name)
        filer.add_category(cname, os.path.join(filer.cache, cname))
        filer.track(mname, cname)
        mgr = eidos.Manager(filer, prefix=name)
        mgr.manage("all_data", self.loaded, "json")
        mgr.manage("index", self.index, "json")
        mgr.save()





'''
need to make the following datasets:


2. raw, adjusted
4. scaled, adjusted
5. x,y coords raw
6. x,y coords scaled
7. fourier? 

everything needs to be input, and then converted to numpy

it's too early for pandas. do pandas after we select the tensors. 

but how do we compute the tensors? 

i know how we can save them: manager. 

keep sep by name only in so far as it neesd to be split.

a single name should have raw, scaled, and both unadjusted


'''
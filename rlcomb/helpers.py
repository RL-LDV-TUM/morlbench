'''
Created on Sep 14, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
This module contains miscellaneous helper
functions that didn't fit in any other place.
'''

import cPickle as pickle
from os.path import isfile


def virtualFunction():
    raise RuntimeError('Virtual function not implemented.')


class SaveableObject(object):
    '''
    This should be a generic class for custom object
    storage, where you may not want to store all
    data associated with it, but only specific parts
    of the __dict__ of the object.
    '''

    def __init__(self, keys, **kwargs):
        '''
        Parameters
        ----------
        keys: a list of __dict__ entries that
            should be stored, when the object
            is pickled.
        '''
        self._saveable_keys_ = keys

#    def __getstate__(self):
#        state = {}
#        for k in self._saveable_keys_:
#            state[k] = self.__dict__[k]
#
#        state['_saveable_keys_'] = self.__dict__['_saveable_keys_']
#        return state
#
#    def __setstate__(self, state):
#        for k, v in state.items():
#            self.__dict__[k] = v

    def save(self, filename, overwrite=False):
        '''
        Save the object to disk using cPickle.
        '''
        if isfile(filename) and not overwrite:
            raise RuntimeError('file %s already exists.' % (filename))
        f = open(filename, 'wb')
        if not f:
            raise RuntimeError('could not open %s' % (filename))
        state = {}
        for k in self._saveable_keys_:
            state[k] = self.__dict__[k]
        state['_saveable_keys_'] = self.__dict__['_saveable_keys_']
        pickle.dump(state, f)
        f.close()

    def load(self, filename):
        '''
        Load pickled object from file.
        '''
        if not isfile(filename):
            raise RuntimeError('file %s does not exist' % (filename))
        f = open(filename, 'rb')
        if not f:
            raise RuntimeError('could not open %s' % (filename))
        state = pickle.load(f)
        f.close()
        self.__dict__.update(state)\

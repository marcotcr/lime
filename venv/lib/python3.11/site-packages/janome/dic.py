# Copyright 2015 moco_beta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import os
import io
import pickle
import gzip
from struct import pack, unpack
import traceback
import logging
import sys
import re
import pkgutil
import zlib
import base64
from functools import lru_cache
from .fst import Matcher, create_minimum_transducer, compileFST

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
handler = logging.StreamHandler()
handler.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s\t%(name)s - %(levelname)s\t%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

MODULE_FST_DATA = 'fst_data%d.py'
MODULE_ENTRIES_EXTRA = 'entries_extra%d.py'
MODULE_ENTRIES_COMPACT = 'entries_compact%d.py'
MODULE_ENTRIES_BUCKETS = 'entries_buckets.py'
MODULE_CONNECTIONS = 'connections%d.py'
MODULE_CHARDEFS = 'chardef.py'
MODULE_UNKNOWNS = 'unknowns.py'

FILE_USER_FST_DATA = 'user_fst.data'
FILE_USER_ENTRIES_DATA = 'user_entries.data'


def save_fstdata(data, dir, part=0):
    _save_as_module(os.path.join(dir, MODULE_FST_DATA % part), data, binary=True)


def start_save_entries(dir, bucket_idx, morph_offset):
    _start_entries_as_module(os.path.join(dir, MODULE_ENTRIES_COMPACT % bucket_idx), morph_offset)
    _start_entries_as_module(os.path.join(dir, MODULE_ENTRIES_EXTRA % bucket_idx), morph_offset)


def end_save_entries(dir, bucket_idx):
    _end_entries_as_module(os.path.join(dir, MODULE_ENTRIES_COMPACT % bucket_idx))
    _end_entries_as_module(os.path.join(dir, MODULE_ENTRIES_EXTRA % bucket_idx))


def save_entry(dir, bucket_idx, morph_id, entry):
    _save_entry_as_module_compact(os.path.join(dir, MODULE_ENTRIES_COMPACT % bucket_idx), morph_id, entry)
    _save_entry_as_module_extra(os.path.join(dir, MODULE_ENTRIES_EXTRA % bucket_idx), morph_id, entry)


def save_entry_buckets(dir, buckets):
    _save_as_module(os.path.join(dir, MODULE_ENTRIES_BUCKETS), buckets)


def save_connections(connections, dir='.'):
    # split whole connections to 2 buckets to reduce memory usage while installing.
    # TODO: find better ways...
    bucket_size = (len(connections) // 2) + 1
    offset = 0
    for i in range(1, 3):
        _save_as_module(os.path.join(dir, MODULE_CONNECTIONS % i),
                        connections[offset:offset + bucket_size])
        offset += bucket_size


def save_chardefs(chardefs, dir='.'):
    _save_as_module(os.path.join(dir, MODULE_CHARDEFS), chardefs)


def save_unknowns(unknowns, dir='.'):
    _save_as_module(os.path.join(dir, MODULE_UNKNOWNS), unknowns)


def _save(file, data, compresslevel):
    if not data:
        return
    with gzip.open(file, 'wb', compresslevel) as f:
        f.write(data)
        f.flush()


def _load(file):
    if not os.path.exists(file):
        return None
    with gzip.open(file, 'rb') as f:
        data = f.read()
        return data


def _load_package_data(package, resource):
    try:
        rawdata = pkgutil.get_data(package, resource)
    except IOError:
        return None
    return zlib.decompress(rawdata, zlib.MAX_WBITS | 16)


def _save_as_module(file, data, binary=False):
    if not data:
        return
    with open(file, 'w') as f:
        f.write('DATA=')
        if binary:
            f.write('"')
            f.write(base64.b64encode(data).decode('ascii'))
            f.write('"')
        else:
            f.write(str(data).replace('\\\\', '\\'))
        f.flush()


def _start_entries_as_module(file, morph_id_offset):
    idx_file = re.sub(r'\.py$', '_idx.py', file)
    with open(file, 'w') as f:
        with open(idx_file, 'w') as f_idx:
            f.write('DATA={')
            f_idx.write('DATA={')
            f_idx.write(f'"offset": {morph_id_offset}, "positions": [')


def _end_entries_as_module(file):
    idx_file = re.sub(r'\.py$', '_idx.py', file)
    with open(file, 'a') as f:
        with open(idx_file, 'a') as f_idx:
            f.write('}\n')
            f_idx.write(']}\n')
            f.flush()
            f_idx.flush()


def _save_entry_as_module_compact(file, morph_id, entry):
    idx_file = re.sub(r'\.py$', '_idx.py', file)
    with open(file, 'a') as f:
        with open(idx_file, 'a') as f_idx:
            f.write('%d:(' % morph_id)
            pos = f.tell()
            f_idx.write(f'{pos},')
            s = u"u'%s',%4d,%4d,%5d" % (
                entry[0].encode('unicode_escape').decode('ascii'),
                entry[1],
                entry[2],
                entry[3])
            f.write(s)
            f.write('),')


def _save_entry_as_module_extra(file, morph_id, entry):
    idx_file = re.sub(r'\.py$', '_idx.py', file)
    with open(file, 'a') as f:
        with open(idx_file, 'a') as f_idx:
            f.write('%d:(' % morph_id)
            pos = f.tell()
            f_idx.write(f'{pos},')
            s = u"u'%s',u'%s',u'%s',u'%s',u'%s',u'%s'" % (
                entry[4].encode('unicode_escape').decode('ascii'),
                entry[5].encode('unicode_escape').decode('ascii'),
                entry[6].encode('unicode_escape').decode('ascii'),
                entry[7].encode('unicode_escape').decode('ascii'),
                entry[8].encode('unicode_escape').decode('ascii'),
                entry[9].encode('unicode_escape').decode('ascii'))
            f.write(s)
            f.write('),')


class Dictionary(ABC):
    """
    Base dictionary class
    """

    @abstractmethod
    def lookup(self, s, matcher):
        pass

    @abstractmethod
    def lookup_extra(self, num):
        pass

    @abstractmethod
    def get_trans_cost(self, id1, id2):
        pass


class RAMDictionary(Dictionary):
    """
    RAM dictionary class
    """

    def __init__(self, entries, connections):
        self.entries = entries
        self.connections = connections

    def lookup(self, s, matcher):
        (matched, outputs) = matcher.run(s)
        if not matched:
            return []
        try:
            res = []
            for e in outputs:
                num = unpack('I', e)[0]
                res.append((num,) + self.entries[num][:4])
            return res
        except Exception:
            logger.error('Cannot load dictionary data. The dictionary may be corrupted?')
            logger.error(f'input={s}')
            logger.error(f'outputs={str(outputs)}')
            traceback.format_exc()
            sys.exit(1)

    def lookup_extra(self, num):
        try:
            return self.entries[num][4:]
        except Exception:
            logger.error('Cannot load dictionary data. The dictionary may be corrupted?')
            traceback.format_exc()
            sys.exit(1)

    def get_trans_cost(self, id1, id2):
        return self.connections[id1][id2]


class MMapDictionary(Dictionary):
    """
    MMap dictionary class
    """

    def __init__(self, entries_compact, entries_extra, open_files, connections):
        self.entries_compact = entries_compact
        self.bucket_ranges = entries_compact.keys()
        self.entries_extra = entries_extra
        self.open_files = open_files
        self.connections = connections

    def lookup(self, s, matcher):
        (matched, outputs) = matcher.run(s)
        if not matched:
            return []
        try:
            matched_entries = []
            for e in outputs:
                idx = unpack('I', e)[0]
                matched_entries.append((idx,) + self._find_entry(idx))
            return matched_entries
        except Exception:
            logger.error('Cannot load dictionary data. The dictionary may be corrupted?')
            logger.error(f'input={s}')
            logger.error(f'outputs={str(outputs)}')
            traceback.format_exc()
            sys.exit(1)

    @lru_cache(maxsize=8192)
    def _find_entry(self, idx):
        bucket = next(filter(lambda b: idx >= b[0] and idx < b[1], self.bucket_ranges))
        mm, mm_idx = self.entries_compact[bucket]
        rel_idx = idx - mm_idx['offset']
        _pos1s = mm_idx['positions'][rel_idx] + 2
        _pos1e = mm.find(b"',", _pos1s)
        _pos2s = _pos1e + 2
        _pos2e = _pos2s + 4
        _pos3s = _pos2e + 1
        _pos3e = _pos3s + 4
        _pos4s = _pos3e + 1
        _pos4e = _pos4s + 5
        _entry = (
            mm[_pos1s:_pos1e].decode('unicode_escape'),
            int(mm[_pos2s:_pos2e]),
            int(mm[_pos3s:_pos3e]),
            int(mm[_pos4s:_pos4e]))
        return _entry

    @lru_cache(maxsize=1024)
    def lookup_extra(self, idx):
        try:
            bucket = next(filter(lambda b: idx >= b[0] and idx < b[1], self.bucket_ranges))
            mm, mm_idx = self.entries_extra[bucket]
            rel_idx = idx - mm_idx['offset']
            _pos1s = mm_idx['positions'][rel_idx] + 2
            _pos1e = mm.find(b"',u'", _pos1s)
            _pos2s = _pos1e + 4
            _pos2e = mm.find(b"',u'", _pos2s)
            _pos3s = _pos2e + 4
            _pos3e = mm.find(b"',u'", _pos3s)
            _pos4s = _pos3e + 4
            _pos4e = mm.find(b"',u'", _pos4s)
            _pos5s = _pos4e + 4
            _pos5e = mm.find(b"',u'", _pos5s)
            _pos6s = _pos5e + 4
            _pos6e = mm.find(b"')", _pos6s)
            return (
                mm[_pos1s:_pos1e].decode('unicode_escape'), mm[_pos2s:_pos2e].decode(
                    'unicode_escape'), mm[_pos3s:_pos3e].decode('unicode_escape'),
                mm[_pos4s:_pos4e].decode('unicode_escape'), mm[_pos5s:_pos5e].decode(
                    'unicode_escape'), mm[_pos6s:_pos6e].decode('unicode_escape')
            )
        except Exception:
            logger.error('Cannot load extra info. The dictionary may be corrupted?')
            logger.error(f'idx={idx}')
            traceback.format_exc()
            sys.exit(1)

    def get_trans_cost(self, id1, id2):
        return self.connections[id1][id2]

    def __del__(self):
        for mm, mm_idx in self.entries_compact.values():
            mm.close()
        if self.entries_extra:
            for mm, mm_idx in self.entries_extra.values():
                mm.close()
        for fp in self.open_files:
            fp.close()


class UnknownsDictionary(object):
    """
    Dictionary class for handling unknown words
    """

    def __init__(self, chardefs, unknowns):
        self.char_categories = chardefs[0]
        self.char_ranges = chardefs[1]
        self.unknowns = unknowns

    @lru_cache(maxsize=1024)
    def get_char_categories(self, c):
        res = {}
        for chr_range in self.char_ranges:
            if chr_range['from'] <= c <= chr_range['to']:
                cate = chr_range['cate']
                compate_cates = chr_range['compat_cates'] if 'compat_cates' in chr_range else []
                res[cate] = compate_cates
        if not res:
            res = {'DEFAULT': []}
        return res

    def unknown_invoked_always(self, cate):
        if cate in self.char_categories:
            return self.char_categories[cate]['INVOKE']
        return False

    def unknown_grouping(self, cate):
        if cate in self.char_categories:
            return self.char_categories[cate]['GROUP']
        return False

    def unknown_length(self, cate):
        if cate in self.char_categories:
            return self.char_categories[cate]['LENGTH']
        return -1


class UserDictionary(RAMDictionary):
    """
    User dictionary class (on-the-fly)
    """

    def __init__(self, user_dict, enc, type, connections, progress_handler=None):
        """
        Initialize user defined dictionary object.

        :param user_dict: user dictionary file (CSV format)
        :param enc: character encoding
        :param type: user dictionary type. supported types are 'ipadic' and 'simpledic'
        :param connections: connection cost matrix. expected value is SYS_DIC.connections
        :param progress_handler: handler mainly to indicate progress, implementation of ProgressHandler

        .. seealso:: http://mocobeta.github.io/janome/en/#use-with-user-defined-dictionary
        """
        fst_data, entries = UserDictionary.build_dic(user_dict, enc, type, progress_handler)
        super().__init__(entries, connections)
        self.compiledFST = [fst_data]
        self.matcher = Matcher([fst_data])

    def lookup(self, s):
        return super().lookup(s, self.matcher)

    @classmethod
    def line_to_entry_ipadic(cls, line):
        """Convert IPADIC formatted string to an user dictionary entry"""
        surface, left_id, right_id, cost, \
            pos_major, pos_minor1, pos_minor2, pos_minor3, \
            infl_type, infl_form, base_form, reading, phonetic = line.split(',')
        part_of_speech = ','.join([pos_major, pos_minor1, pos_minor2, pos_minor3])
        return (surface, int(left_id), int(right_id), int(cost),
                part_of_speech, infl_type, infl_form, base_form, reading, phonetic)

    @classmethod
    def line_to_entry_simpledic(cls, line):
        """Convert simpledict formatted string to an user dictionary entry"""
        surface, pos_major, reading = line.split(',')
        part_of_speech = ','.join([pos_major, '*', '*', '*'])
        return (surface, 0, 0, -100000, part_of_speech, '*', '*', surface, reading, reading)

    @classmethod
    def build_dic(cls, user_dict, enc, dict_type, progress_handler):
        surfaces = []
        entries = {}

        line_to_entry = getattr(cls, 'line_to_entry_' + dict_type)
        # init progress for reading CSV
        if progress_handler:
            with open(user_dict, encoding=enc) as f:
                progress_handler.on_start(
                    total=sum(1 for line in f),
                    desc='Reading user dictionary from CSV')

        with io.open(user_dict, encoding=enc) as f:
            for line in f:
                line = line.rstrip()
                # entry should be a tuple:
                # (surface, left_id, right_id, cost, part_of_speech, infl_type, infl_form, base_form, reading, phonetic)
                entry = line_to_entry(line)
                morph_id = len(surfaces)
                surfaces.append((entry[0].encode('utf8'), pack('I', morph_id)))
                entries[morph_id] = entry

                # update progress
                if progress_handler:
                    progress_handler.on_progress()

        # complete progress for reading CSV
        if progress_handler:
            progress_handler.on_complete()

        inputs = sorted(surfaces)  # inputs must be sorted.
        assert len(surfaces) == len(entries)

        # init progress for create_minimum_transducer
        if progress_handler:
            progress_handler.on_start(
                total=len(inputs),
                desc='Running create_minimum_transducer')

        processed, fst = create_minimum_transducer(
            inputs,
            on_progress=progress_handler.on_progress if progress_handler else None)

        # complete progress for create_minimum_transducer
        if progress_handler:
            progress_handler.on_complete()

        compiledFST = compileFST(fst)
        return compiledFST, entries

    def save(self, to_dir, compressionlevel=9):
        """
        Save compressed compiled dictionary data.

        :param to_dir: directory to save dictionary data
        :compressionlevel: (Optional) gzip compression level. default is 9
        """
        if os.path.exists(to_dir) and not os.path.isdir(to_dir):
            raise Exception(f'Not a directory : {to_dir}')
        elif not os.path.exists(to_dir):
            os.makedirs(to_dir, mode=int('0755', 8))
        _save(os.path.join(to_dir, FILE_USER_FST_DATA), self.compiledFST[0], compressionlevel)
        _save(os.path.join(to_dir, FILE_USER_ENTRIES_DATA), pickle.dumps(self.entries), compressionlevel)


class CompiledUserDictionary(RAMDictionary):
    """
    User dictionary class (compiled)
    """

    def __init__(self, dic_dir, connections):
        fst_data, entries = CompiledUserDictionary.load_dict(dic_dir)
        super().__init__(entries, connections)
        self.matcher = Matcher([fst_data])

    def lookup(self, s):
        return super().lookup(s, self.matcher)

    @classmethod
    def load_dict(cls, dic_dir):
        if not os.path.exists(dic_dir) or not os.path.isdir(dic_dir):
            raise Exception(f'No such directory : {dic_dir}')
        data = _load(os.path.join(dic_dir, FILE_USER_FST_DATA))
        entries = pickle.loads(_load(os.path.join(dic_dir, FILE_USER_ENTRIES_DATA)))
        return data, entries


class LoadingDictionaryError(Exception):
    def __init__(self):
        self.message = 'Cannot load dictionary data. Try mmap mode for very large dictionary.'

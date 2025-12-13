# type: ignore
import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))

from janome.dic import LoadingDictionaryError
from . import connections1, connections2
from . import chardef, unknowns

connections = list(connections1.DATA)
connections.extend(connections2.DATA)

__entries = None

def __add_extra_info(entries, extra_entries_info):
    for k, v in extra_entries_info.items():
        entries[k] = tuple(list(entries[k]) + list(v))

def entries(compact = False):
    global __entries
    if not __entries:
        try:
            from . import entries_compact0, entries_compact1, entries_compact2, entries_compact3, entries_compact4, entries_compact5, entries_compact6, entries_compact7, entries_compact8, entries_compact9
        except:
            raise LoadingDictionaryError()
        __entries = entries_compact0.DATA
        __entries.update(entries_compact1.DATA)
        __entries.update(entries_compact2.DATA)
        __entries.update(entries_compact3.DATA)
        __entries.update(entries_compact4.DATA)
        __entries.update(entries_compact5.DATA)
        __entries.update(entries_compact6.DATA)
        __entries.update(entries_compact7.DATA)
        __entries.update(entries_compact8.DATA)
        __entries.update(entries_compact9.DATA)
        del entries_compact0.DATA
        del entries_compact1.DATA
        del entries_compact2.DATA
        del entries_compact3.DATA
        del entries_compact4.DATA
        del entries_compact5.DATA
        del entries_compact6.DATA
        del entries_compact7.DATA
        del entries_compact8.DATA
        del entries_compact9.DATA
    if not compact and len(__entries[0]) < 5:
        # need to load extra token info
        try:
            from . import entries_extra0, entries_extra1, entries_extra2, entries_extra3, entries_extra4, entries_extra5, entries_extra6, entries_extra7, entries_extra8, entries_extra9
        except:
            raise LoadingDictionaryError()
        __add_extra_info(__entries, entries_extra0.DATA)
        __add_extra_info(__entries, entries_extra1.DATA)
        __add_extra_info(__entries, entries_extra2.DATA)
        __add_extra_info(__entries, entries_extra3.DATA)
        __add_extra_info(__entries, entries_extra4.DATA)
        __add_extra_info(__entries, entries_extra5.DATA)
        __add_extra_info(__entries, entries_extra6.DATA)
        __add_extra_info(__entries, entries_extra7.DATA)
        __add_extra_info(__entries, entries_extra8.DATA)
        __add_extra_info(__entries, entries_extra9.DATA)
        del entries_extra0.DATA
        del entries_extra1.DATA
        del entries_extra2.DATA
        del entries_extra3.DATA
        del entries_extra4.DATA
        del entries_extra5.DATA
        del entries_extra6.DATA
        del entries_extra7.DATA
        del entries_extra8.DATA
        del entries_extra9.DATA
    return __entries

def mmap_entries(compact = False):
    import mmap
    from importlib import import_module
    from . import entries_buckets

    __mmap_entries_compact = {}
    __mmap_entries_extra = None
    __open_files = []
    for i in range(0, 10):
        bucket = entries_buckets.DATA[i]
        fp = open(os.path.join(base_dir, 'entries_compact%d.py' % i), 'rb')
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        mm_idx = import_module('.entries_compact%d_idx' % i, 'janome.sysdic')
        __open_files.append(fp)
        __mmap_entries_compact[bucket] = (mm, mm_idx.DATA)
    if not compact:
        __mmap_entries_extra = {}
        for i in range(0, 10):
            bucket = entries_buckets.DATA[i]
            fp = open(os.path.join(base_dir, 'entries_extra%d.py' % i), 'rb')
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            mm_idx = import_module('.entries_extra%d_idx' % i, 'janome.sysdic')
            __open_files.append(fp)
            __mmap_entries_extra[bucket] = (mm, mm_idx.DATA)
    return (__mmap_entries_compact, __mmap_entries_extra, __open_files)

def all_fstdata():
    import base64
    from . import fst_data0,fst_data1
    res=[]
    res.append(base64.b64decode(fst_data0.DATA))
    res.append(base64.b64decode(fst_data1.DATA))
    return res

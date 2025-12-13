# Copyright 2022 moco_beta
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

import threading

from .sysdic import entries, mmap_entries, connections, chardef, unknowns  # type: ignore
from .dic import RAMDictionary, MMapDictionary, UnknownsDictionary


class SystemDictionary(RAMDictionary, UnknownsDictionary):
    """
    System dictionary class
    """

    __INSTANCE = None
    __lock = threading.Lock()

    @classmethod
    def instance(cls):
        if not cls.__INSTANCE:
            with cls.__lock:
                if not cls.__INSTANCE:
                    cls.__INSTANCE = SystemDictionary(entries(), connections, chardef.DATA, unknowns.DATA)
        return cls.__INSTANCE

    def __init__(self, entries, connections, chardefs, unknowns):
        RAMDictionary.__init__(self, entries, connections)
        UnknownsDictionary.__init__(self, chardefs, unknowns)


class MMapSystemDictionary(MMapDictionary, UnknownsDictionary):
    """
    MMap System dictionary class
    """

    __INSTANCE = None
    __lock = threading.Lock()

    @classmethod
    def instance(cls):
        if not cls.__INSTANCE:
            with cls.__lock:
                if not cls.__INSTANCE:
                    cls.__INSTANCE = MMapSystemDictionary(mmap_entries(), connections, chardef.DATA, unknowns.DATA)
        return cls.__INSTANCE

    def __init__(self, mmap_entries, connections, chardefs, unknowns):
        MMapDictionary.__init__(self, mmap_entries[0], mmap_entries[1], mmap_entries[2], connections)
        UnknownsDictionary.__init__(self, chardefs, unknowns)

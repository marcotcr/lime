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

import copy
from struct import pack, unpack
from collections import OrderedDict
import logging
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
handler = logging.StreamHandler()
handler.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s\t%(name)s - %(levelname)s\t%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# bit flags to represent class of arcs
# refer to Apache Lucene FST's implementation
FLAG_FINAL_ARC = 1 << 0             # 1
FLAG_LAST_ARC = 1 << 1              # 2
FLAG_TARGET_NEXT = 1 << 2           # 4  TODO: not used. can be removed?
FLAG_STOP_NODE = 1 << 3             # 8  TODO: not used. can be removed?
FLAG_ARC_HAS_OUTPUT = 1 << 4        # 16
FLAG_ARC_HAS_FINAL_OUTPUT = 1 << 5  # 32

# all characters
CHARS = set()


def set_fst_log_level(level):
    logger.setLevel(level)
    handler.setLevel(level)


class State(object):
    """
    State Class
    """
    __slots__ = ['id', 'final', 'trans_map', 'final_output']

    def __init__(self, id=None):
        self.id = id
        self.final = False
        self.trans_map = {}
        self.final_output = set()

    def is_final(self):
        return self.final

    def set_final(self, final):
        self.final = final

    def transition(self, char):
        if char in self.trans_map:
            return self.trans_map[char]['state']
        else:
            return None

    def set_transition(self, char, state):
        self.trans_map[char] = {'state': state,
                                'output': bytes() if char not in self.trans_map else self.trans_map[char]['output']}

    def state_output(self):
        return self.final_output

    def set_state_output(self, output):
        self.final_output = set([bytes(e) for e in output])

    def clear_state_output(self):
        self.final_output = set()

    def output(self, char):
        if char in self.trans_map:
            return self.trans_map[char]['output']
        else:
            return bytes()

    def set_output(self, char, out):
        if char in self.trans_map:
            self.trans_map[char]['output'] = bytes(out)

    def clear(self):
        self.final = False
        self.trans_map = {}
        self.final_output = set()

    def __eq__(self, other):
        if other is None or not isinstance(other, State):
            return False
        else:
            return \
                self.final == other.final and \
                self.trans_map == other.trans_map and \
                self.final_output == other.final_output

    def __hash__(self):
        return hash(str(self.final) + str(self.trans_map) + str(self.final_output))


def copy_state(src, id):
    state = State(id)
    state.final = src.final
    for c, t in src.trans_map.items():
        state.set_transition(c, copy.copy(t['state']))
        state.set_output(c, t['output'])
    state.final_output = copy.copy(src.final_output)
    return state


class FST(object):
    """
    FST (final dictionary) class
    """
    MAX_SIZE = 300000

    def __init__(self):
        # must preserve inserting order
        self.dictionary = OrderedDict()

    def size(self):
        return len(self.dictionary)

    def member(self, state):
        return self.dictionary.get(hash(state))

    def insert(self, state):
        self.dictionary[hash(state)] = state

    def remove(self, state):
        del self.dictionary[hash(state)]

    def exceed_max_size(self):
        return len(self.dictionary) > FST.MAX_SIZE

    def print_dictionary(self):
        for s in self.dictionary.values():
            for (c, v) in s.trans_map.items():
                print('\t'.join([str(s.id), str(c), str(v['state'].id), str(v['output'])]))
            if s.is_final():
                print('\t'.join([str(s.id), str('final'), str(s.final_output)]))


# naive implementation for building fst
# http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.3698
def create_minimum_transducer(inputs, on_progress=None):
    inputs_size = len(inputs)
    logger.info('(partial) input size: %d' % inputs_size)

    fstDict = FST()
    buffer = []
    # buffer.append(State())  # insert 'initial' state

    # previous word
    prev_word = bytes()

    def find_minimized(state):
        # if an equivalent state exists in the dictionary, use that
        s = fstDict.member(state)
        if s is None:
            # if no equivalent state exists, insert new one and return it
            s = copy_state(state, fstDict.size())
            fstDict.insert(s)
        return s

    def prefix_len(s1, s2):
        # calculate max common prefix length for s1 and s2
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i

    current_word = bytes()
    current_output = bytes()
    processed = 0
    # main loop
    for current_word, current_output in inputs:
        assert(current_word >= prev_word)

        pref_len = prefix_len(prev_word, current_word)

        for c in current_word:
            CHARS.add(c)

        # expand buffer to current word length
        while len(buffer) <= len(current_word):
            buffer.append(State())

        # set state transitions
        for i in range(len(prev_word), pref_len, -1):
            buffer[i - 1].set_transition(prev_word[i - 1], find_minimized(buffer[i]))
        for i in range(pref_len + 1, len(current_word) + 1):
            buffer[i].clear()
            buffer[i - 1].set_transition(current_word[i - 1], buffer[i])
        if current_word != prev_word:
            buffer[len(current_word)].set_final(True)
            buffer[len(current_word)].set_state_output(set([bytes()]))

        # set state outputs
        for j in range(1, pref_len + 1):
            # divide (j-1)th state's output to (common) prefix and suffix
            common_prefix = []
            output = buffer[j - 1].output(current_word[j - 1])
            k = 0
            while k < len(output) and k < len(current_output) and output[k] == current_output[k]:
                common_prefix.append(output[k])
                k += 1
            word_suffix = output[len(common_prefix):]

            # re-set (j-1)'th state's output to prefix
            buffer[j - 1].set_output(current_word[j - 1], common_prefix)

            # re-set jth state's output to suffix or set final state output
            for c in CHARS:
                # TODO: optimize this
                if buffer[j].transition(c) is not None:
                    new_output = word_suffix + buffer[j].output(c)
                    buffer[j].set_output(c, new_output)
            # or, set final state output if it's a final state
            if buffer[j].is_final():
                tmp_set = set()
                for tmp_str in buffer[j].state_output():
                    tmp_set.add(word_suffix + tmp_str)
                buffer[j].set_state_output(tmp_set)

            # update current output (subtract prefix)
            current_output = current_output[len(common_prefix):]

        if current_word == prev_word:
            buffer[len(current_word)].state_output().add(current_output)
        else:
            buffer[pref_len].set_output(current_word[pref_len], current_output)

        # preserve current word for next loop
        prev_word = current_word

        processed += 1

        if on_progress:
            on_progress()

    # minimize the last word
    for i in range(len(current_word), 0, -1):
        buffer[i - 1].set_transition(prev_word[i - 1], find_minimized(buffer[i]))
    find_minimized(buffer[0])

    logger.debug(f'num of state: {fstDict.size()}')
    return (processed, fstDict)


def compileFST(fst):
    """
    convert FST to byte array representing arcs
    """
    arcs = []
    address = {}
    pos = 0
    for (num, s) in enumerate(fst.dictionary.values()):
        for i, (c, v) in enumerate(sorted(s.trans_map.items(), reverse=True)):
            bary = bytearray()
            flag = 0
            output_size, output = 0, bytes()
            if i == 0:
                flag += FLAG_LAST_ARC
            if v['output']:
                flag += FLAG_ARC_HAS_OUTPUT
                output_size = len(v['output'])
                output = v['output']
            # encode flag, label, output_size, output, relative target address
            bary += pack('b', flag)
            bary += pack('B', c)
            if output_size > 0:
                bary += pack('I', output_size)
                bary += output
            next_addr = address.get(v['state'].id)
            assert next_addr is not None
            target = (pos + len(bary) + 4) - next_addr
            assert target > 0
            bary += pack('I', target)
            # add the arc represented in bytes
            arcs.append(bytes(bary))
            # address count up
            pos += len(bary)
        if s.is_final():
            bary = bytearray()
            # final state
            flag = FLAG_FINAL_ARC
            output_count = 0
            if s.final_output and any(len(e) > 0 for e in s.final_output):
                # the arc has final output
                flag += FLAG_ARC_HAS_FINAL_OUTPUT
                output_count = len(s.final_output)
            if not s.trans_map:
                flag += FLAG_LAST_ARC
            # encode flag, output size, output
            bary += pack('b', flag)
            if output_count:
                bary += pack('I', output_count)
                for out in s.final_output:
                    output_size = len(out)
                    bary += pack('I', output_size)
                    if output_size:
                        bary += out
            # add the arc represented in bytes
            arcs.append(bytes(bary))
            # address count up
            pos += len(bary)
        address[s.id] = pos

    logger.debug(f'compiled arcs size: {len(arcs)}')
    arcs.reverse()
    return b''.join(arcs)


class Matcher(object):
    def __init__(self, dict_data, max_cache_size=1024, max_cached_word_len=8):
        if dict_data:
            self.dict_data = dict_data
            self.dict_len = len(dict_data)
            # bytes -> (position, final_outputs, outputs)
            self.cache = [OrderedDict() for _ in range(len(dict_data))]
            self.max_cache_size = max_cache_size
            self.max_cached_word_len = max_cached_word_len
            self.lock = threading.Lock()

    def run(self, word, common_prefix_match=True):
        output = set()
        for i in range(self.dict_len):
            output |= self._run(word, i, common_prefix_match)
        return bool(output), output  # accept if output is not empty

    def _run(self, word, data_num, common_prefix_match):
        outputs = set()
        buf = b''
        i = pos = 0
        data = self.dict_data[data_num]
        word_len, data_len = len(word), len(data)

        # simple lru cache
        # any prefix is in cache?
        for j in range(min(word_len, self.max_cached_word_len), 2, -1):
            if word[:j] in self.cache[data_num]:
                pos, outputs, buf = self.cache[data_num][word[:j]]
                # move this entry to top
                with self.lock:
                    del[self.cache[data_num][word[:j]]]
                    self.cache[data_num][word[:j]] = (pos, set(outputs), buf)
                # A cached entry found. We can skip to the position.
                i = j
                break

        while pos < data_len:
            arc = self.next_arc(data, pos)
            # arc[0]: flag
            # arc[1]: label
            # arc[2]: output
            # arc[3]: final_output
            # arc[4]: target
            # arc[5]: incr
            if arc[0] & FLAG_FINAL_ARC:
                if common_prefix_match or i >= word_len:
                    for out in arc[3]:
                        outputs.add(buf + out)
                if arc[0] & FLAG_LAST_ARC or i > word_len:
                    break
                pos += arc[5]
                if i < self.max_cached_word_len:
                    with self.lock:
                        # add to cache
                        self.cache[data_num][word[:i]] = (pos, set(outputs), buf)
                        # check cache size
                        if len(self.cache[data_num]) >= self.max_cache_size:
                            self.cache[data_num].popitem(last=False)
            elif i < word_len:
                if word[i] == arc[1]:
                    buf += arc[2]
                    i += 1
                    pos += arc[4]
                elif arc[0] & FLAG_LAST_ARC:
                    break
                else:
                    pos += arc[5]
            else:   # i >= word_len
                break

        return outputs

    @lru_cache(maxsize=4096)
    def next_arc(self, data, addr):
        assert addr >= 0
        # arc address
        pos = addr
        # the arc
        label = 0
        output = bytes()
        final_output = [b'']
        target = 0
        # read flag
        flag = data[pos]
        pos += 1
        if flag & FLAG_FINAL_ARC:
            if flag & FLAG_ARC_HAS_FINAL_OUTPUT:
                # read final outputs
                final_output_count = unpack('I', data[pos:pos + 4])[0]
                pos += 4
                buf = []
                for _ in range(final_output_count):
                    output_size = unpack('I', data[pos:pos + 4])[0]
                    pos += 4
                    if output_size:
                        buf.append(data[pos:pos + output_size])
                        pos += output_size
                final_output = buf
        else:
            # read label
            label = data[pos]
            pos += 1
            if flag & FLAG_ARC_HAS_OUTPUT:
                # read output
                output_size = unpack('I', data[pos:pos + 4])[0]
                pos += 4
                output = data[pos:pos + output_size]
                pos += output_size
            # read target's (relative) address
            target = unpack('I', data[pos:pos + 4])[0]
            pos += 4
        return flag, label, output, final_output, target, pos - addr


if __name__ == '__main__':
    inputs1 = [
        ('apr'.encode('utf8'), '30'),
        ('aug'.encode('utf8'), '31'),
        ('dec'.encode('utf8'), '31'.encode('utf8')),
        ('feb'.encode('utf8'), '28'.encode('utf8')),
        ('feb'.encode('utf8'), '29'.encode('utf8')),
        ('jan'.encode('utf8'), '31'.encode('utf8')),
        ('jul'.encode('utf8'), '31'.encode('utf8')),
        ('jun'.encode('utf8'), '30'.encode('utf8'))
    ]
    processed, fst = create_minimum_transducer(inputs1)
    data = compileFST(fst)

    m = Matcher([data])
    # print(m.run('apr'))
    print(m.run('apr'.encode('utf8')))
    print(m.run('aug'.encode('utf8')))

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


"""
The tokenizer module supplies Token and Tokenizer classes.

Usage:

>>> from janome.tokenizer import Tokenizer
>>> t = Tokenizer()
>>> for token in t.tokenize('すもももももももものうち'):
...   print(token)
...
すもも	名詞,一般,*,*,*,*,すもも,スモモ,スモモ
も	助詞,係助詞,*,*,*,*,も,モ,モ
もも	名詞,一般,*,*,*,*,もも,モモ,モモ
も	助詞,係助詞,*,*,*,*,も,モ,モ
もも	名詞,一般,*,*,*,*,もも,モモ,モモ
の	助詞,連体化,*,*,*,*,の,ノ,ノ
うち	名詞,非自立,副詞可能,*,*,*,うち,ウチ,ウチ

with wakati ('分かち書き') mode:

>>> from janome.tokenizer import Tokenizer
>>> t = Tokenizer()
>>> for token in t.tokenize('すもももももももものうち', wakati=True):
...   print(token)
...
すもも
も
もも
も
もも
の
うち

with user dictionary (IPAdic format):

.. code-block:: shell

  $ cat examples/user_ipadic.csv
  東京スカイツリー,1288,1288,4569,名詞,固有名詞,一般,*,*,*,東京スカイツリー,トウキョウスカイツリー,トウキョウスカイツリー
  東武スカイツリーライン,1288,1288,4700,名詞,固有名詞,一般,*,*,*,東武スカイツリーライン,トウブスカイツリーライン,トウブスカイツリーライン
  とうきょうスカイツリー駅,1288,1288,4143,名詞,固有名詞,一般,*,*,*,とうきょうスカイツリー駅,トウキョウスカイツリーエキ,トウキョウスカイツリーエキ

>>> t = Tokenizer("user_ipadic.csv", udic_enc="utf8")
>>> for token in t.tokenize('東京スカイツリーへのお越しは、東武スカイツリーライン「とうきょうスカイツリー駅」が便利です。'):
...  print(token)...
...
東京スカイツリー	名詞,固有名詞,一般,*,*,*,東京スカイツリー,トウキョウスカイツリー,トウキョウスカイツリー
へ	助詞,格助詞,一般,*,*,*,へ,ヘ,エ
の	助詞,連体化,*,*,*,*,の,ノ,ノ
お越し	名詞,一般,*,*,*,*,お越し,オコシ,オコシ
は	助詞,係助詞,*,*,*,*,は,ハ,ワ
、	記号,読点,*,*,*,*,、,、,、
東武スカイツリーライン	名詞,固有名詞,一般,*,*,*,東武スカイツリーライン,トウブスカイツリーライン,トウブスカイツリーライン
「	記号,括弧開,*,*,*,*,「,「,「
とうきょうスカイツリー駅	名詞,固有名詞,一般,*,*,*,とうきょうスカイツリー駅,トウキョウスカイツリーエキ,トウキョウスカイツリーエキ
」	記号,括弧閉,*,*,*,*,」,」,」
が	助詞,格助詞,一般,*,*,*,が,ガ,ガ
便利	名詞,形容動詞語幹,*,*,*,*,便利,ベンリ,ベンリ
です	助動詞,*,*,*,特殊・デス,基本形,です,デス,デス
。	記号,句点,*,*,*,*,。,。,。

with user dictionary (simplified format):

.. code-block:: shell

  $ cat examples/user_simpledic.csv
  東京スカイツリー,カスタム名詞,トウキョウスカイツリー
  東武スカイツリーライン,カスタム名詞,トウブスカイツリーライン
  とうきょうスカイツリー駅,カスタム名詞,トウキョウスカイツリーエキ

>>> t = Tokenizer("user_simpledic.csv", udic_type="simpledic", udic_enc="utf8")
>>> for token in t.tokenize('東京スカイツリーへのお越しは、東武スカイツリーライン「とうきょうスカイツリー駅」が便利です。'):
...   print(token)

"""

import sys
import os
from typing import Iterator, Union, Tuple, Optional, Any
from .lattice import Lattice, Node, SurfaceNode, BOS, EOS, NodeType  # type: ignore
from .dic import UserDictionary, CompiledUserDictionary  # type: ignore
from .system_dic import SystemDictionary, MMapSystemDictionary
from .fst import Matcher

try:
    from janome.sysdic import all_fstdata, connections  # type: ignore
except ImportError:
    # hack for unit testing...
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from sysdic import all_fstdata, connections  # type: ignore


DEFAULT_MMAP_MODE = sys.maxsize > 2**32


class Token(object):
    """
    A Token object contains all information for a token.
    """

    def __init__(self, node: Node, extra: Optional[Tuple] = None):
        self.node = node
        self.extra = extra

    def __getattr__(self, name) -> Any:
        if name == 'surface':
            return self.node.surface
        elif name == 'part_of_speech':
            return self.extra[0] if self.extra else self.node.part_of_speech
        elif name == 'infl_type':
            return self.extra[1] if self.extra else self.node.infl_type
        elif name == 'infl_form':
            return self.extra[2] if self.extra else self.node.infl_form
        elif name == 'base_form':
            return self.extra[3] if self.extra else self.node.base_form
        elif name == 'reading':
            return self.extra[4] if self.extra else self.node.reading
        elif name == 'phonetic':
            return self.extra[5] if self.extra else self.node.phonetic
        elif name == 'node_type':
            return self.node.node_type
        else:
            None

    def __setattr__(self, name: str, value: Any) -> None:
        # see https://stackoverflow.com/questions/61213745/typechecking-dynamically-added-attributes
        object.__setattr__(self, name, value)

    def __str__(self):
        return f'{self.surface}\t' \
            f'{self.part_of_speech},{self.infl_type},{self.infl_form},{self.base_form},{self.reading},{self.phonetic}'


class Tokenizer(object):
    """
    A Tokenizer tokenizes Japanese texts with system and optional user defined dictionary.
    """
    MAX_CHUNK_SIZE = 1024
    CHUNK_SIZE = 500

    def __init__(self, udic: str = '', *,
                 udic_enc: str = 'utf8',
                 udic_type: str = 'ipadic',
                 max_unknown_length: int = 1024,
                 wakati: bool = False,
                 mmap: bool = DEFAULT_MMAP_MODE,
                 dotfile: str = ''):
        """
        Initialize Tokenizer object with optional arguments.

        :param udic: (Optional) user dictionary file (CSV format) or directory path to compiled dictionary data
        :param udic_enc: (Optional) character encoding for user dictionary. default is 'utf-8'
        :param udic_type: (Optional) user dictionray type. supported types are 'ipadic' and 'simpledic'.
                          default is 'ipadic'
        :param max_unknows_length: (Optional) max unknown word length. default is 1024.
        :param wakati: (Optional) if given True load minimum sysdic data for 'wakati' mode.
        :param mmap: (Optional) if given False, memory-mapped file mode is disabled.
                     Set this option to False on any environments that do not support mmap.
                     Default is True on 64bit architecture; otherwise False.

        .. seealso:: http://mocobeta.github.io/janome/en/#use-with-user-defined-dictionary
        """
        self.sys_dic: Union[SystemDictionary, MMapSystemDictionary]
        self.user_dic: Optional[Union[UserDictionary, CompiledUserDictionary]]
        self.wakati = wakati
        self.matcher = Matcher(all_fstdata())
        if mmap:
            self.sys_dic = MMapSystemDictionary.instance()
        else:
            self.sys_dic = SystemDictionary.instance()
        if udic:
            if udic.endswith('.csv'):
                # build user dictionary from CSV
                self.user_dic = UserDictionary(udic, udic_enc, udic_type, connections)
            elif os.path.isdir(udic):
                # load compiled user dictionary
                self.user_dic = CompiledUserDictionary(udic, connections)
            else:
                self.user_dic = None
        else:
            self.user_dic = None
        self.max_unknown_length = max_unknown_length

    def tokenize(self, text: str, *, wakati: bool = False, baseform_unk: bool = True, dotfile: str = '') \
            -> Iterator[Union[Token, str]]:
        """
        Tokenize the input text.

        :param text: unicode string to be tokenized
        :param wakati: (Optinal) if given True returns surface forms only. default is False.
        :param baseform_unk: (Optional) if given True sets base_form attribute for unknown tokens. default is True.
        :param dotfile: (Optional) if specified, graphviz dot file is output to the path for later visualizing
                        of the lattice graph. This option is ignored when the input length is
                        larger than MAX_CHUNK_SIZE.

        :return: generator yielding tokens (wakati=False) or generator yielding string (wakati=True)
        """
        if self.wakati:
            wakati = True
        if dotfile and len(text) < Tokenizer.MAX_CHUNK_SIZE:
            return self.__tokenize_stream(text, wakati, baseform_unk, dotfile)
        else:
            return self.__tokenize_stream(text, wakati, baseform_unk, '')

    def __tokenize_stream(self, text, wakati, baseform_unk, dotfile):
        text = text.strip()
        text_length = len(text)
        processed = 0
        while processed < text_length:
            tokens, pos = self.__tokenize_partial(text[processed:], wakati, baseform_unk, dotfile)
            for token in tokens:
                yield token
            processed += pos

    def __tokenize_partial(self, text, wakati, baseform_unk, dotfile):
        if self.wakati and not wakati:
            raise WakatiModeOnlyException

        chunk_size = min(len(text), Tokenizer.MAX_CHUNK_SIZE)
        lattice = Lattice(chunk_size, self.sys_dic)
        pos = 0
        while not self.__should_split(text, pos):
            encoded_partial_text = text[pos:pos + min(50, chunk_size - pos)].encode('utf-8')
            # user dictionary
            if self.user_dic:
                entries = self.user_dic.lookup(encoded_partial_text)
                for e in entries:
                    lattice.add(SurfaceNode(e, NodeType.USER_DICT))
                matched = len(entries) > 0

            # system dictionary
            entries = self.sys_dic.lookup(encoded_partial_text, self.matcher)
            for e in entries:
                lattice.add(SurfaceNode(e, NodeType.SYS_DICT))
            matched = len(entries) > 0

            # unknown
            cates = self.sys_dic.get_char_categories(text[pos])
            if cates:
                for cate in cates:
                    if matched and not self.sys_dic.unknown_invoked_always(cate):
                        continue
                    # unknown word length
                    length = self.sys_dic.unknown_length(cate) \
                        if not self.sys_dic.unknown_grouping(cate) else self.max_unknown_length
                    assert length >= 0
                    # buffer for unknown word
                    buf = text[pos]
                    for p in range(pos + 1, min(chunk_size, pos + length + 1)):
                        _cates = self.sys_dic.get_char_categories(text[p])
                        if cate in _cates or any(cate in _compat_cates for _compat_cates in _cates.values()):
                            buf += text[p]
                        else:
                            break
                    unknown_entries = self.sys_dic.unknowns.get(cate)
                    assert unknown_entries
                    for entry in unknown_entries:
                        left_id, right_id, cost, part_of_speech = entry
                        base_form = buf if baseform_unk else '*'
                        dummy_dict_entry = (buf, left_id, right_id, cost, part_of_speech, '*', '*', base_form, '*', '*')
                        lattice.add(Node(dummy_dict_entry, NodeType.UNKNOWN))

            pos += lattice.forward()
        lattice.end()
        min_cost_path = lattice.backward()
        assert isinstance(min_cost_path[0], BOS)
        assert isinstance(min_cost_path[-1], EOS)
        if wakati:
            tokens = [node.surface for node in min_cost_path[1:-1]]
        else:
            tokens = []
            for node in min_cost_path[1:-1]:
                if type(node) == SurfaceNode and node.node_type == NodeType.SYS_DICT:
                    tokens.append(Token(node, self.sys_dic.lookup_extra(node.num)))
                elif type(node) == SurfaceNode and node.node_type == NodeType.USER_DICT:
                    tokens.append(Token(node, self.user_dic.lookup_extra(node.num)))
                else:
                    tokens.append(Token(node))
        if dotfile:
            lattice.generate_dotfile(filename=dotfile)
        return (tokens, pos)

    def __should_split(self, text, pos):
        return \
            pos >= len(text) or \
            pos >= Tokenizer.MAX_CHUNK_SIZE or \
            (pos >= Tokenizer.CHUNK_SIZE and self.__splittable(text[:pos]))

    def __splittable(self, text):
        return self.__is_punct(text[-1]) or self.__is_newline(text)

    def __is_punct(self, c):
        return c == u'、' or c == u'。' or c == u',' or c == u'.' or c == u'？' or c == u'?' or c == u'！' or c == u'!'

    def __is_newline(self, text):
        return text.endswith('\n\n') or text.endswith('\r\n\r\n')


class WakatiModeOnlyException(Exception):
    pass

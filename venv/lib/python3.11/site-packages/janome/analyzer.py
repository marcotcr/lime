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
The analyzer module supplies Analyzer framework for pre-processing and post-processing for morphological analysis.

Added in *version 0.3.4*

**NOTE** This is experimental. The class/method interfaces can be modified in the future releases.

Usage:

>>> from janome.tokenizer import Tokenizer
>>> from janome.analyzer import Analyzer
>>> from janome.charfilter import *
>>> from janome.tokenfilter import *
>>> text = '蛇の目はPure Ｐｙｔｈｏｎな形態素解析器です。'
>>> char_filters = [UnicodeNormalizeCharFilter(), RegexReplaceCharFilter('蛇の目', 'janome')]
>>> tokenizer = Tokenizer()
>>> token_filters = [CompoundNounFilter(), POSStopFilter(['記号','助詞']), LowerCaseFilter()]
>>> a = Analyzer(char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters)
>>> for token in a.analyze(text):
...     print(token)
...
janome	名詞,固有名詞,組織,*,*,*,*,*,*
pure	名詞,固有名詞,組織,*,*,*,*,*,*
python	名詞,一般,*,*,*,*,*,*,*
な	助動詞,*,*,*,特殊・ダ,体言接続,だ,ナ,ナ
形態素解析器	名詞,複合,*,*,*,*,形態素解析器,ケイタイソカイセキキ,ケイタイソカイセキキ
です	助動詞,*,*,*,特殊・デス,基本形,です,デス,デス

Usage (word count with TokenCountFilter):

>>> from janome.tokenizer import Tokenizer
>>> from janome.analyzer import Analyzer
>>> from janome.tokenfilter import *
>>> text = 'すもももももももものうち'
>>> token_filters = [POSKeepFilter(['名詞']), TokenCountFilter()]
>>> a = Analyzer(token_filters=token_filters)
>>> for k, v in a.analyze(text):
...   print('%s: %d' % (k, v))
...
もも: 2
すもも: 1
うち: 1
"""

from typing import List, Iterator, Any, Optional
from .tokenizer import Tokenizer
from .charfilter import CharFilter
from .tokenfilter import TokenFilter


class Analyzer(object):
    """
    An Analyzer analyzes Japanese texts with customized :class:`.CharFilter` chain,
    :class:`.Tokenizer` and :class:`.TokenFilter` chain.

    Added in *version 0.3.4*
    """

    def __init__(self, *,
                 char_filters: List[CharFilter] = [],
                 tokenizer: Optional[Tokenizer] = None,
                 token_filters: List[TokenFilter] = []):
        """
        Initialize Analyzer object with CharFilters, a Tokenizer and TokenFilters.

        :param char_filters: (Optional) CharFilters list. CharFilters are applied to the input text
                             in the list order. default is the empty list.
        :param tokenizer: (Optional) A Tokenizer object. Tokenizer tokenizes the text modified by
                          *char_filters*. default is Tokenizer initialized with no extra options.
                          **WARNING:** A Tokenizer initialized with *wakati=True* option is not accepted.
        :param token_filters: (Optional) TokenFilters list. TokenFilters are applied to the Tokenizer's
                              output in the list order. default is the empty list.
        """
        if not tokenizer:
            self.tokenizer = Tokenizer()
        elif tokenizer.wakati:
            raise Exception('Invalid argument: A Tokenizer with wakati=True option is not accepted.')
        else:
            self.tokenizer = tokenizer
        self.char_filters = char_filters
        self.token_filters = token_filters

    def analyze(self, text: str) -> Iterator[Any]:
        """
        Analyze the input text with custom CharFilters, Tokenizer and TokenFilters.

        :param text: unicode string to be tokenized

        :return: token generator. emitted element type depends on the output of the last TokenFilter.
                 (e.g., ExtractAttributeFilter emits strings.)
        """
        for cfilter in self.char_filters:
            text = cfilter(text)
        tokens = self.tokenizer.tokenize(text, wakati=False)
        for tfilter in self.token_filters:
            tokens = tfilter(tokens)  # type: ignore
        return tokens

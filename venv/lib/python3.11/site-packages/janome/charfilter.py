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
import unicodedata
import re


class CharFilter(ABC):
    """
    Base CharFilter class.

    A CharFilter modifies or transforms the input text according to the rule described in apply() method.
    Subclasses must implement apply() method.

    Added in *version 0.3.4*
    """

    @abstractmethod
    def apply(self, text: str) -> str:
        pass

    def __call__(self, text: str) -> str:
        return self.apply(text)


class RegexReplaceCharFilter(CharFilter):
    """
    RegexReplaceCharFilter replaces string matched with a regular expression pattern to replacement string.

    Added in *version 0.3.4*
    """

    def __init__(self, pat: str, repl: str):
        """
        Initialize RegexReplaceCharFilter with a regular expression pattern string and replacement.

        :param pattern: regular expression string.
        :param repl: replacement string.
        """
        self.pattern = re.compile(pat)
        self.replacement = repl

    def apply(self, text: str) -> str:
        return re.sub(self.pattern, self.replacement, text)


class UnicodeNormalizeCharFilter(CharFilter):
    """
    UnicodeNormalizeCharFilter normalizes Unicode string.

    Added in *version 0.3.4*
    """

    def __init__(self, form: str = 'NFKC'):
        """
        Initialize UnicodeNormalizeCharFilter with normalization form.

        See also `unicodedata.normalize <https://docs.python.org/3.6/library/unicodedata.html#unicodedata.normalize>`_

        :param form: (Optional) normalization form. valid values for *form* are 'NFC', 'NFKC', 'NFD', and 'NFKD'.
                     default is 'NFKC'
        """
        self.form = form

    def apply(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)

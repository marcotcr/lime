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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class ProgressHandler(ABC):
    """
    Base progress handler class
    """

    @abstractmethod
    def on_start(self, total, desc=None):
        pass

    @abstractmethod
    def on_progress(self, value=1):
        pass

    @abstractmethod
    def on_complete(self):
        pass


class SimpleProgressIndicator(ProgressHandler):
    def __init__(self, update_frequency=0.0001, format=None):
        self.update_frequency = update_frequency
        self.format = format or '\r{}: {:.1f}% | {}/{}'
        self.total = None
        self.value = None
        self.desc = None

    def print_progress(self, terminator=''):
        logger.handlers[0].terminator = terminator
        logger.info(self.format.format(
            self.desc,
            self.value * 100 / self.total,
            self.value,
            self.total))

    def on_start(self, total, value=0, desc=None):
        self.total = total
        self.value = value
        self.desc = desc or 'Processing'

    def on_progress(self, value=1):
        self.value += value

        if float.is_integer(self.value * self.update_frequency):
            self.print_progress()

    def on_complete(self):
        self.print_progress('\n')
        self.total = self.value = self.desc = None

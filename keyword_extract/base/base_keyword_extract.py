"""**KeyWordExtract** interface."""

from abc import ABC, abstractmethod

from typing import List


class BaseKeyWordExtract(ABC):
    @abstractmethod
    def infer(self, input_list: List[str]):
        """关键词抽取预测

        Parameters
        ----------
        input_list : list
            输入列表
        """

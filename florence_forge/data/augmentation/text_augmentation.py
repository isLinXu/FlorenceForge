"""文本数据增强"""

import random
import re


class TextAugmentation:
    """文本数据增强，适用于 VLM 训练中的 suffix 文本。"""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def random_case_toggle(self, text: str) -> str:
        """随机大小写切换（仅对 ASCII 字母生效）。"""
        if random.random() < self.probability:
            return "".join(c.upper() if c.islower() else c.lower() for c in text)
        return text

    def random_word_shuffle(self, text: str) -> str:
        """随机打乱词序（适用于英文短句）。"""
        if random.random() < self.probability:
            words = text.split()
            if len(words) > 1:
                random.shuffle(words)
                return " ".join(words)
        return text

    def random_word_drop(self, text: str, drop_prob: float = 0.1) -> str:
        """随机丢弃单词。"""
        if random.random() < self.probability:
            words = text.split()
            kept = [w for w in words if random.random() > drop_prob]
            return " ".join(kept) if kept else text
        return text

    def random_whitespace_variation(self, text: str) -> str:
        """随机增加/减少空白字符。"""
        if random.random() < self.probability:
            text = re.sub(r"  +", " ", text.strip())
        return text

    def apply_augmentations(self, text: str) -> str:
        """应用所有文本增强。"""
        text = self.random_case_toggle(text)
        text = self.random_whitespace_variation(text)
        text = self.random_word_drop(text)
        text = self.random_word_shuffle(text)
        return text

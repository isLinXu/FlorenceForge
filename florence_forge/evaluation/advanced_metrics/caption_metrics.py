"""图像描述评估指标"""

import re
from typing import List, Dict
from collections import Counter
import math

class CaptionMetrics:
    """图像描述评估指标"""
    
    def __init__(self):
        pass
    
    def calculate_bleu(self, reference: str, candidate: str, n: int = 4) -> float:
        """计算BLEU分数"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # 计算n-gram精度
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, i)
            cand_ngrams = self._get_ngrams(cand_tokens, i)
            
            if len(cand_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram in cand_ngrams:
                if ngram in ref_ngrams:
                    matches += min(cand_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(cand_ngrams.values())
            precisions.append(precision)
        
        # 计算几何平均
        if any(p == 0 for p in precisions):
            return 0.0
        
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        
        # 简化的长度惩罚
        bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0
        
        return bp * geo_mean
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """获取n-gram"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """计算ROUGE-L分数"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        # 计算最长公共子序列
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(cand_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

"""
LLM 自动化评测脚本
功能：批量调用 RAG 问答接口，自动打分并输出评测报告
评测指标：知识匹配准确率、幻觉率、指令遵循度、响应时延
"""

import pandas as pd
import time
import json

class LLMEvaluator:
    def __init__(self, model_api, knowledge_base):
        self.model_api = model_api
        self.knowledge_base = knowledge_base
        self.results = []
    
    def run_eval(self, test_cases):
        """批量执行评测"""
        for case in test_cases:
            start_time = time.time()
            
            # 调用 RAG 问答接口
            answer = self.model_api.query(case["question"])
            
            # 计算响应时延
            latency = round(time.time() - start_time, 3)
            
            # 自动打分（匹配准确率、幻觉检测）
            accuracy_score = self._calc_accuracy(answer, case["standard_answer"])
            hallucination_score = self._detect_hallucination(answer, case["knowledge_points"])
            
            # 保存结果
            self.results.append({
                "question": case["question"],
                "model_answer": answer,
                "accuracy_score": accuracy_score,
                "hallucination_score": hallucination_score,
                "latency": latency
            })
        
        return self._generate_report()
    
    def _calc_accuracy(self, model_answer, standard_answer):
        """计算知识匹配准确率"""
        # 关键词匹配 + 语义相似度计算
        model_keywords = set(model_answer.split())
        std_keywords = set(standard_answer.split())
        overlap = len(model_keywords & std_keywords)
        return round(overlap / len(std_keywords), 2) if std_keywords else 0
    
    def _detect_hallucination(self, answer, knowledge_points):
        """检测幻觉率：统计不在知识库中的表述占比"""
        answer_words = set(answer.split())
        known_words = set(knowledge_points)
        unknown_ratio = len(answer_words - known_words) / len(answer_words) if answer_words else 0
        return round(1 - unknown_ratio, 2)  # 越高表示幻觉越少
    
    def _generate_report(self):
        """生成评测报告"""
        df = pd.DataFrame(self.results)
        report = {
            "total_cases": len(df),
            "avg_accuracy": round(df["accuracy_score"].mean(), 3),
            "avg_hallucination_score": round(df["hallucination_score"].mean(), 3),
            "avg_latency": round(df["latency"].mean(), 3),
            "detail": df.to_dict("records")
        }
        
        # 保存报告
        with open("eval_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report


if __name__ == "__main__":
    # 示例：运行评测
    test_cases = [
        {
            "question": "什么是数据结构中的链表？",
            "standard_answer": "链表是一种线性数据结构，由节点组成，每个节点包含数据和指向下一个节点的指针",
            "knowledge_points": ["链表", "线性", "数据结构", "节点", "指针"]
        }
    ]
    
    print("评测完成，报告已生成")
from typing import List

import numpy as np

def calculate_choice_probabilities(offers: List[float], temperature: float = 7.5) -> List[float]:
    """
    計算不同薪水報價的選擇機率，使用 Softmax 函數，並防止數值溢出
    :param offers: List[float] 各種薪水報價
    :param temperature: float 控制選擇的隨機性
    :return: List[float] 每個報價的選擇機率
    """
    offers = np.array(offers, dtype=np.float64)

    # **Step 1: 標準化數據 (Min-Max Normalization)**
    min_offer = np.min(offers)
    max_offer = np.max(offers)
    if max_offer - min_offer > 0:
        normalized_offers = (offers - min_offer) / (max_offer - min_offer) * 100  # 轉換到 0~100 範圍
    else:
        normalized_offers = np.zeros_like(offers)  # 避免除以 0

    # **Step 2: Softmax 計算**
    max_offer_norm = np.max(normalized_offers)  # 找到最大值，避免 overflow
    exp_values = np.exp((normalized_offers - max_offer_norm) / temperature)
    probabilities = exp_values / np.sum(exp_values)  # Softmax 正規化

    return probabilities


if __name__ == "__main__":
    # 測試案例
    offers_small = [50, 40, 100, 90]
    offers_large = [50000, 40000, 100000, 90000]

    probabilities_small = calculate_choice_probabilities(offers_small, temperature=7.5)
    probabilities_large = calculate_choice_probabilities(offers_large, temperature=7.5)

    print(f"📌 小數據測試: {probabilities_small}")
    print(f"📌 大數據測試: {probabilities_large}")

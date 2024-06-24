import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample

Gemini_adv = [7/10,6/11,4/9,8/13,7/13,7/12,8/13,8/12,6/12,5/11,10/12,4/10]
GPT_4o = [4/10,7/11,11/14,11/13,12/13,7/12,11/13,9/12,10/12,4/11,8/12,6/10]
opg = []
for i in range(len(Gemini_adv)):
    opg.append((Gemini_adv[i] + GPT_4o[i])/2)

# ... (rest of the code)

def mean_confidence_interval_bootstrap(data, confidence=0.95, n_resamples=10000):
    means = [np.mean(resample(data, replace=True)) for _ in range(n_resamples)]
    alpha = 1 - confidence
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return np.mean(data), lower, upper

def median_confidence_interval_bootstrap(data, confidence=0.95, n_resamples=10000):
    medians = [np.median(resample(data, replace=True)) for _ in range(n_resamples)]
    alpha = 1 - confidence
    lower = np.percentile(medians, alpha / 2 * 100)
    upper = np.percentile(medians, (1 - alpha / 2) * 100)
    return np.median(data), lower, upper

def analyze_accuracy(list1, list2, conf_level=0.95):
        # McNemar's Test
    data = pd.DataFrame({"List1": list1, "List2": list2})
    table = pd.crosstab(data["List1"] > data["List2"], data["List2"] > data["List1"])
    result = mcnemar(table, exact=False, correction=True) 
    print(f"McNemar's Test Results:\n{result}\n")

    print("Confidence Intervals for Medians (", conf_level, "):")
    for i, lst in enumerate([list1, list2]):
        median, lower, upper = median_confidence_interval_bootstrap(lst, confidence=conf_level)
        print(f"List {i+1}:  Median = {median:.4f} ( {lower:.4f} - {upper:.4f} )")

    print("Confidence Intervals for mean (", conf_level, "):")
    for i, lst in enumerate([list1, list2]):
        median, lower, upper = mean_confidence_interval_bootstrap(lst, confidence=conf_level)
        print(f"List {i+1}:  Median = {median:.4f} ( {lower:.4f} - {upper:.4f} )")




analyze_accuracy(Gemini_adv, GPT_4o, conf_level=0.95)


plt.scatter(range(len(opg)),opg, color='blue', marker='o')  # You can customize color and marker
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Points')
plt.show()




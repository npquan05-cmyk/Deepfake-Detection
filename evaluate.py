import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

def evaluate_model(model, val_generator, test_generator):

    # ===== VALIDATION =====
    val_prob = model.predict(val_generator).flatten()
    val_true = val_generator.classes

    fpr, tpr, thresholds = roc_curve(val_true, val_prob)
    auc = roc_auc_score(val_true, val_prob)

    print("Validation AUC:", auc)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("Optimal Threshold:", optimal_threshold)

    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    # ===== TEST =====
    test_prob = model.predict(test_generator).flatten()
    pred = (test_prob > optimal_threshold).astype(int)
    true = test_generator.classes

    print(classification_report(true, pred))

    cm = confusion_matrix(true, pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    test_auc = roc_auc_score(true, test_prob)
    print("Test AUC:", test_auc)

    return optimal_threshold
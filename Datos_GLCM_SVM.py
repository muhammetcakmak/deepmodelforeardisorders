import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, mean_absolute_error, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from sklearn.svm import SVC

def extract_glcms(image):
    # Gri seviyeli görüntü elde etme
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # GLCM hesaplama
    distances = [1]  # Pikseller arasındaki mesafe
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Yönler
    glcm = greycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # GLCM özellikleri
    contrast = greycoprops(glcm, 'contrast').ravel()
    dissimilarity = greycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = greycoprops(glcm, 'homogeneity').ravel()
    energy = greycoprops(glcm, 'energy').ravel()
    correlation = greycoprops(glcm, 'correlation').ravel()

    return np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation))

def load_data(data_path):
    features = []
    labels = []
    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV, BGR renk düzenini kullanır, bu yüzden RGB'ye dönüştürüyoruz
            extracted_features = extract_glcms(image)
            features.append(extracted_features)
            labels.append(class_folder)
    return np.array(features), np.array(labels)

def main():
    train_data_path = "datasets/Datos/train/"
    test_data_path = "datasets/Datos/validation/"

    train_features, train_labels = load_data(train_data_path)
    test_features, test_labels = load_data(test_data_path)

    # Etiketleri int türünde dönüştür
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    num_folds = 5
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_test_labels = []
    all_predictions = []
    all_probabilities = []

    all_precisions = []
    all_recalls = []
    all_auc = []

    for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(train_features, train_labels)):
        print(f"\n{fold + 1}. Katman / {num_folds}")

        fold_train_features, fold_train_labels = train_features[train_indices], train_labels[train_indices]
        fold_test_features, fold_test_labels = train_features[test_indices], train_labels[test_indices]

        # SVM modeli
        svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
        svm_classifier.fit(fold_train_features, fold_train_labels)

        fold_predictions = svm_classifier.predict(fold_test_features)
        fold_probabilities = svm_classifier.predict_proba(fold_test_features)

        fold_accuracy = accuracy_score(fold_test_labels, fold_predictions)
        print(f'SVM Test Doğruluğu (Kat {fold + 1}): {fold_accuracy * 100:.2f}%')

        fold_classification_report = classification_report(fold_test_labels, fold_predictions, digits=4)
        print("\nSVM Sınıflandırma Raporu (Kat {}):".format(fold + 1))
        print(fold_classification_report)

        fold_mae = mean_absolute_error(fold_test_labels, fold_predictions)
        print(f'SVM MAE (Kat {fold + 1}): {fold_mae:.4f}')

        fold_precision, fold_recall, _ = precision_recall_curve(fold_test_labels, fold_probabilities[:, 1], pos_label=1)
        fold_auc = auc(fold_recall, fold_precision)

        all_precisions.append(fold_precision)
        all_recalls.append(fold_recall)
        all_auc.append(fold_auc)

        all_test_labels.extend(fold_test_labels)
        all_predictions.extend(fold_predictions)
        all_probabilities.extend(fold_probabilities[:, 1])

    print("\nGenel SVM Sınıflandırma Raporu:")
    overall_classification_report = classification_report(all_test_labels, all_predictions, digits=4)
    print(overall_classification_report)

    cm = confusion_matrix(all_test_labels, all_predictions)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(len(os.listdir(train_data_path)))
    plt.xticks(tick_marks, os.listdir(train_data_path), rotation=45)
    plt.yticks(tick_marks, os.listdir(train_data_path))

    for i in range(len(os.listdir(train_data_path))):
        for j in range(len(os.listdir(train_data_path))):
            plt.text(j, i, "{:.4f}".format(cm[i][j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.show()

    kappa_score = cohen_kappa_score(all_test_labels, all_predictions)
    print(f"Cohen's Kappa Skoru: {kappa_score:.4f}")

    # Precision-Recall Eğrisi
    plt.figure(figsize=(8, 8))
    for i in range(num_folds):
        plt.plot(all_recalls[i], all_precisions[i], lw=2, label=f'Fold {i+1} (AUC = {all_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    main()

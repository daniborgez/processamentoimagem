import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# ===============================================================================
# CONFIGURAÇÕES INICIAIS
# ===============================================================================

# Caminho para a pasta principal do dataset (MUDE ESTE CAMINHO SE NECESSÁRIO)
DATASET_DIR = "." 
IMAGE_SIZE = (28, 28) # Tamanho padrão para redimensionamento

# ===============================================================================
# FUNÇÃO PARA CARREGAR E PRÉ-PROCESSAR AS IMAGENS
# ===============================================================================

def load_and_preprocess_data(dataset_path):
    """Carrega as imagens, pré-processa e define os rótulos binários."""
    print("Iniciando carregamento e pré-processamento das imagens...")
    
    X = []  # Lista para armazenar as imagens processadas (features)
    y = []  # Lista para armazenar os rótulos (labels)
    
    vogais = ['a_l', 'A_u', 'e_l', 'E_u', 'i_l', 'I_u', 'o_l', 'O_u', 'u_l', 'U_u']
    
    for vogal_class_name in vogais:
        # Define o rótulo binário: 1 se o nome da pasta contém 'i' ou 'I', 0 caso contrário
        label_binario = 1 if ('i' in vogal_class_name.lower()) else 0
        
        # Caminho da subpasta para a classe atual
        folder_path = os.path.join(dataset_path, vogal_class_name)
        
        if not os.path.isdir(folder_path):
            print(f"AVISO: Pasta não encontrada: {folder_path}. Pulando.")
            continue
            
        count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                
                # 1. Leitura da imagem
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # 2. Redimensionamento
                img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                
                # 3. Normalização (Escala de Cinza) e Achatar (Flatten)
                # Normaliza para o intervalo [0, 1] e transforma a matriz 28x28 em um vetor de 784 elementos (features)
                features = img_resized.flatten() / 255.0
                
                X.append(features)
                y.append(label_binario)
                count += 1
        
        print(f"  -> {count} imagens da vogal '{vogal_class_name}' carregadas com label binário {label_binario}")

    if not X:
        raise ValueError("Nenhuma imagem foi carregada. Verifique se o caminho 'DATASET_DIR' e a estrutura de pastas estão corretos.")

    X = np.array(X)
    y = np.array(y)
    print(f"\nCarregamento concluído. Total de amostras: {len(X)}")
    return X, y

# ===============================================================================
# FUNÇÃO DE AVALIAÇÃO
# ===============================================================================

def evaluate_model(model, X_test, y_test, nome_modelo):
    """Calcula e imprime as métricas de avaliação no conjunto de teste."""
    
    y_pred = model.predict(X_test)
    
    # Métricas solicitadas
    accuracy = accuracy_score(y_test, y_pred)
    # As métricas são calculadas para a CLASSE POSITIVA ('i', que é 1)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Matriz de Confusão para contexto
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- Resultados do Modelo: {nome_modelo} ---")
    print(f"Acurácia Média: {accuracy:.4f}")
    print(f"Precision ('i'): {precision:.4f}")
    print(f"Recall ('i'): {recall:.4f}")
    print(f"F1-Score ('i'): {f1:.4f}")
    print("\nMatriz de Confusão:")
    print(f"  (Real/Pred) Não-'i' ('i')")
    print(f"Não-'i' (0): {cm[0]}")
    print(f"    'i' (1): {cm[1]}")
    
    return {'Acurácia': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

# ===============================================================================
# FLUXO PRINCIPAL DO PROJETO (AS 4 ETAPAS)
# ===============================================================================

def run_project():
    try:
        # Carregar e pré-processar
        X, y = load_and_preprocess_data(DATASET_DIR)
        
        # -------------------------------------------------------------
        # ETAPA 1: SEPARAÇÃO DOS DADOS EM TREINO E TESTE
        # -------------------------------------------------------------
        
        # 80% para treino, 20% para teste. O 'stratify=y' garante que as classes
        # 'i' (1) e 'não i' (0) sejam distribuídas proporcionalmente nos dois conjuntos.
        print("\n--- ETAPA 1: Divisão Treino/Teste (80%/20%) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Tamanho do conjunto de Treino: {len(X_train)}")
        print(f"Tamanho do conjunto de Teste: {len(X_test)}")
        
        # -------------------------------------------------------------
        # ETAPA 2 & 3: BASELINE (k-NN) E VALIDAÇÃO CRUZADA
        # -------------------------------------------------------------
        
        print("\n\n--- ETAPA 2 & 3: Baseline - k-Nearest Neighbors (k-NN) ---")
        
        # Configurando e treinando o modelo k-NN (Baseline)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        start_time = time.time()
        knn_model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Treinamento k-NN concluído em {end_time - start_time:.2f} segundos.")
        
        # Avaliação com Validação Cruzada (Cross-Validation - k=5)
        print("\n-> Avaliação com Validação Cruzada (k=5):")
        
        # Usamos 'scoring' para obter a média das métricas nos 5 'folds'
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring=metric)
            cv_results[metric] = np.mean(scores)
            print(f"   Média CV {metric.capitalize()}: {np.mean(scores):.4f}")
            
        # Avaliação final no conjunto de TESTE
        knn_metrics = evaluate_model(knn_model, X_test, y_test, "k-NN (Baseline)")
        
        # -------------------------------------------------------------
        # ETAPA 4: OUTRO ALGORITMO (SVM) E ESCOLHA DO MELHOR
        # -------------------------------------------------------------
        
        print("\n\n--- ETAPA 4: Segundo Algoritmo - Support Vector Machine (SVM) ---")
        
        # Configurando e treinando o modelo SVM
        # O kernel 'linear' é um bom ponto de partida para dados de alta dimensão (features achatadas)
        svm_model = SVC(kernel='linear', random_state=42)
        start_time = time.time()
        svm_model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Treinamento SVM concluído em {end_time - start_time:.2f} segundos.")
        
        # Avaliação com Validação Cruzada (Cross-Validation - k=5)
        print("\n-> Avaliação com Validação Cruzada (k=5):")
        for metric in scoring_metrics:
            scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring=metric)
            cv_results[f"svm_{metric}"] = np.mean(scores)
            print(f"   Média CV {metric.capitalize()}: {np.mean(scores):.4f}")
            
        # Avaliação final no conjunto de TESTE
        svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")
        
        # -------------------------------------------------------------
        # ESCOLHA DO MELHOR ALGORITMO
        # -------------------------------------------------------------
        
        print("\n\n=======================================================")
        print("          COMPARAÇÃO E ESCOLHA DO MELHOR MODELO          ")
        print("=======================================================")
        
        # Critério de escolha: F1-Score (busca um equilíbrio entre Precision e Recall)
        if svm_metrics['F1'] > knn_metrics['F1']:
            print("O **SVM** obteve o melhor F1-Score no conjunto de Teste.")
            print(f"SVM F1-Score: {svm_metrics['F1']:.4f} vs k-NN F1-Score: {knn_metrics['F1']:.4f}")
            best_model_name = "SVM"
            best_metrics = svm_metrics
        else:
            print("O **k-NN (Baseline)** obteve o melhor F1-Score no conjunto de Teste.")
            print(f"k-NN F1-Score: {knn_metrics['F1']:.4f} vs SVM F1-Score: {svm_metrics['F1']:.4f}")
            best_model_name = "k-NN (Baseline)"
            best_metrics = knn_metrics
            
        print("\nMelhores Métricas Finais (Modelo Escolhido - Teste):")
        print(f"Modelo: {best_model_name}")
        print(f"Acurácia: {best_metrics['Acurácia']:.4f}")
        print(f"Precision: {best_metrics['Precision']:.4f}")
        print(f"Recall: {best_metrics['Recall']:.4f}")
        print(f"F1-Score: {best_metrics['F1']:.4f}")
        
    except ValueError as e:
        print(f"ERRO CRÍTICO: {e}")
        print("Certifique-se de que a pasta 'v20220930' está no mesmo diretório do script.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    run_project()
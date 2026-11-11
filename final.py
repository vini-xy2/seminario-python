import sys
import os
import subprocess

def auto_configurar_ambiente():
    """Garante que o script corre num ambiente virtual e instala todas as dependências."""
    if sys.prefix == sys.base_prefix:
        print("A preparar o ambiente virtual isolado...")
        pasta_venv = "venv_tomateiro"
        
        try:
            subprocess.run([sys.executable, "-m", "venv", pasta_venv], check=True)
        except subprocess.CalledProcessError:
            print("[ERRO] Instale o pacote base com: sudo apt install python3-venv")
            sys.exit(1)
            
        python_venv = os.path.join(pasta_venv, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(pasta_venv, 'bin', 'python')
        
        print("A instalar bibliotecas (OpenCV, Pandas, TensorFlow-CPU e Kaggle)...")
        pacotes = ["opencv-python", "numpy", "pandas", "tensorflow-cpu", "kaggle"]
        subprocess.run([python_venv, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + pacotes, check=True)
        
        subprocess.run([python_venv] + sys.argv)
        sys.exit(0)

# 1. Configura o ambiente antes de importar as bibliotecas pesadas
auto_configurar_ambiente()

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Oculta mensagens de log pesadas do TensorFlow no terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Variáveis globais do projeto
TAM = (224, 224) 
COLS = ["doenca", "agente", "tratamento"]

# ==========================================
# FASE 1: TREINO (Só corre se o modelo não existir)
# ==========================================
def baixar_dataset_kaggle():
    """Baixa o PlantVillage Dataset."""
    pasta_destino = "plantvillage dataset"
    if not os.path.exists(pasta_destino):
        print("\n[FASE DE TREINO] A baixar as imagens do Kaggle...")
        try:
            subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", "abdallahalidev/plantvillage-dataset", "--unzip"], check=True)
        except subprocess.CalledProcessError:
            print("\n[ERRO KAGGLE] Falta o ficheiro de autenticação 'kaggle.json'.")
            print("Crie um token na sua conta Kaggle e coloque em ~/.kaggle/kaggle.json")
            sys.exit(1)
    return pasta_destino

def treinar_e_guardar_modelo():
    """Treina a CNN e guarda o ficheiro keras."""
    pasta_dados = baixar_dataset_kaggle()
    caminho_imagens = os.path.join(pasta_dados, "color") if os.path.exists(os.path.join(pasta_dados, "color")) else pasta_dados

    print("A preparar dados de treino e validação...")
    treino = tf.keras.utils.image_dataset_from_directory(caminho_imagens, image_size=TAM, validation_split=0.2, subset="training", seed=42)
    validacao = tf.keras.utils.image_dataset_from_directory(caminho_imagens, image_size=TAM, validation_split=0.2, subset="validation", seed=42)
    
    # Guardamos os nomes das classes diretamente da pasta para usar no diagnóstico
    classes_encontradas = treino.class_names
    pd.Series(classes_encontradas).to_csv("classes_treino.csv", index=False, header=False)

    print("\nA treinar a Rede Neural Convolucional...")
    modelo = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=TAM + (3,)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(classes_encontradas), activation="softmax")
    ])
    
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(treino, validation_data=validacao, epochs=5)
    
    modelo.save("tomate_cnn.keras")
    print("\n[SUCESSO] Modelo 'tomate_cnn.keras' criado e guardado!")

# ==========================================
# FASE 2: DIAGNÓSTICO (O fluxo principal)
# ==========================================
def preprocessar(caminho):
    """Lê e prepara a foto para a rede neural."""
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada ou inválida.")
    img_resized = cv2.resize(img, TAM)
    img_blur = cv2.GaussianBlur(cv2.convertScaleAbs(img_resized, alpha=1.2, beta=15), (5, 5), 0)
    img8 = img_blur.copy()
    x = img8.astype("float32") / 255.0
    return img8, x

def estimar_severidade(img8):
    """Mede a percentagem de lesão na folha."""
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    folha = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    lesao = cv2.inRange(hsv, (0, 40, 20), (25, 255, 200))
    total = cv2.countNonZero(folha) + cv2.countNonZero(lesao)
    return 100 * cv2.countNonZero(lesao) / max(total, 1)

def principal():
    print("="*50)
    print(" SISTEMA UNIFICADO: FITOPATIAS DO TOMATEIRO ")
    print("="*50)
    
    # Se o modelo não existir, roda a FASE 1 primeiro
    if not os.path.exists("tomate_cnn.keras"):
        print("\nO ficheiro de inteligência artificial não foi encontrado. Iniciando o treino primeiro...")
        treinar_e_guardar_modelo()
        
    # Carrega as classes que foram guardadas no treino
    if os.path.exists("classes_treino.csv"):
        classes = pd.read_csv("classes_treino.csv", header=None)[0].tolist()
    else:
        classes = ["saudavel", "requeima", "mancha_bacteriana"] # Fallback

    # Inicia a FASE 2
    modelo_cnn = tf.keras.models.load_model("tomate_cnn.keras")
    caminho = input("\nIntroduza o caminho da foto para diagnóstico: ").strip().strip("'\"")
    
    try:
        img8, x = preprocessar(caminho)
        
        # Classificação
        probs = modelo_cnn.predict(np.expand_dims(x, axis=0), verbose=0)[0]
        classe = classes[int(np.argmax(probs))]
        
        # Gestão de Base de Dados
        db = pd.read_csv("doencas.csv") if os.path.exists("doencas.csv") else pd.DataFrame(columns=COLS)
        
        print("\n" + "-"*40)
        print("RESULTADO DO DIAGNÓSTICO")
        print("-" * 40)
        
        # Lógica de decisão baseada no fluxograma
        if "saudavel" in classe.lower() or "healthy" in classe.lower():
            print("Planta sem sintomas aparentes.")
        else:
            if classe in db["doenca"].values:
                info = db[db["doenca"] == classe].iloc[0]
                print(f"Doença: {info['doenca']}\nAgente: {info['agente']}\nTratamento: {info['tratamento']}")
            else:
                db.loc[len(db)] = [classe, "agente?", "tratamento?"]
                db.to_csv("doencas.csv", index=False)
                print(f"Nova doença registada: {classe}. Atualize o ficheiro CSV com os tratamentos.")
                
            print(f"\nSeveridade estimada: {estimar_severidade(img8):.1f}% da área da folha")
        print("-" * 40)
        
    except Exception as e:
        print(f"[ERRO] Falha no diagnóstico: {e}")

if __name__ == "__main__":
    principal()



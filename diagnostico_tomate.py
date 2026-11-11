import sys
import os
import subprocess

def auto_configurar_ambiente():
    """Garante que o script corre num ambiente virtual para evitar o erro do PEP 668 e gere o espaço de disco."""
    # Verifica se já estamos num ambiente virtual
    no_ambiente_virtual = sys.prefix != sys.base_prefix
    
    if not no_ambiente_virtual:
        print("Ambiente global detetado. A criar um ambiente virtual isolado seguro...")
        pasta_venv = "venv_fitopatias"
        
        # 1. Cria o ambiente virtual
        try:
            subprocess.run([sys.executable, "-m", "venv", pasta_venv], check=True)
        except subprocess.CalledProcessError:
            print("\n[ERRO] O seu sistema não tem o pacote base de ambientes virtuais.")
            print("Por favor, execute este comando no terminal primeiro:")
            print("sudo apt install python3-venv\n")
            sys.exit(1)
            
        # 2. Define o caminho do Python dentro do ambiente virtual criado
        if os.name == 'nt': # Windows
            python_venv = os.path.join(pasta_venv, 'Scripts', 'python.exe')
        else: # Linux/Mac
            python_venv = os.path.join(pasta_venv, 'bin', 'python')
            
        # 3. Instala os pacotes necessários sem usar cache (evita Erro 28: No space left)
        print("A instalar pacotes necessários. Isto pode demorar alguns minutos na primeira vez...")
        pacotes = ["opencv-python", "numpy", "pandas", "tensorflow"]
        
        # A alteração foi feita aqui: adição do "--no-cache-dir"
        subprocess.run([python_venv, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + pacotes, check=True)
        
        print("Instalação concluída!\n")
        
        # 4. Reinicia o script usando o Python do ambiente virtual
        subprocess.run([python_venv] + sys.argv)
        
        # Sai do processo original para não correr o código duas vezes
        sys.exit(0)

# ==========================================
# GATILHO DE AUTO-CONFIGURAÇÃO (deve ficar antes dos outros imports)
# ==========================================
auto_configurar_ambiente()

# Daqui para baixo, o código só executa quando já está no ambiente virtual!
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Variáveis globais baseadas no treino[cite: 1]
TAM = (224, 224) 
CLASSES = ["saudavel", "requeima", "mancha_bacteriana"]
COLS = ["doenca", "agente", "tratamento"]

def carregar(caminho):
    """Lê a imagem e previne o erro de ficheiro não encontrado[cite: 1]."""
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Não foi possível encontrar ou ler a imagem no caminho: {caminho}")
    return img

def preprocessar(img):
    """Redimensiona, ajusta brilho/contraste, remove ruídos e normaliza[cite: 1]."""
    img_resized = cv2.resize(img, TAM)
    img_bc = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=15)
    img_blur = cv2.GaussianBlur(img_bc, (5, 5), 0)
    
    img8 = img_blur.copy() 
    x = img8.astype("float32") / 255.0 
    
    return img8, x

def estimar_severidade(img8):
    """Calcula a percentagem da área da folha afetada[cite: 1]."""
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    folha = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    lesao = cv2.inRange(hsv, (0, 40, 20), (25, 255, 200))
    
    total = cv2.countNonZero(folha) + cv2.countNonZero(lesao)
    sev = 100 * cv2.countNonZero(lesao) / max(total, 1)
    return sev

def classificar(x, modelo):
    """Faz a predição da imagem[cite: 1]."""
    probs = modelo.predict(np.expand_dims(x, axis=0))[0]
    indice_classe = int(np.argmax(probs))
    return CLASSES[indice_classe]

def consultar_ou_cadastrar(classe, db_path="doencas.csv"):
    """Consulta o CSV ou regista um caso novo[cite: 1]."""
    if os.path.exists(db_path):
        db = pd.read_csv(db_path)
    else:
        db = pd.DataFrame(columns=COLS)

    if classe == "saudavel":
        return "Planta sem sintomas aparentes."
    elif classe in db["doenca"].values:
        info = db[db["doenca"] == classe].iloc[0]
        return f"Diagnóstico: {info['doenca']}\nAgente: {info['agente']}\nTratamento: {info['tratamento']}"
    else:
        db.loc[len(db)] = [classe, "agente?", "tratamento?"]
        db.to_csv(db_path, index=False)
        return f"Nova doença detetada e registada no sistema: {classe}"

def principal():
    print("="*50)
    print(" SISTEMA DE DIAGNÓSTICO DE FITOPATIAS DO TOMATEIRO ")
    print("="*50)
    
    if not os.path.exists("tomate_cnn.keras"):
        print("\n[AVISO]: O ficheiro 'tomate_cnn.keras' não foi encontrado na pasta atual.")
        print("A rede neural precisa de estar treinada e o ficheiro guardado para a análise funcionar[cite: 1].")
        return
    
    # Oculta mensagens de log pesadas do TensorFlow no terminal
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    modelo_cnn = tf.keras.models.load_model("tomate_cnn.keras")
    
    caminho_imagem = input("\nPor favor, introduza o caminho completo para a foto da planta: ")
    caminho_imagem = caminho_imagem.strip().strip("'\"") 
    
    try:
        img = carregar(caminho_imagem)
        img8, x = preprocessar(img)
        
        print("\nA analisar a imagem com a rede neural...")
        classe = classificar(x, modelo_cnn)
        resultado = consultar_ou_cadastrar(classe)
        
        print("\n" + "-"*40)
        print("RESULTADO DO DIAGNÓSTICO:")
        print("-"*40)
        print(resultado)
        
        # Segue a lógica do diagrama: se não for saudável, estima severidade[cite: 2]
        if classe != "saudavel":
            severidade = estimar_severidade(img8)
            print(f"\nSeveridade estimada: {severidade:.1f}% da área da folha")
        print("-"*40)
        
    except Exception as e:
        print(f"\n[ERRO]: Não foi possível concluir o diagnóstico: {e}")

if __name__ == "__main__":
    principal()



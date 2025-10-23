from openai import OpenAI
from datasets import load_dataset
import re
import random

# server local 
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# DATASETS LOAD
print("Carregando datasets completos...")
FULL_BOOLQ_DS = load_dataset("google/boolq", split="validation")
FULL_HELLASWAG_DS = load_dataset("hellaswag", split="validation")
print("Datasets carregados.")
# ---------------------------------------------------

def run_boolq(full_dataset, n_samples):
    shuffled_ds = full_dataset.shuffle()
    dataset = shuffled_ds.select(range(n_samples))

    correct = 0
    for idx, ex in enumerate(dataset):
        question = ex["question"]
        passage = ex["passage"]
        label = ex["answer"]  # True/False

        prompt = f"""
Leia o parágrafo e responda a pergunta com apenas 'Sim' ou 'Não'.

Parágrafo: {passage}
Pergunta: {question}
"""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            output = response.choices[0].message.content.strip().lower()
            prediction = "sim" in output

            if prediction == label:
                correct += 1
        except Exception as e:
            print(f"Erro na execução do BoolQ (idx={idx}): {e}")
            continue

    acc = correct / n_samples
    return acc

def run_hellaswag(full_dataset, n_samples):
    # Embaralha e seleciona N amostras
    shuffled_ds = full_dataset.shuffle()
    dataset = shuffled_ds.select(range(n_samples))

    correct = 0
    # debug: mostra quantas amostras e campos
    # print(f"[DEBUG] HellaSwag dataset length: {len(dataset)}")

    for idx, ex in enumerate(dataset):
        # Campos esperados no HF HellaSwag: 'ctx', 'endings' (lista de 4), 'label'
        ctx = ex.get("ctx") or ex.get("context") or ""
        endings = ex.get("endings") or ex.get("ending_candidates") or []
        
        raw_label = ex.get("label")
        try:
            label = int(raw_label)
        except Exception:
            # fallback: se não houver label coloca -1 (invalido)
            label = -1

        # se endings não tem 4 opções, pula (defensivo)
        if not isinstance(endings, list) or len(endings) < 4:
            print(f"[WARN] Exemplo idx={idx} com 'endings' inesperado, pulando.")
            continue

        prompt = f"""
Complete o texto a seguir escolhendo a continuação correta (0 a 3):

Contexto:
{ctx}

Opções:
0: {endings[0]}
1: {endings[1]}
2: {endings[2]}
3: {endings[3]}

Responda apenas com o número da opção (0, 1, 2 ou 3).
"""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            output = response.choices[0].message.content.strip()

            # parsing robusto: primeiro busca um dígito isolado 0-3, senão pega qualquer 0-3
            match = re.search(r"\b[0-3]\b", output)
            if not match:
                match = re.search(r"[0-3]", output)

            prediction = int(match.group()) if match else -1

            if prediction == label and label != -1:
                correct += 1
            else:
                # debug quando não conseguiu parsear
                if prediction == -1:
                    print(f"[DEBUG HellaSwag - Falha de Parsing] idx={idx} Output: '{output}'")
                # opcional: imprimir quando label inválido
                if label == -1:
                    print(f"[DEBUG HellaSwag - Label inválido] idx={idx} raw_label={raw_label}")

        except Exception as e:
            print(f"Erro na execução do HellaSwag (idx={idx}): {e}")
            continue

    acc = correct / n_samples
    return acc

def run_experiment(test_func, full_dataset, test_name, n_runs, n_samples_per_run):
    accuracies = []

    print(f"\n--- Iniciando {test_name} - {n_runs} Execuções de {n_samples_per_run} Amostras Aleatórias Cada ---")

    for i in range(1, n_runs + 1):
        acc = test_func(full_dataset=full_dataset, n_samples=n_samples_per_run)
        accuracies.append(acc)
        print(f"[{test_name} - Execução {i:02d}/{n_runs}] Acurácia: {acc:.4f}")

    average_acc = sum(accuracies) / len(accuracies)

    print("\n" + "="*50)
    print(f"**Resultados Finais do Experimento: {test_name}**")
    print("="*50)

    print("\n1. Resultados Individuais das {} Execuções (para Planilhar):".format(n_runs))
    for i, a in enumerate(accuracies):
        print(f"  Amostra {i+1:02d}: {a:.4f}")

    print(f"\n2. Média das {n_runs} Execuções: {average_acc:.4f}")
    print("="*50)

    return accuracies

if __name__ == "__main__":
    N_RUNS = 20
    N_SAMPLES = 20

    boolq_results = run_experiment(
        test_func=run_boolq,
        full_dataset=FULL_BOOLQ_DS,
        test_name="BoolQ",
        n_runs=N_RUNS,
        n_samples_per_run=N_SAMPLES
    )

    hellaswag_results = run_experiment(
        test_func=run_hellaswag,
        full_dataset=FULL_HELLASWAG_DS,
        test_name="HellaSwag",
        n_runs=N_RUNS,
        n_samples_per_run=N_SAMPLES
    )

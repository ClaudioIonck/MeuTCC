from openai import OpenAI
from datasets import load_dataset
import re

# server local 
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# boolq teste
def run_boolq(n_samples=50):
    dataset = load_dataset("google/boolq", split="validation[:{}]".format(n_samples))
    correct = 0

    for ex in dataset:
        question = ex["question"]
        passage = ex["passage"]
        label = ex["answer"]  # True/False

        prompt = f"""
        Leia o parágrafo e responda a pergunta com apenas 'Sim' ou 'Não'.

        Parágrafo: {passage}
        Pergunta: {question}
        """

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = response.choices[0].message.content.strip().lower()
        prediction = "sim" in output

        if prediction == label:
            correct += 1

    acc = correct / n_samples
    print(f"[BoolQ] Acurácia em {n_samples} amostras: {acc:.2f}")

# hellaswag teste
def run_hellaswag(n_samples=50):
    dataset = load_dataset("hellaswag", split="validation[:{}]".format(n_samples))
    correct = 0

    for ex in dataset:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])  # resposta correta

        prompt = f"""
        Complete o texto escolhendo a alternativa mais plausível (responda só com o número).

        Contexto: {ctx}

        Alternativas:
        0: {endings[0]}
        1: {endings[1]}
        2: {endings[2]}
        3: {endings[3]}
        """

        response = client.chat.completions.create(
            model="gpt-3", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = response.choices[0].message.content.strip()
        match = re.search(r"[0-3]", output)
        prediction = int(match.group()) if match else -1

        if prediction == label:
            correct += 1

    acc = correct / n_samples
    print(f"[HellaSwag] Acurácia em {n_samples} amostras: {acc:.2f}")

# quantidade dos testes
if __name__ == "__main__":
    run_boolq(20)      # 20 amostras
    run_hellaswag(20)  # 20 amostras

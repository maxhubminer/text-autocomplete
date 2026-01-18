
import evaluate

def evaluate_rouge(predictions, references):
    rouge = evaluate.load("rouge")

    # Подсчёт метрик
    results = rouge.compute(predictions=predictions, references=references)
    print('Metrics')
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # пример сгенерированного автодополнения
    print('Examples:')
    print(f"ref: {references[0]}, prediction: {predictions[0]}")
    print(f"ref: {references[10]}, prediction: {predictions[10]}")
    print(f"ref: {references[99]}, prediction: {predictions[99]}")
    print(f"ref: {references[250]}, prediction: {predictions[250]}")
    print(f"ref: {references[500]}, prediction: {predictions[500]}")
    print(f"ref: {references[333]}, prediction: {predictions[333]}")
    
    
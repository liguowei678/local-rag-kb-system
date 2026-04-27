"""
RAGAS 评测（完整5指标）
用法:
  python run_evaluation.py collect
  python run_evaluation.py evaluate
  python run_evaluation.py evaluate --compare baseline_results.json   ← 和基线对比
"""
import os, sys, json, csv, time, requests
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RAG_URL = "http://localhost:8000/api/v1/qa-hybrid-graphrag/query"
DATA_PATH = EVAL_DIR / "test_dataset.csv"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_PATH = EVAL_DIR / "baseline_results.json"

# 阈值：每个指标的允许下降绝对值
THRESHOLDS = {
    "faithfulness": 0.05,
    "answer_relevancy": 0.05,
    "context_recall": 0.05,
    "context_precision": 0.05,
    "answer_correctness": 0.05,
}


def collect():
    with open(DATA_PATH, encoding="gbk") as f:
        rows = [r for r in csv.DictReader(f)][:50]

    outputs = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    if DATASET_PATH.exists():
        with open(DATASET_PATH, encoding="utf-8") as f:
            outputs = json.load(f)
        done = len(outputs["question"])
    else:
        done = 0

    total = len(rows)
    print(f"收集 {total} 条，已有 {done} 条，use_graphrag=True...", flush=True)

    for i in range(done, total):
        q = rows[i]["query"].strip()
        gt = rows[i].get("ground_truth", "").strip()
        time.sleep(2)
        body = {"query": q, "top_k": 2, "use_graphrag": True}
        try:
            r = requests.post(RAG_URL, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            docs = data.get("documents") or []
            contexts = [d.get("content", str(d)) if isinstance(d, dict) else str(d) for d in docs]
        except Exception as e:
            print(f"  [{i+1}/{total}] ✗ {e}", flush=True)
            continue

        outputs["question"].append(q)
        outputs["answer"].append(answer)
        outputs["contexts"].append(contexts)
        outputs["ground_truth"].append(gt)

        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False)
        print(f"  [{i+1}/{total}] ✓ 累计 {len(outputs['question'])} 条", flush=True)

    print(f"收集完成: {len(outputs['question'])} 条 → {DATASET_PATH}", flush=True)


def evaluate(compare_path=None):
    if not DATASET_PATH.exists():
        print("请先 collect")
        return
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    import instructor
    from pydantic import BaseModel, Field
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "deepseek-key")
    client = instructor.patch(OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1"))

    class StatementFaithfulnessAnswer(BaseModel):
        statement: str
        reason: str
        verdict: int = Field(description="1 if supported by context, 0 if not")

    class NLIStatementOutput(BaseModel):
        statements: list[StatementFaithfulnessAnswer]

    class StatementGenOutput(BaseModel):
        statements: list[str]

    def calc_faithfulness(question, answer, contexts, client):
        stmt_resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=StatementGenOutput,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""Break down each sentence in the answer into one or more fully understandable factual statements. No pronouns.

Question: {question}
Answer: {answer}

Return JSON with a "statements" list."""}],
        )
        statements = stmt_resp.statements
        if not statements:
            return 1.0
        context_text = "\n".join(contexts) if contexts else ""
        if not context_text:
            return 0.0
        verdict_resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=NLIStatementOutput,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""Context: {context_text}

For each statement, return verdict 1 if directly inferred from context, else 0.

Statements:
{chr(10).join(f"- {s}" for s in statements)}

Return JSON with a "statements" list."""}],
        )
        verdicts = [v.verdict for v in verdict_resp.statements]
        return sum(verdicts) / len(verdicts) if verdicts else 1.0

    def calc_answer_relevancy(question, answer, client):
        class RelevancyOutput(BaseModel):
            score: float = Field(description="0.0 to 1.0, how directly and completely the answer addresses the question")
            reason: str
        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=RelevancyOutput,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Evaluate the Relevancy of this answer to the question.
Relevancy: how directly and completely the answer addresses the question.

Question: {question}
Answer: {answer}

Return a JSON with score (0.0-1.0) and a brief reason."""}],
        )
        return resp.score

    def calc_context_recall(contexts, ground_truth, client):
        class RecallOutput(BaseModel):
            score: float = Field(description="0.0 to 1.0, proportion of key info in ground_truth that is covered by contexts")
            reason: str
        if not contexts or not ground_truth:
            return 0.0
        context_text = "\n".join(contexts)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=RecallOutput,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Contexts:
{context_text}

Ground Truth:
{ground_truth}

Evaluate Context Recall: what proportion of the key information in the ground truth can be found in the contexts? Return score (0.0-1.0)."""}],
        )
        return resp.score

    def calc_context_precision(contexts, question, client):
        class PrecisionOutput(BaseModel):
            score: float = Field(description="0.0 to 1.0, proportion of contexts that are relevant to the question")
            reason: str
        if not contexts:
            return 0.0
        context_text = "\n".join(contexts)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=PrecisionOutput,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Question: {question}

Contexts:
{context_text}

Evaluate Context Precision: how many of the retrieved context chunks are truly relevant to the question? Return score (0.0-1.0)."""}],
        )
        return resp.score

    def calc_answer_correctness(answer, ground_truth, client):
        class CorrectnessOutput(BaseModel):
            score: float = Field(description="0.0 to 1.0, factually and semantically how well the answer matches ground truth")
            reason: str
        if not answer or not ground_truth:
            return 0.0
        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_model=CorrectnessOutput,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Ground Truth:
{ground_truth}

Answer:
{answer}

Evaluate Answer Correctness: how well does the answer match the ground truth, both factually and semantically? Return score (0.0-1.0)."""}],
        )
        return resp.score

    total = len(data["question"])
    print(f"评测 {total} 条（5个指标）...", flush=True)

    results = {
        "faithfulness": [], "answer_relevancy": [],
        "context_recall": [], "context_precision": [], "answer_correctness": [],
    }

    for i in range(total):
        question = data["question"][i]
        answer = data["answer"][i]
        contexts = data["contexts"][i]
        ground_truth = data["ground_truth"][i]

        sys.stderr.write(f"\r  [{i+1}/{total}] fth...")
        sys.stderr.flush()
        try:
            r = calc_faithfulness(question, answer, contexts, client)
        except Exception:
            r = 0.0
        results["faithfulness"].append(r)

        sys.stderr.write(f" ar...")
        sys.stderr.flush()
        try:
            r = calc_answer_relevancy(question, answer, client)
        except Exception:
            r = 0.0
        results["answer_relevancy"].append(r)

        sys.stderr.write(f" cr...")
        sys.stderr.flush()
        try:
            r = calc_context_recall(contexts, ground_truth, client)
        except Exception:
            r = 0.0
        results["context_recall"].append(r)

        sys.stderr.write(f" cp...")
        sys.stderr.flush()
        try:
            r = calc_context_precision(contexts, question, client)
        except Exception:
            r = 0.0
        results["context_precision"].append(r)

        sys.stderr.write(f" ac...")
        sys.stderr.flush()
        try:
            r = calc_answer_correctness(answer, ground_truth, client)
        except Exception:
            r = 0.0
        results["answer_correctness"].append(r)

        sys.stderr.write(f" ✓\n")
        sys.stderr.flush()

        if (i + 1) % 3 == 0 or i == total - 1:
            partial = {}
            for k, v in results.items():
                valid = [x for x in v if x is not None]
                avg = sum(valid) / len(valid) if valid else 0.0
                partial[k] = round(avg, 4)
            partial["samples"] = i + 1
            partial["progress"] = f"{i+1}/{total}"
            partial["timestamp"] = time.time()
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)
            print(f"     中间结果({i+1}/{total}): {partial}", flush=True)

    # 最终结果
    final = {}
    for k, v in results.items():
        valid = [x for x in v if x is not None]
        avg = sum(valid) / len(valid) if valid else 0.0
        final[k] = round(avg, 4)
    final["samples"] = total
    final["timestamp"] = time.time()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n📊 完整评测结果（{total} 条）:")
    for k, v in final.items():
        if k in ("samples", "timestamp"):
            continue
        print(f"  {k}: {v}")
    print(f"\n结果 → {RESULTS_PATH}")

    # ── 与基线对比 ──
    if compare_path:
        compare_path = Path(compare_path)
        if not compare_path.exists():
            print(f"\n⚠ 基线文件不存在: {compare_path}，跳过对比", flush=True)
            return
        with open(compare_path, encoding="utf-8") as f:
            baseline = json.load(f)

        print(f"\n📊 与基线对比（{compare_path.name}）:")
        print(f"{'指标':<22} {'新结果':<10} {'基线':<10} {'差值':<10} {'状态'}")
        failed = False
        for metric in THRESHOLDS:
            new_val = final.get(metric, 0)
            base_val = baseline.get(metric, 0)
            diff = round(new_val - base_val, 4)
            threshold = THRESHOLDS[metric]
            if diff < -threshold:
                status = "✗ 下降超限"
                failed = True
            else:
                status = "✓"
            print(f"{metric:<22} {new_val:<10} {base_val:<10} {diff:<+10}  {status}")

        if failed:
            print("\n❌ CI/CD 检查不通过：部分指标下降超过阈值")
            sys.exit(1)
        else:
            print("\n✅ CI/CD 检查通过：所有指标在阈值范围内")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    # 检测 --compare 参数（可以是第三个位置参数或 --compare=xxx）
    compare_path = None
    remaining = sys.argv[2:]
    for arg in remaining:
        if arg.startswith("--compare="):
            compare_path = arg.split("=", 1)[1]
        elif arg == "--compare" and len(remaining) > remaining.index(arg) + 1:
            compare_path = remaining[remaining.index(arg) + 1]

    if cmd == "collect":
        collect()
    elif cmd == "evaluate":
        evaluate(compare_path=compare_path)
    elif cmd == "all":
        collect()
        evaluate(compare_path=compare_path)

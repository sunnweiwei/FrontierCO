# FrontierCO: A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization


FrontierCO is a curated benchmark suite for evaluating ML-based solvers on large-scale and real-world Combinatorial Optimization (CO) problems. The benchmark spans 8 classical CO problems across 5 application domains, providing both training and evaluation instances specifically designed to test the frontier of ML and LLM capabilities in solving NP-hard problems.

Combinatorial optimization plays a fundamental role in discrete mathematics, computer science, and operations research, with applications in routing, scheduling, allocation, and more. As ML-based solvers evolve—ranging from neural networks to symbolic reasoning with large language models—FrontierCO offers the first comprehensive dataset suite tailored to test these solvers at realistic scales and difficulties.


# Download Data
Download the raw data from [https://huggingface.co/datasets/CO-Bench/FrontierCO](https://huggingface.co/datasets/CO-Bench/FrontierCO) to the local directory `data`
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='CO-Bench/FrontierCO',
    repo_type='dataset',
    local_dir='data'
)
```

# Classical Solvers and Neural Training Data

Please refer to the instructions under each problem folder for how to: 

- apply the human-designed classical solvers 
- generate the training data for neural solvers 

All the neural solvers evaluated in this work are open-sourced models. Please refer to their official GitHub repos for the training and evaluation code.


# Agent Evaluation
Below is code to run evaluation of *Greedy Refinement* agent on `CFLP` for 64 iterations with 300s timeout.

```python
from agents import GreedyRefine, DirectAnswer, FunSearch, AIDE, ChainOfExperts, ReEvo, BestOfN
from evaluation import YieldingEvaluator, get_data

# Load data
data = get_data('CFLP', src_dir='data')

# Define agent, here we use GreedyRefine
agent = GreedyRefine(
    problem_description=data.problem_description,
    timeout=300,
    model='openai/o3-mini', # We use LiteLLM to call API
)

# Load evaluator
evaluator = YieldingEvaluator(data, timeout=300)

# Run for 64 iterations
for it in range(64):
    code = agent.step()
    if code is None:  # agent decides to terminate
        break
    feedback = evaluator.evaluate(code)  # Run evaluation
    agent.feedback(feedback.dev_score, feedback.dev_feedback)  # Use dev set score as feedback

# Get the final solution
code = agent.finalize()
feedback = evaluator.evaluate(code)
print(feedback.test_feedback)  # Test set score
```

# Agent Implementations

Agents are implemented in the `agents` module. Currently supported agents include: `GreedyRefine`, `DirectAnswer`, `BestOfN`, `FunSearch` ([link](https://github.com/google-deepmind/funsearch)), `AIDE` ([link](https://github.com/WecoAI/aideml)), `ChainOfExperts` ([link](https://github.com/xzymustbexzy/Chain-of-Experts)), and `ReEvo` ([link](https://github.com/ai4co/reevo)). LLMs are supported via [liteLLM](https://github.com/BerriAI/litellm).

Each agent implements the following functions:
- `step()`: Returns the next candidate code for evaluation.




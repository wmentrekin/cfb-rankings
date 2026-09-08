# Model Evaluation

Maintain a small representative evaluation set to determine whether cheaper routing preserves
accepted output quality in the repositories where `$work` is used.

## Evaluation Set

Collect 8 to 15 sanitized tasks for each materially different agent profile. Include routine and
difficult examples from:

- repository and asset discovery
- implementation
- plan and code review
- test design and test execution
- debugging
- orchestration decisions

Do not use only public coding benchmarks. Preserve task inputs, acceptance criteria, and an
objective or reviewer-scored expected result.

## Metrics

Record:

- accepted on first attempt
- accepted after escalation
- defects found by review
- retries and correction loops
- tool calls
- input and output tokens when available
- wall-clock time
- API cost or provider quota consumed when available
- model and reasoning configuration actually used

The primary comparison is cost or quota per accepted task. A cheaper invocation that creates
more retries, review work, or defects may be more expensive overall.

## Procedure

1. run the cheapest plausible tier for the task profile
2. score the result against the same acceptance criteria
3. escalate one tier only after a failed or materially weak result
4. compare quality, total attempts, total token use, elapsed time, and accepted-task cost
5. update the default routing only when results are consistent across representative tasks

Avoid using live production incidents, secrets, destructive operations, or tasks whose expected
result cannot be scored consistently.

## Refresh Cadence

Review provider mappings quarterly and whenever a relevant model is launched, removed, repriced,
or changes its reasoning controls. Re-run only the affected profiles first; a full suite is
necessary only when routing decisions materially change.

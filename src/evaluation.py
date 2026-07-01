from discretization import discretize_base_state, discretize_disturbed_state


def evaluate_base_policy(agent, env, cap_steps=None, logs=0):
    state = discretize_base_state(env.reset())
    steps = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        while True:
            action = agent.select_action(state)
            next_cont, _, done = env.step(action)
            state = discretize_base_state(next_cont)
            steps += 1

            if logs and steps % logs == 0:
                print(f"[policy] survived {steps} steps...")
            if done:
                return steps, "failure"
            if cap_steps is not None and steps >= cap_steps:
                return steps, "cap"
    finally:
        agent.epsilon = old_epsilon


def evaluate_disturbed_policy(agent, env, cap_steps=None, drunk=False, logs=0):
    state = discretize_disturbed_state(env.reset(drunk=drunk))
    steps = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        while True:
            action = agent.select_action(state)
            env_action = -1 if action == 0 else 1
            next_cont, _, done = env.step(env_action)
            state = discretize_disturbed_state(next_cont)
            steps += 1

            if logs and steps % logs == 0:
                print(f"[policy] survived {steps} steps...")
            if done:
                return steps, "failure"
            if cap_steps is not None and steps >= cap_steps:
                return steps, "cap"
    finally:
        agent.epsilon = old_epsilon

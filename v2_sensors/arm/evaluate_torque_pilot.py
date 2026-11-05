"""Six-case validation using the frozen simulator and torque-only inference."""
import full_run
from pilot_torque_runtime import policy
import evaluate_curriculum

# Install on import too: forkserver workers import this entry point afresh.
full_run.arm_policy = policy

if __name__ == '__main__':
    evaluate_curriculum.main()

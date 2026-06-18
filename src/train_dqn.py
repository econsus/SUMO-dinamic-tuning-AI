import os
import sys
import argparse
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_proj_root, "src")
for p in [_proj_root, _src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from data_repo import DataRepo
from data.traffic_data import import_records, data_records
from envs.sumo_env import SUMOEnv
from agents.dqn_agent import DQNAgent
from logger import TrainingLogger


def extract_target_data() -> np.ndarray:
    import_records()
    rows = []
    for row in data_records:
        vals = {}
        for d in row:
            vals.update(d)
        sl = vals.get("SL") or 0
        spt = vals.get("SPT") or 0
        ul = vals.get("UL") or 0
        upt = vals.get("UPT") or 0
        os_val = vals.get("OS") or 0
        ou_val = vals.get("OU") or 0
        if os_val == 0 or ou_val == 0:
            continue
        rows.append([sl, spt, ul, upt, os_val, ou_val])
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="DQN training for SUMO calibration")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sumo-binary", default="sumo")

    parser.add_argument("--warmup-minutes", type=int, default=5,
                        help="SUMO warm-up duration before RL starts (default: 5)")
    parser.add_argument("--period-minutes", type=int, default=5,
                        help="Duration of each RL step / reward window (default: 5)")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--replay-per-step", type=int, default=5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.7)
    args = parser.parse_args()

    target_data = extract_target_data()
    n_data = len(target_data)
    print(f"Loaded {n_data} data point(s)")
    for i, row in enumerate(target_data):
        print(f"  [{i}] SL={row[0]:.0f}  SPT={row[1]:.0f}  UL={row[2]:.0f}  UPT={row[3]:.0f}  "
              f"OS={row[4]:.0f}  OU={row[5]:.0f}")

    if n_data == 0:
        print("No data points to train on.")
        return

    repo = DataRepo()
    env = SUMOEnv(
        sumo_config_path=str(repo.sumo_config_path),
        sumo_binary=args.sumo_binary,
        warmup_minutes=args.warmup_minutes,
        period_minutes=args.period_minutes,
    )
    env.set_target_data(target_data)

    agent = DQNAgent(
        state_dim=4,
        action_dim=env.n_actions,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
    )

    logger = TrainingLogger(log_dir="logs", hyperparams=vars(args))
    epsilon = args.epsilon_start

    print(f"\nRunning {args.iterations} iteration(s)...")
    print(f"Logs -> {logger.run_dir}\n")

    for iteration in range(args.iterations):
        env.reset_data_pointer()
        total_iter_reward = 0.0
        total_steps = 0
        iter_loss_sum = 0.0
        iter_loss_count = 0

        for data_idx in range(n_data):
            obs, _ = env.reset()
            data_reward = 0.0

            for step_in_data in range(4):
                action = agent.act(obs, epsilon)
                next_obs, reward, terminated, _, info = env.step(action)
                agent.remember(obs, action, reward, next_obs, terminated)

                step_loss = None
                for _ in range(args.replay_per_step):
                    l = agent.replay()
                    if l is not None:
                        step_loss = l
                if step_loss is not None:
                    iter_loss_sum += step_loss
                    iter_loss_count += 1

                logger.log_step(
                    iteration=iteration + 1,
                    data_idx=data_idx,
                    step_in_data=step_in_data + 1,
                    reward=reward,
                    sim_south=info["sim_south"],
                    sim_north=info["sim_north"],
                    expected_south=info["expected_south"],
                    expected_north=info["expected_north"],
                    lcCooperative=info["params"]["lcCooperative"],
                    lcAssertive=info["params"]["lcAssertive"],
                    epsilon=epsilon,
                    loss=step_loss,
                )

                obs = next_obs
                data_reward += reward
                total_steps += 1

            total_iter_reward += data_reward

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        avg_reward = total_iter_reward / (n_data * 4) if n_data > 0 else 0.0
        avg_loss = (iter_loss_sum / iter_loss_count) if iter_loss_count > 0 else None

        logger.log_iteration(
            iteration=iteration + 1,
            total_reward=total_iter_reward,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            epsilon=epsilon,
            total_steps=total_steps,
        )

        loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
        print(f"  Iter {iteration + 1:2d}/{args.iterations}  "
              f"total={total_iter_reward:+9.1f}  "
              f"avg={avg_reward:+7.3f}  "
              f"loss={loss_str}  "
              f"eps={epsilon:.3f}  "
              f"buf={len(agent.replay_buffer):4d}  "
              f"steps={total_steps}")

    final_ckpt = logger.run_dir / "dqn_final.pt"
    agent.save(str(final_ckpt))
    logger.close()
    print(f"\nDone. Model saved to {final_ckpt}")
    env.close()


if __name__ == "__main__":
    main()

// pm2 process for training variant A:  pm2 start ecosystem.config.js
// Runs the pipeline's train stage with the venv's python (no activation needed). A crash restarts
// it after a minute and it resumes from out/runs/A/checkpoint; a clean finish (exit 0) stays stopped.
const ROOT = __dirname;

module.exports = {
  apps: [
    {
      name: "ccslm-train-A",
      cwd: ROOT,
      script: `${ROOT}/.venv/bin/python`,
      args: "-m ccslm.pipeline --stages train --variant A",
      interpreter: "none",
      env: {
        HF_HOME: "/workspace/hf",
        PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True",
        PYTHONUNBUFFERED: "1",
        CCSLM_TRAIN_ARGS: "--max-tokens 96000 --save-every 10 --eval-every 25 --eval-items 16",
      },
      autorestart: true,
      stop_exit_codes: [0],
      max_restarts: 5,
      min_uptime: "10m",
      restart_delay: 60000,
      kill_timeout: 30000,
      out_file: `${ROOT}/out/logs/pm2-train-A.log`,
      error_file: `${ROOT}/out/logs/pm2-train-A.log`,
      merge_logs: true,
      time: true,
    },
  ],
};

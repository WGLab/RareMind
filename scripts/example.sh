sbatch -p gpuq \
        --gres=gpu:a100-40g:1 \
        --cpus-per-gpu=1 \
        --mem-per-cpu=20G \
        --time=5-00:00:00 \
        --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
                --input_dir /home/nguyenqm/projects/RareMind/inputs \
                --dataset chop_time_lead_example \
                --output_dir /home/nguyenqm/projects/RareMind/outputs_qwen3 \
                --config /home/nguyenqm/projects/RareMind/configs/default.yaml"


sbatch -p gpu-xe9680q \
        --gres=gpu:h100:1 \
        --cpus-per-gpu=1 \
        --mem-per-cpu=20G \
        --time=9-00:00:00 \
        --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
                --input_dir /home/nguyenqm/projects/RareMind/inputs \
                --dataset GMDB_text_part1 \
                --output_dir /home/nguyenqm/projects/RareMind/outputs_qwen3 \
                --config /home/nguyenqm/projects/RareMind/configs/default.yaml"

sbatch -p gpu-xe9680q \
        --gres=gpu:h100:1 \
        --cpus-per-gpu=1 \
        --mem-per-cpu=20G \
        --time=9-00:00:00 \
        --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
                --input_dir /home/nguyenqm/projects/RareMind/inputs \
                --dataset GMDB_image_part1 \
                --output_dir /home/nguyenqm/projects/RareMind/outputs_qwen3 \
                --config /home/nguyenqm/projects/RareMind/configs/default_image.yaml"

sbatch -p gpu-xe9680q \
        --gres=gpu:h100:1 \
        --cpus-per-gpu=1 \
        --mem-per-cpu=20G \
        --time=9-00:00:00 \
        --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
                --input_dir /home/nguyenqm/projects/RareMind/inputs \
                --dataset GMDB_text_part2 \
                --output_dir /home/nguyenqm/projects/RareMind/outputs_qwen3 \
                --config /home/nguyenqm/projects/RareMind/configs/default.yaml"

sbatch -p gpu-xe9680q \
        --gres=gpu:h100:1 \
        --cpus-per-gpu=1 \
        --mem-per-cpu=20G \
        --time=9-00:00:00 \
        --wrap="bash /home/nguyenqm/projects/RareMind/scripts/run_agents.sh \
                --input_dir /home/nguyenqm/projects/RareMind/inputs \
                --dataset GMDB_image_part2 \
                --output_dir /home/nguyenqm/projects/RareMind/outputs_qwen3 \
                --config /home/nguyenqm/projects/RareMind/configs/default_image.yaml"
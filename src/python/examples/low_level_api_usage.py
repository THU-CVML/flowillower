# simulate_api_usage.py
import tempfile
from pathlib import Path
import shutil
import tomli_w # For writing the manifest
from math import sin
import random

# 假设 flowillower 包在PYTHONPATH中，或者此脚本与 flowillower 目录同级
# Assuming flowillower package is in PYTHONPATH, or this script is at the same level as the flowillower directory
try:
    from flowillower.viskits.scalar_dashboard_viskit import ScalarMetricsDashboardVisKit
    from flowillower.viskits.pygwalker_viskit import PygwalkerDashboardVisKit
    # 如果有其他视件，也在这里导入
    # If other Viskits exist, import them here too
except ImportError as e:
    print(f"错误：无法导入Flowillower视件。请确保 'flowillower' 包在PYTHONPATH中。\n{e}")
    print("提示：您可以尝试在 flowillower_project_root 目录下运行此脚本，或者使用 'pip install -e .' 安装flowillower包。")
    exit(1)

def run_simulated_trial(trial_name: str, base_path: Path):
    """
    模拟一次实验的运行，记录数据到多个视件。
    Simulates a single experiment run, logging data to multiple Viskits.
    """
    trial_root_path = base_path / trial_name
    if trial_root_path.exists():
        print(f"警告：Trial目录 '{trial_root_path}' 已存在。其中的内容可能会被覆盖。")
        # shutil.rmtree(trial_root_path) # 可选：如果需要，则清除旧目录
    trial_root_path.mkdir(parents=True, exist_ok=True)
    print(f"Trial 根目录已创建/确认: {trial_root_path}")

    # --- 1. 实例化视件 ---
    # --- 1. Instantiate Viskits ---
    # API侧实例化视件时，data_sources_map 通常为空，因为这些视件主要通过 report_data 创建自己的数据。
    # Specific UI config 也可以在这里传递，如果API侧希望预设某些UI表现。
    # When Viskits are instantiated on the API side, data_sources_map is typically empty,
    # as these Viskits primarily create their own data via report_data.
    # Specific UI config can also be passed here if the API side wants to preset UI appearance.

    scalar_viskit_id = "main_scalar_dashboard"
    scalar_viskit = ScalarMetricsDashboardVisKit(
        instance_id=scalar_viskit_id,
        trial_root_path=trial_root_path,
        data_sources_map={}, # 初始为空，因为它会创建自己的数据资产
        specific_ui_config_dict={"charts_per_row": 2, "show_metric_summary": True} # 示例UI配置
    )
    print(f"已实例化 ScalarMetricsDashboardVisKit: {scalar_viskit_id}")

    pygwalker_viskit_id = "interactive_metric_explorer"
    pygwalker_viskit = PygwalkerDashboardVisKit(
        instance_id=pygwalker_viskit_id,
        trial_root_path=trial_root_path,
        data_sources_map={}, # 初始为空
        specific_ui_config_dict={"theme_key": "streamlit", "explorer_height": 750} # 示例UI配置
    )
    print(f"已实例化 PygwalkerDashboardVisKit: {pygwalker_viskit_id}")

    # --- 2. 模拟数据上报 ---
    # --- 2. Simulate Data Reporting ---
    all_trial_asset_descriptions = [] # 收集所有视件返回的资产描述
    num_steps = 30
    print(f"\n开始模拟 {num_steps} 个步骤的数据上报...")

    for step in range(num_steps):
        # 生成一些模拟数据
        # Generate some mock data
        current_loss = 1.0 / (step + 1) + random.uniform(-0.05, 0.05)
        current_accuracy = 0.7 + sin(step / 5) * 0.1 + random.uniform(-0.02, 0.02)
        
        data_payload_train = {
            "loss": current_loss,
            "accuracy": current_accuracy
        }
        data_payload_val = {
            "loss": current_loss + 0.1 + random.uniform(-0.03, 0.03),
            "accuracy": current_accuracy - 0.05 + random.uniform(-0.01, 0.01)
        }

        # 上报到 ScalarMetricsDashboardVisKit
        # Report to ScalarMetricsDashboardVisKit
        assets_from_scalar_train = scalar_viskit.report_data(
            data_payload=data_payload_train,
            step=step,
            group_id="training" # track name
        )
        all_trial_asset_descriptions.extend(assets_from_scalar_train)

        assets_from_scalar_val = scalar_viskit.report_data(
            data_payload=data_payload_val,
            step=step,
            group_id="validation"
        )
        all_trial_asset_descriptions.extend(assets_from_scalar_val)

        # 上报相同的数据到 PygwalkerDashboardVisKit
        # Report the same data to PygwalkerDashboardVisKit
        # Pygwalker 的 report_data 期望一个扁平的字典，其中可以包含 track 和 global_step
        # Pygwalker's report_data expects a flat dictionary, which can include track and global_step
        pyg_payload_train = {"global_step": step, "track": "training", **data_payload_train}
        assets_from_pyg_train = pygwalker_viskit.report_data(
            data_payload=pyg_payload_train,
            step=step # step 参数仍然需要，用于内部逻辑或作为主要步骤源
                     # step argument is still needed for internal logic or as primary step source
        )
        all_trial_asset_descriptions.extend(assets_from_pyg_train)

        pyg_payload_val = {"global_step": step, "track": "validation", **data_payload_val}
        assets_from_pyg_val = pygwalker_viskit.report_data(
            data_payload=pyg_payload_val,
            step=step
        )
        all_trial_asset_descriptions.extend(assets_from_pyg_val)

        if (step + 1) % 10 == 0:
            print(f"  已上报步骤 {step + 1}/{num_steps}")

    print("数据上报完成。")

    # --- 3. （模拟）创建/更新 Trial 数据清单 ---
    # --- 3. (Simulate) Create/Update Trial Data Manifest ---
    # 在实际的API库中，会有一个更复杂的逻辑来合并和去重资产描述。
    # 这里，我们简单地将所有唯一的资产描述（基于asset_id）收集起来。
    # In a real API library, there would be more complex logic to merge and deduplicate asset descriptions.
    # Here, we simply collect all unique asset descriptions (based on asset_id).
    
    final_manifest_assets = []
    seen_asset_ids = set()
    for desc in reversed(all_trial_asset_descriptions): # 从后向前，保留最新的描述
        asset_id = desc.get("asset_id")
        if asset_id and asset_id not in seen_asset_ids:
            final_manifest_assets.insert(0, desc) # 保持原始顺序（大致上）
            seen_asset_ids.add(asset_id)
    
    trial_manifest_content = {
        "trial_name": trial_name,
        "trial_root_path_comment": str(trial_root_path), # 仅供参考 For reference only
        "data_assets": final_manifest_assets
    }

    manifest_file_path = trial_root_path / "_trial_manifest.toml"
    try:
        with open(manifest_file_path, "wb") as f:
            tomli_w.dump(trial_manifest_content, f)
        print(f"\nTrial数据清单已（模拟）写入: {manifest_file_path}")
        print("清单内容 (Manifest Content):")
        # 打印部分内容以供预览 Print partial content for preview
        with open(manifest_file_path, "r", encoding="utf-8") as f_read:
            print(f_read.read())

    except Exception as e:
        print(f"写入清单文件失败: {e}")

    print(f"\n模拟实验 '{trial_name}' 完成。")
    print("您现在可以在 '视件视界' IDE 中指定以下路径作为 'Trial根路径' 来查看数据：")
    print(trial_root_path.resolve())
    print("然后选择相应的视件类型 (例如 'scalar_metrics_dashboard' 或 'pygwalker_interactive_dashboard')，")
    print(f"并使用实例ID (例如 '{scalar_viskit_id}' 或 '{pygwalker_viskit_id}')。")
    print("在IDE中实例化视件后，它应该能加载这里记录的数据。")


if __name__ == "__main__":
    # 使用临时目录作为所有trials的基础路径
    # Use a temporary directory as the base path for all trials
    # 这使得每次运行脚本时数据都是全新的，并且容易清理
    # This makes data fresh each time the script runs and easy to clean up
    
    # 检查用户是否想使用持久路径
    # Check if user wants to use a persistent path
    use_persistent_path = input("是否使用持久路径 './simulated_trials' 而不是临时目录? (y/N): ").strip().lower() == 'y'

    if use_persistent_path:
        from flowillower.help import runs_path
        sim_base_path = runs_path/"./simulated_trials"
        print(f"将使用持久路径: {sim_base_path.resolve()}")
    else:
        # 创建一个顶级的临时目录，用于本次脚本运行的所有trial
        # Create a top-level temporary directory for all trials of this script run
        # 这个目录在脚本结束时不会自动删除，以便用户可以在IDE中查看
        # This directory is NOT automatically deleted at script end, so user can view in IDE
        sim_base_path = Path(tempfile.mkdtemp(prefix="flowillower_sim_trials_"))
        print(f"所有模拟的Trial数据将存储在临时目录: {sim_base_path.resolve()}")
        print("提示：此目录在脚本结束后不会自动删除。您可以手动删除它。")


    sim_base_path.mkdir(parents=True, exist_ok=True)

    # 运行一个模拟的trial
    # Run a simulated trial
    run_simulated_trial("my_first_simulated_trial", sim_base_path)

    # （可选）运行另一个模拟的trial
    # (Optional) Run another simulated trial
    # run_simulated_trial("another_deep_learning_run", sim_base_path)

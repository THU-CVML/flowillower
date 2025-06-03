# flowillower/app/component_ide.py
import streamlit as st
from pathlib import Path
import tempfile
import json
import shutil
from typing import Dict, Any, Optional, Type, List
import tomli

# --- 模块导入 (使用基于包的绝对导入) ---
# --- Module Imports (Using package-based absolute imports) ---
# 假设 flowillower 包在 Python 路径中
# Assuming the flowillower package is in the Python path
try:
    from flowillower.visualizers.base_visualizer import (
        VISUALIZER_REGISTRY,
        get_visualizer_class,
        VisualizationComponent,
    )

    # 显式导入所有组件模块以确保它们被注册
    # Explicitly import all component modules to ensure they are registered
    import flowillower.visualizers.scalar_dashboard_visualizer
    import flowillower.visualizers.treescope_visualizer
    import flowillower.visualizers.pygwalker_visualizer

except ImportError as e:
    st.error(
        "错误：无法导入可视化组件模块。请确保您已正确设置项目结构，"
        "并且 'flowillower' 包位于您的PYTHONPATH中，或者您是从项目根目录运行此应用。"
        f"\n详细信息: {e}"
        "Error: Could not import visualization component modules. Ensure your project structure is correct, "
        "and the 'flowillower' package is in your PYTHONPATH, or you are running this app from the project root."
    )
    st.stop()

# --- 应用标题和配置 ---
# --- App Title and Configuration ---
st.set_page_config(
    layout="wide", page_title="视件视界 - Flowillower 组件IDE"
)  # 更新页面标题 Updated page title
st.title(
    "🔬 视件视界 - Flowillower 可视化组件IDE"
)  # 更新应用主标题 Updated app main title
st.markdown("在此环境中独立测试、调试和预览您的可视化组件。")

# --- 会话状态初始化 ---
# --- Session State Initialization ---
if "selected_visualizer_type_name" not in st.session_state:
    st.session_state.selected_visualizer_type_name = None
if "component_instance_id" not in st.session_state:
    st.session_state.component_instance_id = "ide_test_instance_001"
if "trial_root_path_str" not in st.session_state:
    st.session_state.trial_root_path_str = tempfile.mkdtemp(
        prefix="flowillower_ide_trial_"
    )
if "component_specific_config_str" not in st.session_state:
    st.session_state.component_specific_config_str = "{}"
if "active_visualizer_instance" not in st.session_state:
    st.session_state.active_visualizer_instance = None
if "generated_data_sources_map" not in st.session_state:
    st.session_state.generated_data_sources_map = None
if "current_simulated_global_step" not in st.session_state:
    st.session_state.current_simulated_global_step = 0
if "all_simulated_steps" not in st.session_state:
    st.session_state.all_simulated_steps = []


def cleanup_temp_dir(path_str):
    try:
        if (
            path_str
            and Path(path_str).exists()
            and "flowillower_ide_trial_" in path_str
        ):
            shutil.rmtree(path_str)
            st.toast(f"临时目录已清理: {path_str}")
    except Exception as e:
        st.warning(f"清理临时目录失败 {path_str}: {e}")


with st.sidebar:
    st.header("组件选择与配置")

    registered_type_names = list(VISUALIZER_REGISTRY.keys())
    if not registered_type_names:
        st.error("错误：没有已注册的可视化组件类型。")
        st.stop()

    default_selection_index = 0
    if st.session_state.selected_visualizer_type_name in registered_type_names:
        default_selection_index = registered_type_names.index(
            st.session_state.selected_visualizer_type_name
        )
    elif registered_type_names:
        st.session_state.selected_visualizer_type_name = registered_type_names[0]

    current_selected_type = st.selectbox(
        "选择可视化组件类型",
        options=registered_type_names,
        index=default_selection_index,
        help="选择您想要测试的可视化组件。",
    )
    # 仅当选择发生变化时更新会话状态，以避免不必要的rerun或状态冲突
    # Only update session state if selection changes to avoid unnecessary reruns or state conflicts
    if current_selected_type != st.session_state.selected_visualizer_type_name:
        st.session_state.selected_visualizer_type_name = current_selected_type
        # 清除与旧组件相关的数据，因为组件类型已更改
        # Clear data related to the old component as the type has changed
        st.session_state.active_visualizer_instance = None
        st.session_state.generated_data_sources_map = None
        st.session_state.all_simulated_steps = []
        st.session_state.current_simulated_global_step = 0
        st.rerun()

    SelectedVisualizerClass: Optional[Type[VisualizationComponent]] = (
        get_visualizer_class(st.session_state.selected_visualizer_type_name)
    )

    if SelectedVisualizerClass:
        st.caption(f"显示名称: `{SelectedVisualizerClass.get_display_name()}`")
    else:
        # 这种情况理论上不应发生，因为selectbox的选项来自注册表
        # This case should theoretically not happen as selectbox options come from the registry
        st.error(f"无法加载组件类 '{st.session_state.selected_visualizer_type_name}'。")
        st.stop()

    st.session_state.component_instance_id = st.text_input(
        "组件实例ID", value=st.session_state.component_instance_id
    )
    st.markdown(f"**临时Trial根路径:** `{st.session_state.trial_root_path_str}`")

    example_data_target_dir = (
        Path(st.session_state.trial_root_path_str) / "example_assets_for_ide"
    )

    example_gen_config_str = "{}"
    if (
        st.session_state.selected_visualizer_type_name == "treescope_model_viewer"
    ):  # 硬编码检查特定组件类型 Hardcoded check for specific component type
        example_gen_config_str = st.text_input(
            "示例数据生成配置 (JSON)",
            value='{"group_id": "my_ide_model_group"}',
            help='例如: {"group_id": "custom_group_name"} (特定于Treescope查看器)',
        )

    if st.button("生成示例数据"):
        if SelectedVisualizerClass:
            try:
                if example_data_target_dir.exists():
                    shutil.rmtree(example_data_target_dir)
                example_data_target_dir.mkdir(parents=True, exist_ok=True)

                parsed_example_gen_config = {}
                if example_gen_config_str:
                    try:
                        parsed_example_gen_config = json.loads(example_gen_config_str)
                    except json.JSONDecodeError:
                        st.warning("示例数据生成配置不是有效的JSON，将使用默认值。")

                st.session_state.generated_data_sources_map = (
                    SelectedVisualizerClass.generate_example_data(
                        example_data_path=example_data_target_dir,
                        data_sources_config=parsed_example_gen_config,
                    )
                )
                st.success(
                    f"'{SelectedVisualizerClass.get_display_name()}' 的示例数据已生成。"
                )

                temp_all_steps = set()
                if st.session_state.generated_data_sources_map:
                    for (
                        ds_name,
                        ds_info,
                    ) in st.session_state.generated_data_sources_map.items():
                        if (
                            isinstance(ds_info, dict)
                            and "items" in ds_info
                            and isinstance(ds_info["items"], list)
                        ):
                            for item in ds_info["items"]:
                                if isinstance(item, dict) and "related_step" in item:
                                    temp_all_steps.add(int(item["related_step"]))
                        elif isinstance(ds_info, dict) and "path" in ds_info:
                            try:
                                full_path = (
                                    Path(st.session_state.trial_root_path_str)
                                    / ds_info["path"]
                                )
                                if full_path.exists() and full_path.suffix == ".toml":
                                    with open(full_path, "rb") as f:
                                        d = tomli.load(f)
                                    if "metrics" in d and isinstance(
                                        d["metrics"], list
                                    ):
                                        for point in d["metrics"]:
                                            if "global_step" in point:
                                                temp_all_steps.add(
                                                    int(point["global_step"])
                                                )
                            except Exception:
                                pass

                st.session_state.all_simulated_steps = sorted(list(temp_all_steps))
                if not st.session_state.all_simulated_steps:
                    st.session_state.all_simulated_steps = [0]

                if st.session_state.all_simulated_steps:
                    st.session_state.current_simulated_global_step = (
                        st.session_state.all_simulated_steps[0]
                    )
                else:
                    st.session_state.current_simulated_global_step = 0
                st.rerun()

            except Exception as e:
                st.error(f"生成示例数据失败: {e}")
                st.exception(e)
                st.session_state.generated_data_sources_map = None
        else:
            st.warning("请先选择一个组件类型。")

    st.session_state.component_specific_config_str = st.text_area(
        "组件特定配置 (JSON)",
        value=st.session_state.component_specific_config_str,
        height=100,
    )

    if st.button("🚀 实例化组件", type="primary"):
        if (
            SelectedVisualizerClass
            and st.session_state.component_instance_id
            and st.session_state.generated_data_sources_map
        ):
            try:
                specific_config = json.loads(
                    st.session_state.component_specific_config_str
                )
                st.session_state.active_visualizer_instance = SelectedVisualizerClass(
                    component_instance_id=st.session_state.component_instance_id,
                    trial_root_path=Path(st.session_state.trial_root_path_str),
                    data_sources_map=st.session_state.generated_data_sources_map,
                    component_specific_config=specific_config,
                )
                st.success(
                    f"组件 '{st.session_state.component_instance_id}' 已实例化。"
                )
                active_viz_instance_for_load = (
                    st.session_state.active_visualizer_instance
                )
                active_viz_instance_for_load.load_data()

                if (
                    hasattr(active_viz_instance_for_load, "_all_available_steps")
                    and active_viz_instance_for_load._all_available_steps
                ):
                    st.session_state.all_simulated_steps = (
                        active_viz_instance_for_load._all_available_steps
                    )
                    if (
                        st.session_state.all_simulated_steps
                        and st.session_state.current_simulated_global_step
                        not in st.session_state.all_simulated_steps
                    ):
                        st.session_state.current_simulated_global_step = (
                            st.session_state.all_simulated_steps[0]
                        )
                    elif not st.session_state.all_simulated_steps:
                        st.session_state.all_simulated_steps = [0]
                        st.session_state.current_simulated_global_step = 0

                active_viz_instance_for_load.configure_global_step_interaction(
                    current_step=st.session_state.current_simulated_global_step,
                    all_available_steps=st.session_state.all_simulated_steps,
                    on_step_change_request_callback=lambda step: st.session_state.update(
                        {"current_simulated_global_step": step}
                    ),
                )
                st.rerun()

            except Exception as e:
                st.error(f"实例化组件失败: {e}")
                st.exception(e)
                st.session_state.active_visualizer_instance = None
        else:
            st.warning("请先选择组件类型，输入实例ID，并生成示例数据。")

st.header("组件预览与交互")
active_viz_instance: Optional[VisualizationComponent] = (
    st.session_state.active_visualizer_instance
)

if active_viz_instance:
    # 确保 SelectedVisualizerClass 在rerun后仍然有效
    # Ensure SelectedVisualizerClass is still valid after rerun
    # (通常通过再次从 st.session_state.selected_visualizer_type_name 获取)
    # (Usually by getting it again from st.session_state.selected_visualizer_type_name)
    RehydratedSelectedVisualizerClass = get_visualizer_class(
        st.session_state.selected_visualizer_type_name
    )

    st.markdown(
        f"**当前活动组件:** `{active_viz_instance.component_instance_id}` "
        f"(类型: `{RehydratedSelectedVisualizerClass.get_display_name() if RehydratedSelectedVisualizerClass else 'N/A'}`)"
    )
    st.markdown(f"**Trial根路径:** `{active_viz_instance.trial_root_path}`")

    st.markdown("---")
    st.subheader("全局步骤模拟")
    col_step1, col_step2 = st.columns([3, 1])

    current_step_for_ui = st.session_state.current_simulated_global_step
    all_steps_for_ui = st.session_state.all_simulated_steps

    if all_steps_for_ui and current_step_for_ui not in all_steps_for_ui:
        current_step_for_ui = (
            min(all_steps_for_ui, key=lambda x: abs(x - current_step_for_ui))
            if all_steps_for_ui
            else 0
        )

    with col_step1:
        if all_steps_for_ui:
            if len(all_steps_for_ui) == 1:
                st.markdown(
                    f"当前模拟全局步骤: **{all_steps_for_ui[0]}** (只有一步可用)"
                )
                if (
                    st.session_state.current_simulated_global_step
                    != all_steps_for_ui[0]
                ):
                    st.session_state.current_simulated_global_step = all_steps_for_ui[0]
                new_sim_step = all_steps_for_ui[0]
            else:
                new_sim_step = st.select_slider(
                    "当前模拟全局步骤",
                    options=all_steps_for_ui,
                    value=current_step_for_ui,
                    key=f"sim_step_slider_{st.session_state.selected_visualizer_type_name}",
                )
        else:
            new_sim_step = st.number_input(
                "当前模拟全局步骤 (无可用步骤)",
                value=current_step_for_ui,
                key=f"sim_step_input_{st.session_state.selected_visualizer_type_name}",
            )

    if new_sim_step != st.session_state.current_simulated_global_step:
        st.session_state.current_simulated_global_step = new_sim_step

    active_viz_instance.configure_global_step_interaction(
        current_step=st.session_state.current_simulated_global_step,
        all_available_steps=all_steps_for_ui,
        on_step_change_request_callback=lambda step: st.session_state.update(
            {"current_simulated_global_step": step}
        ),
    )

    with col_step2:
        if st.button("🔄 重新加载数据"):
            try:
                active_viz_instance.load_data()
                st.toast("组件数据已重新加载。")
                if (
                    hasattr(active_viz_instance, "_all_available_steps")
                    and active_viz_instance._all_available_steps
                ):
                    st.session_state.all_simulated_steps = (
                        active_viz_instance._all_available_steps
                    )
                    if (
                        st.session_state.all_simulated_steps
                        and st.session_state.current_simulated_global_step
                        not in st.session_state.all_simulated_steps
                    ):
                        st.session_state.current_simulated_global_step = (
                            st.session_state.all_simulated_steps[0]
                        )
                    elif not st.session_state.all_simulated_steps:
                        st.session_state.all_simulated_steps = [0]
                        st.session_state.current_simulated_global_step = 0

                active_viz_instance.configure_global_step_interaction(
                    current_step=st.session_state.current_simulated_global_step,
                    all_available_steps=st.session_state.all_simulated_steps,
                    on_step_change_request_callback=lambda step: st.session_state.update(
                        {"current_simulated_global_step": step}
                    ),
                )
                st.rerun()
            except Exception as e:
                st.error(f"重新加载数据失败: {e}")

    st.markdown("---")
    st.subheader("渲染输出")
    try:
        with st.container(border=True):
            active_viz_instance.render()
    except Exception as e:
        st.error(f"渲染组件 '{active_viz_instance.component_instance_id}' 时出错: {e}")
        st.exception(e)
else:
    st.info(
        "请在侧边栏中选择一个组件类型，生成示例数据，然后点击“实例化组件”以开始调试。"
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"IDE 会话临时路径: {st.session_state.trial_root_path_str}")
if st.sidebar.button("清理当前会话的临时Trial目录"):
    cleanup_temp_dir(st.session_state.trial_root_path_str)
    st.session_state.trial_root_path_str = tempfile.mkdtemp(
        prefix="flowillower_ide_trial_"
    )
    st.session_state.active_visualizer_instance = None
    st.session_state.generated_data_sources_map = None
    st.session_state.all_simulated_steps = []
    st.session_state.current_simulated_global_step = 0
    st.rerun()
from flowillower.help import version

st.sidebar.caption(f"Flowillower 组件IDE - 版本 {version}")

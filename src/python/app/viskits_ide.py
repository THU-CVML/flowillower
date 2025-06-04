# flowillower/app/component_ide.py
import streamlit as st
from pathlib import Path
import tempfile
import json
import shutil
from typing import Dict, Any, Optional, Type, List
import tomli 
# 必须有这几行，阻止 PyTorch 报错
# https://github.com/VikParuchuri/marker/issues/442
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# --- 应用标题和配置 ---
st.set_page_config(layout="wide", page_title="视件视界 - Flowillower 视件IDE") 
st.title("🔬 视件视界 - Flowillower 可视化组件集成开发环节") 
st.header("🍺 VisKits VisScope - Visualization Component IDE for Flowillower") 
st.markdown("在此环境中独立测试、调试和预览您的可视化视件。") 

# --- 模块导入 (使用基于包的绝对导入) ---
try:
    from flowillower.viskits.base_viskit import ( 
        VISKIT_REGISTRY, 
        get_viskit_class, 
        VisKit, 
        PYDANTIC_AVAILABLE 
    )
    # # 显式导入所有视件模块以确保它们被注册
    # # Explicitly import all Viskit modules to ensure they are registered
    import flowillower.viskits.scalar_dashboard_viskit 
    # import flowillower.viskits.treescope_viskit # Assuming you will rename/create this
    import flowillower.viskits.pygwalker_viskit # Assuming you will rename/create this
    # import flowillower.viskits.torchlens_viskit # Assuming you will rename/create this

except ImportError as e:
    st.error(
        "错误：无法导入视件模块。请确保您已正确设置项目结构，"
        "并且 'flowillower' 包位于您的PYTHONPATH中，或者您是从项目根目录运行此应用。"
        f"\n详细信息: {e}"
    )
    st.stop()


# --- 会话状态初始化 ---
if "selected_viskit_type_name" not in st.session_state: 
    st.session_state.selected_viskit_type_name = None
if "viskit_instance_id" not in st.session_state: 
    st.session_state.viskit_instance_id = "ide_test_viskit_001" 
if "trial_root_path_str" not in st.session_state:
    st.session_state.trial_root_path_str = tempfile.mkdtemp(prefix="flowillower_ide_trial_")
if "viskit_specific_ui_config_str" not in st.session_state: 
    st.session_state.viskit_specific_ui_config_str = "{}" 
if "active_viskit_instance" not in st.session_state: 
    st.session_state.active_viskit_instance = None
if "generated_data_sources_map_for_current_config" not in st.session_state: 
    st.session_state.generated_data_sources_map_for_current_config = None
if "current_simulated_global_step" not in st.session_state:
    st.session_state.current_simulated_global_step = 0 
if "all_simulated_steps" not in st.session_state: 
    st.session_state.all_simulated_steps = []
if "last_reported_assets" not in st.session_state: 
    st.session_state.last_reported_assets = None
if "last_report_params" not in st.session_state: # 新增：用于显示上一次render_report_ui返回的参数
    st.session_state.last_report_params = None


def cleanup_temp_dir(path_str):
    try:
        if path_str and Path(path_str).exists() and "flowillower_ide_trial_" in path_str: 
            shutil.rmtree(path_str)
            st.toast(f"临时目录已清理: {path_str}")
    except Exception as e:
        st.warning(f"清理临时目录失败 {path_str}: {e}")

def sync_steps_from_viskit(viskit_instance: Optional[VisKit]): 
    if viskit_instance and hasattr(viskit_instance, '_all_available_steps'):
        st.session_state.all_simulated_steps = viskit_instance._all_available_steps if viskit_instance._all_available_steps is not None else []
    else: 
        st.session_state.all_simulated_steps = []

    if not st.session_state.all_simulated_steps: 
        st.session_state.all_simulated_steps = [0] 
        st.session_state.current_simulated_global_step = 0
    elif st.session_state.current_simulated_global_step not in st.session_state.all_simulated_steps:
        st.session_state.current_simulated_global_step = st.session_state.all_simulated_steps[0]
    
    if viskit_instance:
        viskit_instance.configure_global_step_interaction(
            current_step=st.session_state.current_simulated_global_step,
            all_available_steps=st.session_state.all_simulated_steps,
            on_step_change_request_callback=lambda step: st.session_state.update({"current_simulated_global_step": step})
        )


with st.sidebar:
    st.header("视件选择与配置") 

    registered_type_names = list(VISKIT_REGISTRY.keys()) 
    if not registered_type_names:
        st.error("错误：没有已注册的视件类型。") 
        st.stop()

    default_selection_index = 0
    if st.session_state.selected_viskit_type_name in registered_type_names: 
        default_selection_index = registered_type_names.index(st.session_state.selected_viskit_type_name)
    elif registered_type_names: 
        st.session_state.selected_viskit_type_name = registered_type_names[0]
    
    current_selected_type = st.selectbox(
        "选择视件类型", 
        options=registered_type_names,
        index=default_selection_index,
        help="选择您想要测试的视件。" 
    )
    if current_selected_type != st.session_state.selected_viskit_type_name:
        st.session_state.selected_viskit_type_name = current_selected_type
        st.session_state.active_viskit_instance = None 
        st.session_state.generated_data_sources_map_for_current_config = None 
        st.session_state.all_simulated_steps = []
        st.session_state.current_simulated_global_step = 0
        st.session_state.viskit_specific_ui_config_str = "{}"  
        st.session_state.last_reported_assets = None 
        st.session_state.last_report_params = None 
        st.rerun()


    SelectedVisKitClass: Optional[Type[VisKit]] = get_viskit_class(st.session_state.selected_viskit_type_name) 

    if SelectedVisKitClass:
        st.caption(f"显示名称: `{SelectedVisKitClass.get_display_name()}`")
    else:
        st.error(f"无法加载视件类 '{st.session_state.selected_viskit_type_name}'。") 
        st.stop()

    st.session_state.viskit_instance_id = st.text_input( 
        "视件实例ID", value=st.session_state.viskit_instance_id 
    )
    st.markdown(f"**临时Trial根路径:** `{st.session_state.trial_root_path_str}`")
    
    example_data_target_dir = Path(st.session_state.trial_root_path_str) / "example_assets_for_ide"
    
    example_gen_config_str = "{}"
    if st.session_state.selected_viskit_type_name == "treescope_model_viewer": 
        example_gen_config_str = st.text_input(
            "示例数据生成配置 (JSON) - Treescope",
            value='{"group_id": "my_ide_treescope_group"}', 
            help='例如: {"group_id": "custom_group_name"} (特定于Treescope查看器)'
        )
    elif st.session_state.selected_viskit_type_name == "torchlens_flow_viewer": 
        example_gen_config_str = st.text_input(
            "示例数据生成配置 (JSON) - Torchlens",
            value='{"group_id": "my_ide_torchlens_group"}',
            help='例如: {"group_id": "custom_cnn_flow"} (特定于Torchlens查看器)'
        )
    
    if st.button("🚀 实例化/重新实例化视件", type="primary"): 
        if SelectedVisKitClass and st.session_state.viskit_instance_id: 
            try:
                specific_config_dict_from_json = json.loads(st.session_state.viskit_specific_ui_config_str) 
                
                data_map_for_init = st.session_state.generated_data_sources_map_for_current_config or {}

                st.session_state.active_viskit_instance = SelectedVisKitClass( 
                    instance_id=st.session_state.viskit_instance_id, 
                    trial_root_path=Path(st.session_state.trial_root_path_str),
                    data_sources_map=data_map_for_init, 
                    specific_ui_config_dict=specific_config_dict_from_json 
                )
                st.success(f"视件 '{st.session_state.viskit_instance_id}' 已实例化/重新实例化。") 
                
                active_viskit_for_load = st.session_state.active_viskit_instance 
                active_viskit_for_load.load_data() 
                sync_steps_from_viskit(active_viskit_for_load) 
                
                st.session_state.last_reported_assets = None 
                st.session_state.last_report_params = None 
                st.rerun() 

            except json.JSONDecodeError:
                st.error("视件特定UI配置不是有效的JSON。") 
                st.session_state.active_viskit_instance = None 
            except Exception as e:
                st.error(f"实例化视件失败: {e}") 
                st.exception(e)
                st.session_state.active_viskit_instance = None 
        else:
            st.warning("请选择视件类型并输入实例ID。") 


    if st.button("📊 生成示例数据 (填充/覆盖)"): 
        if SelectedVisKitClass:
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

                newly_generated_map = SelectedVisKitClass.generate_example_data(
                    ide_instance_id=st.session_state.viskit_instance_id,  
                    ide_trial_root_path=Path(st.session_state.trial_root_path_str), 
                    data_sources_config=parsed_example_gen_config
                )
                st.session_state.generated_data_sources_map_for_current_config = newly_generated_map 
                st.success(f"'{SelectedVisKitClass.get_display_name()}' 的示例数据已生成/覆盖。")
                st.session_state.last_reported_assets = None 
                st.session_state.last_report_params = None 
                
                if st.session_state.active_viskit_instance: 
                    st.session_state.active_viskit_instance.data_sources_map = newly_generated_map 
                    st.session_state.active_viskit_instance.load_data()
                    sync_steps_from_viskit(st.session_state.active_viskit_instance) 
                else: 
                    temp_all_steps = set()
                    if newly_generated_map:
                        for ds_name, ds_info in newly_generated_map.items():
                            if isinstance(ds_info, dict) and "items" in ds_info and isinstance(ds_info["items"], list): 
                                for item in ds_info["items"]:
                                    if isinstance(item, dict) and "related_step" in item:
                                        try: temp_all_steps.add(int(item["related_step"]))
                                        except: pass
                            elif isinstance(ds_info, dict) and "path" in ds_info: 
                                 try:
                                    full_path = Path(st.session_state.trial_root_path_str) / ds_info["path"]
                                    if full_path.exists() and full_path.suffix == ".toml":
                                        with open(full_path, "rb") as f:
                                            d = tomli.load(f)
                                        if "metrics" in d and isinstance(d["metrics"], list):
                                            for point in d["metrics"]:
                                                if "global_step" in point:
                                                    try: temp_all_steps.add(int(point["global_step"]))
                                                    except: pass
                                 except Exception: pass 
                    st.session_state.all_simulated_steps = sorted(list(temp_all_steps))
                    if not st.session_state.all_simulated_steps: st.session_state.all_simulated_steps = [0]
                    st.session_state.current_simulated_global_step = st.session_state.all_simulated_steps[0]
                st.rerun()

            except Exception as e:
                st.error(f"生成示例数据失败: {e}")
                st.exception(e)
                st.session_state.generated_data_sources_map_for_current_config = None
        else:
            st.warning("请先选择一个视件类型。") 
    
    default_specific_config_dict = {}
    if SelectedVisKitClass: 
        default_specific_config_dict = SelectedVisKitClass.get_default_config_as_dict()
    
    if "last_selected_viskit_for_config" not in st.session_state or \
       st.session_state.last_selected_viskit_for_config != st.session_state.selected_viskit_type_name: 
        st.session_state.viskit_specific_ui_config_str = json.dumps(default_specific_config_dict, indent=2) 
        st.session_state.last_selected_viskit_for_config = st.session_state.selected_viskit_type_name 

    st.session_state.viskit_specific_ui_config_str = st.text_area( 
        "视件特定UI配置 (JSON)",  
        value=st.session_state.viskit_specific_ui_config_str, 
        height=150, 
        key=f"specific_config_text_area_{st.session_state.selected_viskit_type_name}" 
    )


# --- 主区域：视件预览、上报UI和交互 ---
st.header("视件预览与交互") 
active_viskit_instance: Optional[VisKit] = st.session_state.active_viskit_instance 

if active_viskit_instance: 
    RehydratedSelectedVisKitClass = get_viskit_class(st.session_state.selected_viskit_type_name) 

    st.markdown(f"**当前活动视件:** `{active_viskit_instance.instance_id}` " 
                f"(类型: `{RehydratedSelectedVisKitClass.get_display_name() if RehydratedSelectedVisKitClass else 'N/A'}`)")
    st.markdown(f"**Trial根路径:** `{active_viskit_instance.trial_root_path}`")
    st.markdown(f"**视件私有存储:** `{active_viskit_instance.private_storage_path}`") 
    
    tab_preview, tab_report_data = st.tabs(["视件预览 (Preview)", "上报数据测试 (Report Data Test)"]) 

    with tab_preview:
        sync_steps_from_viskit(active_viskit_instance) 

        st.subheader("全局步骤模拟")
        col_step1, col_step2 = st.columns([3,1])
        
        current_step_for_ui = st.session_state.current_simulated_global_step
        all_steps_for_ui = st.session_state.all_simulated_steps
        
        with col_step1:
            if all_steps_for_ui: 
                if len(all_steps_for_ui) == 1:
                    st.markdown(f"当前模拟全局步骤: **{all_steps_for_ui[0]}** (只有一步可用)")
                    new_sim_step = all_steps_for_ui[0]
                    if st.session_state.current_simulated_global_step != new_sim_step: 
                        st.session_state.current_simulated_global_step = new_sim_step
                else: 
                    value_for_slider = current_step_for_ui if current_step_for_ui in all_steps_for_ui else all_steps_for_ui[0]
                    new_sim_step = st.select_slider(
                        "当前模拟全局步骤", options=all_steps_for_ui, value=value_for_slider, 
                        key=f"sim_step_slider_{st.session_state.selected_viskit_type_name}" 
                    )
            else: 
                st.markdown("当前模拟全局步骤: 无可用步骤。")
                new_sim_step = 0 
        
        if new_sim_step != st.session_state.current_simulated_global_step:
            st.session_state.current_simulated_global_step = new_sim_step
            sync_steps_from_viskit(active_viskit_instance) 
            st.rerun() 
        
        with col_step2:
            if st.button("🔄 重新加载视件数据"): 
                try:
                    active_viskit_instance.load_data() 
                    st.toast("视件的可视化数据已重新加载。") 
                    sync_steps_from_viskit(active_viskit_instance) 
                    st.rerun() 
                except Exception as e:
                    st.error(f"重新加载视件数据失败: {e}") 

        st.markdown("---")
        st.subheader("渲染输出")
        try:
            with st.container(border=True):
                active_viskit_instance.render() 
        except Exception as e:
            st.error(f"渲染视件 '{active_viskit_instance.instance_id}' 时出错: {e}") 
            st.exception(e)

    with tab_report_data:
        st.subheader(f"测试 `{SelectedVisKitClass.get_display_name() if SelectedVisKitClass else ''}` 的 `report_data` 方法")
        report_ui_container = st.container(border=True)
        
        report_params = active_viskit_instance.render_report_ui(report_ui_container) 
        has_new_report_request = report_params is not None and isinstance(report_params, dict)

        if has_new_report_request:
            st.session_state.last_report_params = report_params 
            try:
                returned_asset_descriptions = active_viskit_instance.report_data(**report_params) 
                st.session_state.last_reported_assets = returned_asset_descriptions
                st.success("`report_data` 调用成功！")
                
                active_viskit_instance.load_data() 
                sync_steps_from_viskit(active_viskit_instance) 
                
                st.rerun() 

            except Exception as e:
                st.error(f"`report_data` 调用失败: {e}")
                st.exception(e)
                st.session_state.last_reported_assets = {"error": str(e)}
        
        if st.session_state.last_report_params and isinstance(st.session_state.last_report_params, dict):
            st.markdown("---")
            st.write("上次提交的 `report_data` 调用参数:")
            st.json(st.session_state.last_report_params)
        
        if st.session_state.last_reported_assets is not None:
            st.markdown("---")
            st.write("上次 `report_data` 的返回结果:")
            st.json(st.session_state.last_reported_assets)

else:
    st.info("请在侧边栏中选择一个视件类型，然后点击“实例化/重新实例化视件”以开始。您可以稍后生成示例数据或通过“上报数据测试”选项卡添加数据。") 

st.sidebar.markdown("---")
st.sidebar.caption(f"IDE 会话临时路径: {st.session_state.trial_root_path_str}")
if st.sidebar.button("清理当前会话的临时Trial目录"):
    cleanup_temp_dir(st.session_state.trial_root_path_str)
    st.session_state.trial_root_path_str = tempfile.mkdtemp(prefix="flowillower_ide_trial_")
    st.session_state.active_viskit_instance = None 
    st.session_state.generated_data_sources_map_for_current_config = None 
    st.session_state.all_simulated_steps = []
    st.session_state.current_simulated_global_step = 0
    st.session_state.last_reported_assets = None
    st.session_state.last_report_params = None # 清理这个状态
    st.rerun()

# 导入版本信息 (Import version information)
from flowillower.help import version # 假设 help.py 在 flowillower 包的根目录
st.sidebar.caption(f"Flowillower 视件视界 - 版本 {version}")

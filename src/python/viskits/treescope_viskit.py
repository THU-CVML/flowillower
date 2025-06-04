# flowillower/viskits/treescope_viskit.py
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import shutil 
import html 
import random # Added import

# PyTorch, treescope imports
import torch
import torch.nn as nn
import os # Added import for path fix
# Fix for PyTorch C++ extensions in Streamlit (https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4)
if hasattr(torch, 'classes') and hasattr(torch.classes, '__file__') and torch.classes.__file__ is not None: # Check if torch.classes.__file__ is not None
    torch.classes.__path__ = [os.path.join(Path(torch.__file__).parent.as_posix(), Path(torch.classes.__file__).name)] # Use Path for robust path joining
else:
    # Fallback or warning if torch.classes.__file__ is None or torch.classes is not as expected
    # This might happen in some environments or with certain PyTorch versions.
    # For now, we'll proceed, but this could be a source of issues if the fix is critical.
    print("Warning: PyTorch C++ extension path fix could not be applied fully. `torch.classes.__file__` might be None.")

import treescope # Assuming it's installed

from flowillower.viskits.base_viskit import ( 
    VisKit, 
    register_viskit, 
    PYDANTIC_AVAILABLE,
    STREAMLIT_PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field
    if STREAMLIT_PYDANTIC_AVAILABLE:
        import streamlit_pydantic as sp
else: 
    class BaseModel: pass 
    from dataclasses import dataclass


# Define a simple PyTorch model for example generation
class ExamplePyTorchModelForTreescope(nn.Module): 
    def __init__(self, step_influence=1.0, num_layers=2, hidden_size=10):
        super().__init__()
        self.layers = nn.ModuleList()
        # Ensure at least one linear layer before potential ReLU
        if hidden_size > 0 : # Add check for hidden_size
             self.layers.append(nn.Linear(hidden_size, hidden_size))
             for _ in range(num_layers -1):
                 self.layers.append(nn.Linear(hidden_size, hidden_size))
                 self.layers.append(nn.ReLU())
             self.output_layer = nn.Linear(hidden_size, 5)
        else: # Fallback for hidden_size = 0 or less (though unlikely for a real model)
            self.output_layer = nn.Linear(1,1) # Minimal layer
        
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(step_influence * random.uniform(0.8, 1.2)) 
                if random.random() < 0.1: 
                    param.add_(torch.randn_like(param) * 0.1 * step_influence)

    def forward(self, x):
        if not hasattr(self, 'layers') or not self.layers: # Check if layers list is empty or not present
             if hasattr(self, 'output_layer'):
                 return self.output_layer(x)
             else: # Should not happen if __init__ ran correctly
                 return x


        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Define the UI configuration Pydantic Model for this VisKit
if PYDANTIC_AVAILABLE:
    class TreescopeUIConfig(BaseModel):
        html_height: int = Field(default=700, ge=300, le=3000, description="内嵌Treescope HTML的高度（像素）")
else: 
    @dataclass
    class TreescopeUIConfig: # type: ignore
        html_height: int = 700


TREESCOPE_VIEW_COLLECTION_DATA_TYPE = "treescope_view_collection_v1" 
INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE = "single_treescope_html_v1"     


@register_viskit(name="treescope_model_viewer") 
class TreescopeModelViewVisKit(VisKit): 
    ui_config: TreescopeUIConfig
    _step_to_html_map: Dict[int, Path] 
    _display_name_for_panel: str
    _data_loaded_once_treescope: bool # Declare the flag

    LOGICAL_DATA_SOURCE_NAME = "treescope_html_collection"


    def __init__(self,
                 instance_id: str, 
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]], 
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(instance_id, trial_root_path, data_sources_map, specific_ui_config_dict)
        self._step_to_html_map = {}
        self._data_loaded_once_treescope = False # Initialize the flag
        
        primary_source_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 
        self._display_name_for_panel = "Treescope View" 
        if primary_source_info and "display_name" in primary_source_info:
            self._display_name_for_panel = primary_source_info["display_name"]
        # self.ui_config (TreescopeUIConfig) is initialized by the base class

    @classmethod
    def get_config_model(cls) -> Optional[Type[TreescopeUIConfig]]:
        return TreescopeUIConfig if PYDANTIC_AVAILABLE else None

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return cls.TREESCOPE_VIEW_COLLECTION_DATA_TYPE in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Treescope 模型查看器 (Treescope Model Viewer)"

    @classmethod
    def _generate_example_payloads_and_steps(cls,
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        payloads = []
        example_steps = [0, 5, 10, 15]
        base_decay_factor = 0.95 
        group_id = (data_sources_config or {}).get("group_id", "example_pytorch_model_treescope")
        asset_name_prefix = (data_sources_config or {}).get("asset_name_prefix", "model_view")
        hidden_size_example = (data_sources_config or {}).get("hidden_size", 10) # Allow config for example model

        for i, step_val in enumerate(example_steps):
            step_influence_factor = base_decay_factor ** i 
            model_instance = ExamplePyTorchModelForTreescope(
                step_influence=step_influence_factor,
                hidden_size=hidden_size_example # Use configured or default hidden_size
            )
            
            payloads.append({
                "data_payload": model_instance, 
                "step": step_val,
                "group_id": group_id,
                "asset_name": f"{asset_name_prefix}_step_{step_val}"
            })
        return payloads

    def report_data(self,
                    data_payload: Any, 
                    step: int,
                    group_id: Optional[str] = None, 
                    asset_name: Optional[str] = None, 
                    **kwargs) -> List[Dict[str, Any]]:
        if group_id is None:
            group_id = self.instance_id 
        
        safe_asset_name = asset_name if asset_name and asset_name.strip() else f"treescope_view"
        safe_asset_name_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in safe_asset_name)
        group_id_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in group_id)
        
        output_dir_relative_to_private_storage = Path(group_id_for_path) / f"step_{step}"
        output_dir_full = self.private_storage_path / output_dir_relative_to_private_storage
        output_dir_full.mkdir(parents=True, exist_ok=True)

        html_file_name = f"{safe_asset_name_for_path}.html"
        html_file_full_path = output_dir_full / html_file_name

        try:
            # Use more comprehensive rendering options from the old example
            with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
                with treescope.using_expansion_strategy(max_height=9999): # Increased max_height
                    html_str = treescope.render_to_html(data_payload, compressed=True, roundtrip_mode=False)
            
            with open(html_file_full_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            
            path_in_manifest = (Path("viskits_data") / self.instance_id / output_dir_relative_to_private_storage / html_file_name).as_posix()

            asset_description = {
                "asset_id": f"{self.instance_id}_{group_id}_step_{step}_{safe_asset_name}", 
                "display_name": f"{self.get_display_name()}: {group_id} - {safe_asset_name} (Step {step})",
                "data_type_original": INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE, 
                "path": path_in_manifest,
                "related_step": step,
                "group_id": group_id, 
                "asset_name_source": asset_name 
            }
            return [asset_description]

        except Exception as e:
            print(f"错误: 为步骤 {step} 生成Treescope HTML失败 (视件: {self.instance_id}): {e}")
            return []


    def load_data(self) -> None:
        self._step_to_html_map = {}
        source_collection_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 

        if not source_collection_info or source_collection_info.get("data_type") != TREESCOPE_VIEW_COLLECTION_DATA_TYPE:
            self._all_available_steps = []
            return

        items = source_collection_info.get("items", [])
        if not isinstance(items, list):
            st.warning(f"视件 {self.instance_id}: 数据源中的 'items' 不是列表。")
            self._all_available_steps = []
            return
            
        temp_all_steps = set()
        for item_info in items:
            if isinstance(item_info, dict) and \
               item_info.get("data_type_original") == INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE and \
               "path" in item_info and "related_step" in item_info:
                try:
                    step = int(item_info["related_step"])
                    relative_path_str = item_info["path"] 
                    full_path = (self.trial_root_path / relative_path_str).resolve()
                    
                    if full_path.exists() and full_path.is_file():
                        self._step_to_html_map[step] = full_path 
                        temp_all_steps.add(step)
                    else:
                        st.warning(f"视件 {self.instance_id}: HTML文件未找到或不是文件: {full_path} (步骤 {step})")
                except ValueError:
                    st.warning(f"视件 {self.instance_id}: item中的步骤值无效: {item_info.get('related_step')}")
                except Exception as e:
                    st.error(f"视件 {self.instance_id}: 处理item {item_info} 出错: {e}")
            else:
                st.warning(f"视件 {self.instance_id}: 数据源中的item格式无效或类型不匹配: {item_info.get('data_type_original')}")
        
        self._all_available_steps = sorted(list(temp_all_steps))


    def render_config_ui(self, config_container) -> bool:
        if not (PYDANTIC_AVAILABLE): 
            config_container.caption("Pydantic 未安装，无法渲染配置表单。")
            return False
        
        ConfigModel = self.get_config_model()
        if not ConfigModel: 
            config_container.caption("此视件没有可配置的UI模型。")
            return False

        changed = False
        config_container.markdown("##### 视件显示设置 (VisKit Display Settings)")

        # For Treescope, only html_height is currently configurable via Pydantic model
        # We will create the UI manually for this one field.
        with config_container.form(key=f"{self.instance_id}_treescope_cfg_form_manual"):
            current_height = self.ui_config.html_height
            new_html_height = st.number_input(
                "HTML 显示高度 (HTML Display Height)",
                min_value=200, max_value=3000, step=100,
                value=current_height,
                key=f"{self.instance_id}_cfg_html_height_manual" # Ensure unique key
            )
            submitted = st.form_submit_button("应用高度设置")
            if submitted:
                if new_html_height != current_height:
                    try:
                        self.ui_config.html_height = new_html_height 
                        self.save_ui_config()
                        changed = True
                    # except ValidationError as ve: # Not strictly needed if only one field and it's validated by number_input
                    #     st.error(f"配置验证失败: {ve}")
                    except Exception as e_cfg:
                        st.error(f"应用配置时出错: {e_cfg}")
        return changed


    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        report_container.markdown(f"#### 上报模型视图数据到 `{self.instance_id}` (Treescope)")
        
        with report_container.form(key=f"{self.instance_id}_treescope_report_form"):
            step = st.number_input("全局步骤 (Global Step)", min_value=0, value=self._current_global_step or 0, step=1)
            group_id = st.text_input("模型/参数组ID (Model/Parameter Group ID)", value="transformer_block_1_attention")
            asset_name = st.text_input("视图名称 (View Name)", value=f"attention_map_step_{step}")
            
            st.markdown("**模拟模型参数 (Simulate Model Parameters):**")
            num_layers = st.slider("模型层数 (Number of Layers in Example Model)", 1, 5, 2, key=f"{self.instance_id}_report_num_layers")
            hidden_size = st.select_slider("隐藏层大小 (Hidden Size)", [8, 10, 16, 20], value=10, key=f"{self.instance_id}_report_hidden_size")
            step_influence_factor = st.number_input("步骤影响因子 (Step Influence Factor for Weights)", value=0.9, min_value=0.1, max_value=2.0, step=0.1, key=f"{self.instance_id}_report_step_influence")

            submit_button = st.form_submit_button("上报模型视图 (Report Model View)")

            if submit_button:
                try:
                    example_model_payload = ExamplePyTorchModelForTreescope(
                        step_influence=step_influence_factor,
                        num_layers=num_layers,
                        hidden_size=hidden_size
                    )
                    return {
                        "data_payload": example_model_payload,
                        "step": int(step),
                        "group_id": group_id if group_id.strip() else None,
                        "asset_name": asset_name if asset_name.strip() else None
                    }
                except Exception as e:
                    st.error(f"创建示例模型失败: {e}")
                    return None
        return None


    def render(self) -> None:
        st.subheader(self._display_name_for_panel)

        with st.expander("显示设置 (Display Settings)", expanded=False):
            if self.render_config_ui(st.container()):
                st.rerun() 

        if not self._step_to_html_map:
            if not self._data_loaded_once_treescope: # Use the initialized flag
                self.load_data()
                self._data_loaded_once_treescope = True 
            if not self._step_to_html_map:
                st.info(f"视件 {self.instance_id}: 没有可显示的Treescope视图数据。")
                return

        if self._all_available_steps is None or not self._all_available_steps:
            if self._step_to_html_map:
                self._all_available_steps = sorted(list(self._step_to_html_map.keys()))
            else: 
                st.warning("没有可用的步骤。")
                return

        step_to_display = self._get_closest_available_step(self._current_global_step)

        if step_to_display is None or step_to_display not in self._step_to_html_map:
            st.warning(f"视件 {self.instance_id}: 在步骤 {self._current_global_step} (或附近) 未找到Treescope视图。")
            available_s = ", ".join(map(str, sorted(self._step_to_html_map.keys())))
            st.caption(f"可用步骤: {available_s if available_s else 'None'}")
            return

        html_file_path = self._step_to_html_map[step_to_display]

        if html_file_path.exists():
            try:
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                if self._current_global_step is not None and step_to_display != self._current_global_step:
                    st.caption(f"显示最接近步骤 {self._current_global_step} 的视图 (步骤 {step_to_display})。")
                else:
                    st.caption(f"显示步骤 {step_to_display} 的视图。")

                st.components.v1.html(html_content, height=self.ui_config.html_height, scrolling=True)
            except Exception as e:
                st.error(f"视件 {self.instance_id}: 渲染HTML文件 '{html_file_path}' 失败: {e}")
        else:
            st.error(f"视件 {self.instance_id}: HTML文件未找到: {html_file_path}")


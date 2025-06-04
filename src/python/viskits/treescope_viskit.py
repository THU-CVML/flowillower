# flowillower/viskits/treescope_viskit.py
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import shutil 
import html 
import random 
import os
import streamlit.components.v1 as components

# PyTorch, treescope imports
import torch
import torch.nn as nn
# 路径修复 Path fix
if hasattr(torch, 'classes') and hasattr(torch.classes, '__file__') and torch.classes.__file__ is not None: 
    torch.classes.__path__ = [os.path.join(Path(torch.__file__).parent.as_posix(), Path(torch.classes.__file__).name)]
else:
    print("Warning: PyTorch C++ extension path fix could not be applied fully or is not needed.")

import base64
def better_st_html(html_str, **kwargs):
    # https://discuss.streamlit.io/t/html-file-couldnt-be-rendered-with-components-html/10356
    html_str = base64.b64encode(html_str.encode("utf-8")).decode()
    return components.iframe(
        src=f"data:text/html;base64,{html_str}",
        **kwargs)

import treescope 

from flowillower.viskits.base_viskit import ( 
    VisKit, 
    register_viskit, 
    PYDANTIC_AVAILABLE,
    STREAMLIT_PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field, ValidationError
    from typing_extensions import Literal
    if STREAMLIT_PYDANTIC_AVAILABLE:
        import streamlit_pydantic as sp
else: 
    class BaseModel: pass 
    Literal = str # Fallback
    from dataclasses import dataclass 


# 更复杂的示例模型 (A more complex example model)
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x): # x: (batch, seq_len, embed_dim)
        # For simplicity, assume q, k, v are the same as x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention mechanism (no scaling or softmax for example brevity)
        attn_output = torch.bmm(q, k.transpose(1,2)) 
        attn_output = torch.bmm(attn_output, v) 
        return self.out_proj(attn_output)

class SimpleMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim_multiplier=2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * hidden_dim_multiplier)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * hidden_dim_multiplier, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ExampleComplexModel(nn.Module): 
    def __init__(self, step_influence=1.0, embed_dim=16, num_blocks=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.initial_projection = nn.Linear(embed_dim, embed_dim) # Example input layer
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attention': SimpleAttention(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': SimpleMLP(embed_dim)
            })
            self.blocks.append(block)
        
        self.output_head = nn.Linear(embed_dim, 5)
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.mul_(step_influence * random.uniform(0.9, 1.1)) 
                if "bias" not in name and random.random() < 0.05: 
                    param.add_(torch.randn_like(param) * 0.05 * step_influence)

    def forward(self, x): # x: (batch, seq_len, embed_dim)
        x = self.initial_projection(x)
        for block in self.blocks:
            x = x + block.attention(block.norm1(x))
            x = x + block.mlp(block.norm2(x))
        return self.output_head(x.mean(dim=1)) # Global average pooling before head


# Define the UI configuration Pydantic Model for this VisKit
if PYDANTIC_AVAILABLE:
    class TreescopeUIConfig(BaseModel):
        html_height: int = Field(default=600, ge=200, le=3000, description="每个内嵌Treescope HTML的高度（像素）")
        views_per_row: int = Field(default=2, ge=1, le=3, description="每行显示的独立视图(组ID)数量") 
        enable_step_comparison: bool = Field(default=True, description="启用同组ID下当前步骤与上一步骤的对比视图")
        # comparison_views_per_row: Literal[1, 2] = Field(default=2, description="对比视图中每行显示的步骤数 (通常为2)")
else: 
    @dataclass
    class TreescopeUIConfig: # type: ignore
        html_height: int = 600
        views_per_row: int = 2
        enable_step_comparison: bool = True
        # comparison_views_per_row: int = 2


TREESCOPE_VIEW_COLLECTION_DATA_TYPE = "treescope_view_collection_v1" 
INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE = "single_treescope_html_v1"     


@register_viskit(name="treescope_model_viewer") 
class TreescopeModelViewVisKit(VisKit): 
    ui_config: TreescopeUIConfig
    _step_data_map: Dict[int, Dict[str, Dict[str, Any]]] # step -> group_id -> {path: Path, display_name: str}
    _overall_display_name: str 
    _data_loaded_once_treescope: bool 

    LOGICAL_DATA_SOURCE_NAME = "treescope_html_collection" 
    COLLECTION_DATA_TYPE_FOR_IDE = TREESCOPE_VIEW_COLLECTION_DATA_TYPE


    def __init__(self,
                 instance_id: str, 
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]], 
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(instance_id, trial_root_path, data_sources_map, specific_ui_config_dict)
        self._step_data_map = {} 
        self._data_loaded_once_treescope = False 
        
        primary_source_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 
        self._overall_display_name = "Treescope Views" 
        if primary_source_info and "display_name" in primary_source_info:
            self._overall_display_name = primary_source_info["display_name"]

    @classmethod
    def get_config_model(cls) -> Optional[Type[TreescopeUIConfig]]:
        return TreescopeUIConfig if PYDANTIC_AVAILABLE else None

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return cls.TREESCOPE_VIEW_COLLECTION_DATA_TYPE in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Treescope 模型/子模块查看器" # Updated display name

    @classmethod
    def _generate_example_payloads_and_steps(cls,
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        payloads = []
        example_steps = [0, 3, 6] 
        base_decay_factor = 0.5
        embed_dim_example = (data_sources_config or {}).get("embed_dim", 16)

        # 创建一个共享的基础模型实例，并在每个步骤中修改它
        # Create a shared base model instance and modify it at each step
        base_model = ExampleComplexModel(embed_dim=embed_dim_example, num_blocks=1)

        for i, step_val in enumerate(example_steps):
            # 模拟模型参数随步骤变化
            # Simulate model parameter changes with steps
            with torch.no_grad():
                for param in base_model.parameters():
                    param.mul_(base_decay_factor) # 应用衰减 Apply decay
                    if random.random() < 0.1: # 添加一些随机扰动 Add some random noise
                        param.add_(torch.randn_like(param) * 0.02)
            
            # 上报模型的不同部分
            # Report different parts of the model
            # 注意：这里传递的是模块的引用。Treescope会处理它。
            # Note: We are passing references to modules. Treescope will handle them.
            payloads.append({
                "data_payload": base_model.blocks[0].attention, 
                "step": step_val,
                "group_id": "block_0_attention", 
                "asset_name": f"attention_s{step_val}"
            })
            payloads.append({
                "data_payload": base_model.blocks[0].mlp, 
                "step": step_val,
                "group_id": "block_0_mlp", 
                "asset_name": f"mlp_s{step_val}"
            })
            if step_val > 0 : # 上报整个模型（频率可以低一些）Report entire model (can be less frequent)
                 payloads.append({
                    "data_payload": base_model, 
                    "step": step_val,
                    "group_id": "full_model_overview", 
                    "asset_name": f"full_model_s{step_val}"
                })
        return payloads

    def report_data(self,
                    data_payload: Any, 
                    step: int,
                    group_id: Optional[str] = None, 
                    asset_name: Optional[str] = None, 
                    **kwargs) -> List[Dict[str, Any]]:
        # (report_data 逻辑与之前版本类似，路径和返回的资产描述结构保持一致)
        # (report_data logic is similar to the previous version, path and returned asset description structure remain consistent)
        if group_id is None: group_id = self.instance_id 
        safe_asset_name = asset_name if asset_name and asset_name.strip() else f"treescope_view"
        safe_asset_name_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in safe_asset_name)
        group_id_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in group_id)
        
        output_dir_in_private_storage = Path(group_id_for_path) / f"step_{step}"
        output_dir_full = self.private_storage_path / output_dir_in_private_storage
        output_dir_full.mkdir(parents=True, exist_ok=True)

        html_file_name = f"{safe_asset_name_for_path}.html"
        html_file_full_path = output_dir_full / html_file_name

        try:
            with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
                with treescope.using_expansion_strategy(max_height=None): # max_height=None for full expansion
                    html_str = treescope.render_to_html(data_payload, compressed=True, roundtrip_mode=False)
            
            with open(html_file_full_path, "w", encoding="utf-8") as f: f.write(html_str)
            
            path_in_manifest = (self.private_storage_path.relative_to(self.trial_root_path) / output_dir_in_private_storage / html_file_name).as_posix()

            asset_description = {
                "asset_id": f"{self.instance_id}_{group_id}_s{step}_{safe_asset_name}", # 更具描述性的asset_id More descriptive asset_id
                "display_name": f"{group_id.replace('_',' ').title()} - {safe_asset_name} (S{step})", 
                "data_type_original": INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE, 
                "path": path_in_manifest, "related_step": step, "group_id": group_id, 
                "asset_name_source": asset_name 
            }
            return [asset_description]
        except Exception as e:
            print(f"错误: 步骤 {step} 生成Treescope HTML失败 (视件: {self.instance_id}, 组: {group_id}): {e}")
            return []


    def load_data(self) -> None:
        self._step_data_map = {} 
        source_collection_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 

        if not source_collection_info or source_collection_info.get("data_type") != self.COLLECTION_DATA_TYPE_FOR_IDE: 
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
               "path" in item_info and "related_step" in item_info and "group_id" in item_info:
                try:
                    step = int(item_info["related_step"])
                    group_id = str(item_info["group_id"]) 
                    relative_path_str = item_info["path"] 
                    full_path = (self.trial_root_path / relative_path_str).resolve()
                    asset_display_name = item_info.get("display_name", group_id) # 用于子标题 Get display name for subheader
                    
                    if full_path.exists() and full_path.is_file():
                        if step not in self._step_data_map:
                            self._step_data_map[step] = {}
                        # 存储路径和显示名称 Store path and display name
                        self._step_data_map[step][group_id] = {"path": full_path, "display_name": asset_display_name}
                        temp_all_steps.add(step)
                    else:
                        st.warning(f"视件 {self.instance_id}: HTML文件未找到: {full_path} (步骤 {step}, 组 {group_id})")
                except ValueError:
                    st.warning(f"视件 {self.instance_id}: item中的步骤值无效: {item_info.get('related_step')}")
                except Exception as e:
                    st.error(f"视件 {self.instance_id}: 处理item {item_info} 出错: {e}")
            else:
                st.warning(f"视件 {self.instance_id}: 数据源中的item格式无效或缺少group_id: {item_info}")
        
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

        with config_container.form(key=f"{self.instance_id}_treescope_cfg_form_manual"):
            current_config = self.ui_config
            new_values = {}
            
            new_values["html_height"] = st.number_input(
                "HTML 显示高度", min_value=200, max_value=3000, step=100,
                value=current_config.html_height, key=f"{self.instance_id}_cfg_html_height" 
            )
            new_values["views_per_row"] = st.slider( 
                "每行视图数 (组ID)", 1, 3, # 限制最大为3以避免过于拥挤 Limit max to 3 to avoid overcrowding
                value=current_config.views_per_row, key=f"{self.instance_id}_cfg_views_per_row"
            )
            new_values["enable_step_comparison"] = st.checkbox(
                "启用步骤对比视图", value=current_config.enable_step_comparison,
                key=f"{self.instance_id}_cfg_step_compare"
            )

            submitted = st.form_submit_button("应用设置")
            if submitted:
                try:
                    # 创建一个新的配置模型实例以进行验证
                    # Create a new config model instance for validation
                    updated_model = ConfigModel(**new_values) 
                    if self.ui_config.model_dump() != updated_model.model_dump():
                        self.ui_config = updated_model 
                        self.save_ui_config()
                        changed = True
                except ValidationError as ve: 
                    st.error(f"配置验证失败: {ve}")
                except Exception as e_cfg:
                    st.error(f"应用配置时出错: {e_cfg}")
        return changed


    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        report_container.markdown(f"#### 上报模型/子模块视图数据到 `{self.instance_id}` (Treescope)")
        
        with report_container.form(key=f"{self.instance_id}_treescope_report_form"):
            step = st.number_input("全局步骤 (Global Step)", min_value=0, value=self._current_global_step or 0, step=1)
            
            # 允许用户选择上报整个模型还是特定子模块
            # Allow user to choose reporting whole model or specific sub-module
            model_part_options = ["整个模型 (Full Model)", "注意力块 (Attention Block)", "MLP块 (MLP Block)"]
            selected_part_name = st.selectbox("选择要上报的模型部分 (Select Model Part to Report)", model_part_options)

            # group_id 和 asset_name 可以基于所选部分自动生成或允许用户输入
            # group_id and asset_name can be auto-generated based on selection or user-input
            default_group_id = selected_part_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            group_id = st.text_input("模型/参数组ID (Group ID)", value=default_group_id)
            asset_name = st.text_input("视图名称 (View Name)", value=f"{default_group_id}_s{step}")
            
            st.markdown("**模拟模型参数 (Simulate Model Parameters for Reporting):**")
            embed_dim_report = st.select_slider("嵌入维度 (Embedding Dim)", [16, 32], value=16, key=f"{self.instance_id}_rep_emb_dim")
            num_blocks_report = st.slider("块数量 (Num Blocks)", 1, 2, 1, key=f"{self.instance_id}_rep_num_blocks")
            step_influence_report = st.number_input("步骤影响因子 (Step Influence)", value=1.0, min_value=0.1, max_value=2.0, step=0.1, key=f"{self.instance_id}_rep_step_inf")

            submit_button = st.form_submit_button("上报此视图 (Report This View)")

            if submit_button:
                try:
                    model_to_report = ExampleComplexModel(
                        step_influence=step_influence_report,
                        embed_dim=embed_dim_report,
                        num_blocks=num_blocks_report
                    )
                    
                    payload_to_report: Any = model_to_report
                    if selected_part_name == "注意力块 (Attention Block)" and model_to_report.blocks:
                        payload_to_report = model_to_report.blocks[0].attention
                    elif selected_part_name == "MLP块 (MLP Block)" and model_to_report.blocks:
                        payload_to_report = model_to_report.blocks[0].mlp
                    # 如果是 "整个模型"，则 payload_to_report 保持为 model_to_report
                    # If "Full Model", payload_to_report remains model_to_report

                    return {
                        "data_payload": payload_to_report,
                        "step": int(step),
                        "group_id": group_id if group_id.strip() else None,
                        "asset_name": asset_name if asset_name.strip() else None
                    }
                except Exception as e:
                    st.error(f"创建/选择上报模型失败: {e}")
                    return None
        return None

    
    def render(self) -> None:
        st.subheader(self._overall_display_name) 

        with st.expander("显示设置 (Display Settings)", expanded=self.ui_config.enable_step_comparison): # 默认展开如果启用对比
            if self.render_config_ui(st.container()):
                st.rerun() 

        if not self._step_data_map: 
            if not self._data_loaded_once_treescope: 
                self.load_data()
                self._data_loaded_once_treescope = True 
            if not self._step_data_map: 
                st.info(f"视件 {self.instance_id}: 没有可显示的Treescope视图数据。")
                return

        if self._all_available_steps is None or not self._all_available_steps:
            if self._step_data_map: 
                self._all_available_steps = sorted(list(self._step_data_map.keys()))
            else: 
                st.warning("没有可用的步骤。"); return
            if self._current_global_step not in self._all_available_steps and self._all_available_steps:
                self._current_global_step = self._all_available_steps[0]

        step_to_display_current = self._get_closest_available_step(self._current_global_step)

        if step_to_display_current is None: # or step_to_display_current not in self._step_data_map: (map might be empty)
            st.warning(f"视件 {self.instance_id}: 在步骤 {self._current_global_step} (或附近) 未找到Treescope视图数据。")
            available_s = ", ".join(map(str, sorted(self._step_data_map.keys())))
            st.caption(f"可用步骤: {available_s if available_s else 'None'}")
            return

        views_for_current_step = self._step_data_map.get(step_to_display_current, {})
        if not views_for_current_step:
            st.info(f"视件 {self.instance_id}: 步骤 {step_to_display_current} 没有可显示的Treescope视图。")
            return

        # 确定要显示的group_ids (所有在此步骤有数据的group_id)
        # Determine group_ids to display (all group_ids that have data at this step)
        group_ids_to_render = sorted(list(views_for_current_step.keys()))
        
        if self._current_global_step is not None and step_to_display_current != self._current_global_step:
            st.caption(f"显示最接近步骤 {self._current_global_step} 的视图 (实际步骤 {step_to_display_current})。")
        else:
            st.caption(f"显示步骤 {step_to_display_current} 的视图。")

        num_groups = len(group_ids_to_render)
        views_per_row = self.ui_config.views_per_row

        for i in range(0, num_groups, views_per_row):
            group_chunk = group_ids_to_render[i : i + views_per_row]
            # 为每行创建列 (st.columns需要一个整数或列表)
            # Create columns for each row (st.columns needs an int or list)
            row_cols = st.columns(len(group_chunk)) 
            
            for j, group_id in enumerate(group_chunk):
                with row_cols[j]:
                    current_view_info = views_for_current_step.get(group_id)
                    if not current_view_info: continue

                    current_html_path = current_view_info["path"]
                    # 使用存储在 item_info 中的 display_name (如果有)
                    # Use display_name from item_info (if stored)
                    group_display_name = current_view_info.get("display_name", group_id.replace("_", " ").title())
                    
                    st.markdown(f"###### {group_display_name}") 

                    if self.ui_config.enable_step_comparison:
                        # 查找此group_id的上一个可用步骤
                        # Find previous available step for this group_id
                        prev_step_to_display = None
                        current_step_index_in_all = self._all_available_steps.index(step_to_display_current) if step_to_display_current in self._all_available_steps else -1
                        
                        # 向前搜索此group_id有数据的步骤
                        # Search backwards for a step where this group_id has data
                        for k_prev in range(current_step_index_in_all - 1, -1, -1):
                            candidate_prev_step = self._all_available_steps[k_prev]
                            if candidate_prev_step in self._step_data_map and group_id in self._step_data_map[candidate_prev_step]:
                                prev_step_to_display = candidate_prev_step
                                break
                        
                        if prev_step_to_display is not None:
                            prev_html_path = self._step_data_map[prev_step_to_display][group_id]["path"]
                            comparison_cols = st.columns(2)
                            with comparison_cols[0]:
                                st.caption(f"上一步 (S{prev_step_to_display})")
                                if prev_html_path.exists():
                                    with open(prev_html_path, "r", encoding="utf-8") as f: better_st_html(f.read(), height=self.ui_config.html_height, scrolling=True)
                                else: st.error(f"HTML文件未找到: {prev_html_path}")
                            with comparison_cols[1]:
                                st.caption(f"当前步骤 (S{step_to_display_current})")
                                if current_html_path.exists():
                                    with open(current_html_path, "r", encoding="utf-8") as f: better_st_html(f.read(), height=self.ui_config.html_height, scrolling=True)
                                else: st.error(f"HTML文件未找到: {current_html_path}")
                        else: # 没有上一步可对比，只显示当前
                            st.caption(f"当前步骤 (S{step_to_display_current}) (无上一步可对比)")
                            if current_html_path.exists():
                                with open(current_html_path, "r", encoding="utf-8") as f: better_st_html(f.read(), height=self.ui_config.html_height, scrolling=True)
                            else: st.error(f"HTML文件未找到: {current_html_path}")
                    else: # 不启用步骤对比
                        if current_html_path.exists():
                            with open(current_html_path, "r", encoding="utf-8") as f: better_st_html(f.read(), height=self.ui_config.html_height, scrolling=True)
                        else: st.error(f"HTML文件未找到: {current_html_path}")


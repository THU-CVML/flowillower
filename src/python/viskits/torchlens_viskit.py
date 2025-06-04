# flowillower/viskits/torchlens_viskit.py
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import shutil 
import html 
import random 
import os
import streamlit.components.v1 as components

# PyTorch, torchlens, treescope imports
import torch
import torch.nn as nn
# 路径修复 Path fix
if hasattr(torch, 'classes') and hasattr(torch.classes, '__file__') and torch.classes.__file__ is not None: 
    torch.classes.__path__ = [os.path.join(Path(torch.__file__).parent.as_posix(), Path(torch.classes.__file__).name)]
else:
    print("Warning: PyTorch C++ extension path fix could not be applied fully or is not needed.")

try:
    import torchlens as tl 
    TORCHLENS_AVAILABLE = True
except ImportError:
    TORCHLENS_AVAILABLE = False
    print("Warning: torchlens library not found. TorchlensFlowVisKit will not function correctly.")

try:
    import treescope      
    TREESCOPE_AVAILABLE = True
except ImportError:
    TREESCOPE_AVAILABLE = False
    print("Warning: treescope library not found. TorchlensFlowVisKit's HTML rendering will be affected.")


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
    Literal = str 
    from dataclasses import dataclass 


# 与 TreescopeVisKit 类似的示例模型，但输入维度可能不同
# Similar example model to TreescopeVisKit, but input dimensions might differ
class ExampleModelForTorchlens(nn.Module): 
    def __init__(self, step_influence=1.0, in_channels=1, embed_dim=8, num_blocks=1): # Adjusted for CNN-like input
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 模拟一个简单的卷积层开始
        # Simulate a simple convolutional start
        self.conv_start = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.relu_start = nn.ReLU(inplace=False)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim), # LayerNorm might expect specific input shape
                'attention_like': nn.Linear(embed_dim, embed_dim), # Simplified attention-like layer
                'norm2': nn.LayerNorm(embed_dim),
                'mlp_like': nn.Linear(embed_dim, embed_dim) # Simplified MLP-like layer
            })
            self.blocks.append(block)
        
        self.output_head = nn.Linear(embed_dim, 10) # Example output classes
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.mul_(step_influence * random.uniform(0.9, 1.1)) 
                if "bias" not in name and random.random() < 0.05: 
                    param.add_(torch.randn_like(param) * 0.05 * step_influence)

    def forward(self, x): # x: (batch, in_channels, H, W)
        x = self.relu_start(self.conv_start(x))
        # 需要将 x reshape 以适应 LayerNorm 和 Linear 层
        # Need to reshape x to fit LayerNorm and Linear layers
        # 例如，全局平均池化或展平
        # e.g., global average pooling or flatten
        if x.dim() == 4: # B C H W
            x = x.mean(dim=[2,3]) # Global Average Pooling -> (B, C_embed_dim)
        
        for block in self.blocks:
            # LayerNorm expects (N, ..., C)
            x_norm1 = block.norm1(x)
            x = x + block.attention_like(x_norm1)
            x_norm2 = block.norm2(x)
            x = x + block.mlp_like(x_norm2)
        return self.output_head(x) 


if PYDANTIC_AVAILABLE:
    class TorchlensUIConfig(BaseModel):
        pdf_height: int = Field(default=700, ge=300, le=2000, description="计算图PDF的显示高度（像素）")
        html_height: int = Field(default=700, ge=300, le=2000, description="Treescope HTML（张量流）的显示高度（像素）")
        enable_step_comparison: bool = Field(default=True, description="启用张量流HTML的步骤对比视图")
        # comparison_views_per_row: Literal[1, 2] = Field(default=2) # For HTML comparison
else: 
    @dataclass
    class TorchlensUIConfig: # type: ignore
        pdf_height: int = 700
        html_height: int = 700
        enable_step_comparison: bool = True
        # comparison_views_per_row: int = 2


TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE = "torchlens_output_collection_v1" 
INDIVIDUAL_TORCHLENS_OUTPUT_SET_DATA_TYPE = "single_torchlens_output_set_v1"     


@register_viskit(name="torchlens_flow_viewer") 
class TorchlensFlowVisKit(VisKit): 
    ui_config: TorchlensUIConfig
    # step -> group_id -> {"pdf_path": Path, "html_path": Path, "display_name": str}
    _step_data_map: Dict[int, Dict[str, Dict[str, Any]]] 
    _overall_display_name: str 
    _data_loaded_once_torchlens: bool  # 添加缺失的标志声明

    LOGICAL_DATA_SOURCE_NAME = "torchlens_output_collection" 
    COLLECTION_DATA_TYPE_FOR_IDE = TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE


    def __init__(self,
                 instance_id: str, 
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]], 
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(instance_id, trial_root_path, data_sources_map, specific_ui_config_dict)
        self._step_data_map = {} 
        self._data_loaded_once_torchlens = False  # 初始化标志
        
        primary_source_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 
        self._overall_display_name = "Torchlens Flow Views" 
        if primary_source_info and "display_name" in primary_source_info:
            self._overall_display_name = primary_source_info["display_name"]

    @classmethod
    def get_config_model(cls) -> Optional[Type[TorchlensUIConfig]]:
        return TorchlensUIConfig if PYDANTIC_AVAILABLE else None

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return cls.COLLECTION_DATA_TYPE_FOR_IDE in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Torchlens 计算流查看器"

    @classmethod
    def _generate_example_payloads_and_steps(cls,
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        payloads = []
        example_steps = [0, 2, 4] 
        base_decay_factor = 0.92
        group_id = (data_sources_config or {}).get("group_id", "example_cnn_flow_torchlens")
        asset_name_prefix = (data_sources_config or {}).get("asset_name_prefix", "flow_view")
        in_channels_example = (data_sources_config or {}).get("in_channels", 1)
        embed_dim_example = (data_sources_config or {}).get("embed_dim", 8)

        for i, step_val in enumerate(example_steps):
            step_influence_factor = base_decay_factor ** i 
            model_instance = ExampleModelForTorchlens(
                step_influence=step_influence_factor,
                in_channels=in_channels_example,
                embed_dim=embed_dim_example
            )
            example_input_tensor = torch.randn(1, in_channels_example, 28, 28) 
            
            payloads.append({
                "data_payload": {"model": model_instance, "input_tensor": example_input_tensor},
                "step": step_val,
                "group_id": group_id, 
                "asset_name": f"{asset_name_prefix}_s{step_val}"
            })
        return payloads

    def report_data(self,
                    data_payload: Dict[str, Any], 
                    step: int,
                    group_id: Optional[str] = None, 
                    asset_name: Optional[str] = None, 
                    **kwargs) -> List[Dict[str, Any]]:
        if not (TORCHLENS_AVAILABLE and TREESCOPE_AVAILABLE):
            print("错误: Torchlens 或 Treescope 未安装，无法报告数据。")
            return []

        if not isinstance(data_payload, dict) or "model" not in data_payload or "input_tensor" not in data_payload:
            print(f"错误: {self.get_display_name()} 的 report_data 期望 data_payload 是一个包含 'model' 和 'input_tensor' 的字典。")
            return []

        model_to_log = data_payload["model"]
        input_tensor_to_log = data_payload["input_tensor"]

        if not isinstance(model_to_log, nn.Module) or not isinstance(input_tensor_to_log, torch.Tensor):
            print("错误: data_payload 中的 'model' 必须是 nn.Module，'input_tensor' 必须是 torch.Tensor。")
            return []

        if group_id is None: group_id = self.instance_id 
        safe_asset_name = asset_name if asset_name and asset_name.strip() else f"torchlens_flow"
        safe_asset_name_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in safe_asset_name)
        group_id_for_path = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in group_id)
        
        output_dir_in_private_storage = Path(group_id_for_path) / f"step_{step}"
        output_dir_full = self.private_storage_path / output_dir_in_private_storage
        output_dir_full.mkdir(parents=True, exist_ok=True)

        pdf_file_name_base = f"{safe_asset_name_for_path}_graph" 
        torchlens_vis_outpath_prefix = (output_dir_full / pdf_file_name_base).as_posix()
        
        html_file_name = f"{safe_asset_name_for_path}_tensors.html"
        html_file_full_path = output_dir_full / html_file_name

        try:
            print(f"正在为步骤 {step} 运行 torchlens.log_forward_pass...")
            model_history = tl.log_forward_pass(
                model_to_log, input_tensor_to_log, layers_to_save='all', 
                vis_opt='unrolled', 
                vis_outpath=torchlens_vis_outpath_prefix, 
                vis_save_only=True,
                vis_fileformat="pdf"
            )
            
            # 更加详细的PDF文件查找逻辑
            possible_pdf_paths = [
                Path(torchlens_vis_outpath_prefix + ".pdf"),
                Path(torchlens_vis_outpath_prefix + ".gv.pdf"),
                output_dir_full / f"{pdf_file_name_base}.pdf",
                output_dir_full / f"{pdf_file_name_base}.gv.pdf"
            ]
            
            actual_pdf_file_path = None
            for pdf_path in possible_pdf_paths:
                if pdf_path.exists():
                    actual_pdf_file_path = pdf_path
                    print(f"找到PDF文件: {actual_pdf_file_path}")
                    break
            
            if actual_pdf_file_path is None:
                # 列出输出目录中的所有文件以进行调试
                print(f"未找到PDF文件。输出目录 {output_dir_full} 中的文件:")
                for file in output_dir_full.iterdir():
                    print(f"  - {file.name}")
                
                # 创建占位符PDF
                actual_pdf_file_path = output_dir_full / f"{pdf_file_name_base}_placeholder.pdf"
                placeholder_pdf_content = "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000125 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n196\n%%EOF"
                with open(actual_pdf_file_path, "w") as f_dummy_pdf:
                    f_dummy_pdf.write(placeholder_pdf_content)
                print(f"创建了占位符PDF: {actual_pdf_file_path}")

            # 生成张量HTML
            print(f"正在生成张量HTML...")
            tensor_contents_dict = {
                label: model_history[label].tensor_contents 
                for label in model_history.layer_labels
            }
            with treescope.using_expansion_strategy(max_height=None):
                html_str = treescope.render_to_html(tensor_contents_dict, compressed=True)
            with open(html_file_full_path, "w", encoding="utf-8") as f: 
                f.write(html_str)
            print(f"生成了HTML文件: {html_file_full_path}")
            
            base_relative_path = self.private_storage_path.relative_to(self.trial_root_path) / output_dir_in_private_storage
            pdf_path_in_manifest = (base_relative_path / actual_pdf_file_path.name).as_posix()
            html_path_in_manifest = (base_relative_path / html_file_name).as_posix()

            asset_description = {
                "asset_id": f"{self.instance_id}_{group_id}_s{step}_{safe_asset_name}_set", 
                "display_name": f"{group_id.replace('_',' ').title()} - {safe_asset_name} (S{step})", 
                "data_type_original": INDIVIDUAL_TORCHLENS_OUTPUT_SET_DATA_TYPE, 
                "related_step": step, "group_id": group_id, 
                "asset_name_source": asset_name,
                "paths": { 
                    "pdf_graph": pdf_path_in_manifest,
                    "tensors_html": html_path_in_manifest
                }
            }
            print(f"生成资产描述: {asset_description}")
            return [asset_description]
        except Exception as e:
            print(f"错误: 步骤 {step} 生成Torchlens/Treescope数据失败 (视件: {self.instance_id}, 组: {group_id}): {e}")
            import traceback
            traceback.print_exc()
            return []


    def load_data(self) -> None:
        print(f"[DEBUG] TorchlensFlowVisKit.load_data() 开始 - 实例: {self.instance_id}")
        print(f"[DEBUG] 当前 data_sources_map: {self.data_sources_map}")
        print(f"[DEBUG] LOGICAL_DATA_SOURCE_NAME: {self.LOGICAL_DATA_SOURCE_NAME}")
        
        self._step_data_map = {} 
        source_collection_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME) 
        print(f"[DEBUG] 获取到的 source_collection_info: {source_collection_info}")

        if not source_collection_info or source_collection_info.get("data_type") != self.COLLECTION_DATA_TYPE_FOR_IDE: 
            print(f"[DEBUG] 没有找到有效的数据源集合信息")
            print(f"[DEBUG] 期望的数据类型: {self.COLLECTION_DATA_TYPE_FOR_IDE}")
            if source_collection_info:
                print(f"[DEBUG] 实际的数据类型: {source_collection_info.get('data_type')}")
            self._all_available_steps = []
            return

        items = source_collection_info.get("items", [])
        if not isinstance(items, list):
            st.warning(f"视件 {self.instance_id}: 数据源中的 'items' 不是列表。")
            self._all_available_steps = []
            return
            
        print(f"[DEBUG] 找到 {len(items)} 个数据项")
        temp_all_steps = set()
        loaded_items_count = 0
        
        for item_info in items:
            print(f"[DEBUG] 处理 item: {item_info}")
            if isinstance(item_info, dict) and \
               item_info.get("data_type_original") == INDIVIDUAL_TORCHLENS_OUTPUT_SET_DATA_TYPE and \
               "paths" in item_info and isinstance(item_info["paths"], dict) and \
               "related_step" in item_info and "group_id" in item_info:
                try:
                    step = int(item_info["related_step"])
                    group_id = str(item_info["group_id"]) 
                    paths_dict = item_info["paths"]
                    pdf_relative_path = paths_dict.get("pdf_graph")
                    html_relative_path = paths_dict.get("tensors_html")
                    asset_display_name = item_info.get("display_name", group_id) 

                    if not pdf_relative_path or not html_relative_path:
                        st.warning(f"视件 {self.instance_id}: 步骤 {step} 的item缺少PDF或HTML路径。")
                        continue
                    
                    pdf_full_path = (self.trial_root_path / pdf_relative_path).resolve()
                    html_full_path = (self.trial_root_path / html_relative_path).resolve()
                    
                    print(f"[DEBUG] 检查文件 - PDF: {pdf_full_path.exists()}, HTML: {html_full_path.exists()}")
                    print(f"[DEBUG] PDF路径: {pdf_full_path}")
                    print(f"[DEBUG] HTML路径: {html_full_path}")
                    
                    if pdf_full_path.exists() and html_full_path.exists():
                        if step not in self._step_data_map:
                            self._step_data_map[step] = {}
                        self._step_data_map[step][group_id] = {
                            "pdf_path": pdf_full_path, 
                            "html_path": html_full_path,
                            "display_name": asset_display_name
                        }
                        temp_all_steps.add(step)
                        loaded_items_count += 1
                        print(f"[DEBUG] 成功加载步骤 {step}, 组 {group_id}")
                    else:
                        st.warning(f"视件 {self.instance_id}: 步骤 {step}, 组 {group_id} 的一个或多个文件未找到。PDF exists: {pdf_full_path.exists()}, HTML exists: {html_full_path.exists()}")
                except ValueError:
                    st.warning(f"视件 {self.instance_id}: item中的步骤值无效: {item_info.get('related_step')}")
                except Exception as e:
                    st.error(f"视件 {self.instance_id}: 处理item {item_info} 出错: {e}")
            else:
                print(f"[DEBUG] 跳过无效item: {item_info}")
                print(f"[DEBUG] - 是字典: {isinstance(item_info, dict)}")
                if isinstance(item_info, dict):
                    print(f"[DEBUG] - data_type_original: {item_info.get('data_type_original')} (期望: {INDIVIDUAL_TORCHLENS_OUTPUT_SET_DATA_TYPE})")
                    print(f"[DEBUG] - 有paths: {'paths' in item_info}")
                    print(f"[DEBUG] - 有related_step: {'related_step' in item_info}")
                    print(f"[DEBUG] - 有group_id: {'group_id' in item_info}")
        
        self._all_available_steps = sorted(list(temp_all_steps))
        print(f"[DEBUG] 加载完成 - 总共加载了 {loaded_items_count} 个有效项，可用步骤: {self._all_available_steps}")


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
        with config_container.form(key=f"{self.instance_id}_torchlens_cfg_form"):
            current_config = self.ui_config
            new_values = {}
            new_values["pdf_height"] = st.number_input("PDF高度", min_value=300, value=current_config.pdf_height, step=50, key=f"{self.instance_id}_pdf_h")
            new_values["html_height"] = st.number_input("HTML高度", min_value=300, value=current_config.html_height, step=50, key=f"{self.instance_id}_html_h")
            new_values["enable_step_comparison"] = st.checkbox("启用HTML步骤对比", value=current_config.enable_step_comparison, key=f"{self.instance_id}_html_comp")
            
            submitted = st.form_submit_button("应用设置")
            if submitted:
                try:
                    updated_model = ConfigModel(**new_values) 
                    if self.ui_config.model_dump() != updated_model.model_dump():
                        self.ui_config = updated_model 
                        self.save_ui_config()
                        changed = True
                except ValidationError as ve: st.error(f"配置验证失败: {ve}")
                except Exception as e_cfg: st.error(f"应用配置时出错: {e_cfg}")
        return changed

    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        report_container.markdown(f"#### 上报模型计算流数据到 `{self.instance_id}` (Torchlens)")
        with report_container.form(key=f"{self.instance_id}_torchlens_report_form"):
            step = st.number_input("全局步骤 (Global Step)", min_value=0, value=self._current_global_step or 0, step=1)
            group_id = st.text_input("模型/分析组ID (Model/Analysis Group ID)", value="cnn_layer_outputs")
            asset_name = st.text_input("视图名称 (View Name)", value=f"flow_s{step}")
            
            st.markdown("**模拟模型与输入 (Simulate Model & Input for Reporting):**")
            in_channels = st.select_slider("输入通道数 (In Channels)", [1, 3], value=1, key=f"{self.instance_id}_report_in_channels")
            embed_dim = st.select_slider("嵌入/特征维度 (Embed/Feature Dim)", [8, 16, 32], value=8, key=f"{self.instance_id}_report_embed_dim")
            step_influence = st.number_input("步骤影响因子 (Step Influence)", value=1.0, step=0.1, key=f"{self.instance_id}_report_step_influence")

            submit_button = st.form_submit_button("上报计算流数据 (Report Flow Data)")
            if submit_button:
                try:
                    model = ExampleModelForTorchlens(step_influence=step_influence, in_channels=in_channels, embed_dim=embed_dim)
                    input_tensor = torch.randn(1, in_channels, 28, 28) 
                    return {
                        "data_payload": {"model": model, "input_tensor": input_tensor},
                        "step": int(step), "group_id": group_id, "asset_name": asset_name
                    }
                except Exception as e: st.error(f"创建示例模型/输入失败: {e}"); return None
        return None

    def render(self) -> None:
        st.subheader(self._overall_display_name) 
        with st.expander("显示设置", expanded=self.ui_config.enable_step_comparison):
            if self.render_config_ui(st.container()): st.rerun() 

        if not self._step_data_map: 
            if not self._data_loaded_once_torchlens: 
                self.load_data()
                self._data_loaded_once_torchlens = True 
            if not self._step_data_map: 
                st.info(f"视件 {self.instance_id}: 没有可显示的Torchlens数据。")
                return

        if not self._all_available_steps:
            if self._step_data_map: self._all_available_steps = sorted(list(self._step_data_map.keys()))
            else: st.warning("没有可用的步骤。"); return
            # 确保 _current_global_step 在 _all_available_steps 中有效
            # Ensure _current_global_step is valid within _all_available_steps
            if self._all_available_steps and (self._current_global_step is None or self._current_global_step not in self._all_available_steps) :
                self._current_global_step = self._all_available_steps[0]


        step_to_display_current = self._get_closest_available_step(self._current_global_step)
        if step_to_display_current is None:
            st.warning(f"视件 {self.instance_id}: 在步骤 {self._current_global_step} (或附近) 未找到数据。"); return

        data_for_current_step = self._step_data_map.get(step_to_display_current, {})
        if not data_for_current_step:
            st.info(f"视件 {self.instance_id}: 步骤 {step_to_display_current} 没有数据。"); return
        
        group_ids_at_step = sorted(list(data_for_current_step.keys()))
        if not group_ids_at_step:
            st.info(f"视件 {self.instance_id}: 步骤 {step_to_display_current} 没有可显示的组。"); return
        
        active_group_id = group_ids_at_step[0] # 默认显示第一个组 Default to showing the first group
        # TODO: 如果一个步骤有多个group_id的数据，可以提供UI让用户选择要看哪个group_id
        # If a step has data for multiple group_ids, could provide UI to select which one to view.
        
        current_view_files = data_for_current_step.get(active_group_id)

        if not current_view_files or not current_view_files.get("pdf_path") or not current_view_files.get("html_path"):
            st.warning(f"视件 {self.instance_id}: 步骤 {step_to_display_current}, 组 {active_group_id} 数据文件不完整。"); return

        current_pdf_path = current_view_files["pdf_path"]
        current_html_path = current_view_files["html_path"]
        view_display_name = current_view_files.get("display_name", active_group_id.replace("_"," ").title())

        st.markdown(f"#### {view_display_name}")
        if self._current_global_step is not None and step_to_display_current != self._current_global_step:
            st.caption(f"显示最接近步骤 {self._current_global_step} 的数据 (实际步骤 {step_to_display_current})。")
        else:
            st.caption(f"显示步骤 {step_to_display_current} 的数据。")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### 计算图 (Computation Graph)")
            if current_pdf_path.exists():
                try:
                    from streamlit_pdf_viewer import pdf_viewer 
                    pdf_viewer(current_pdf_path.read_bytes(), height=self.ui_config.pdf_height)
                except ImportError: st.error("streamlit-pdf-viewer 未安装。请运行 'pip install streamlit-pdf-viewer' 以查看PDF。")
                except Exception as e: st.error(f"渲染PDF '{current_pdf_path}' 失败: {e}")
            else: st.warning(f"PDF文件未找到: {current_pdf_path}")
        
        with col_right:
            st.markdown("##### 张量流 (Tensor Flow - Treescope)")
            if self.ui_config.enable_step_comparison:
                prev_step_to_display = None
                if step_to_display_current in self._all_available_steps: # 确保当前步骤有效
                    current_step_index_in_all = self._all_available_steps.index(step_to_display_current)
                    for k_prev in range(current_step_index_in_all - 1, -1, -1):
                        candidate_prev_step = self._all_available_steps[k_prev]
                        if candidate_prev_step in self._step_data_map and active_group_id in self._step_data_map[candidate_prev_step]:
                            prev_step_to_display = candidate_prev_step; break
                
                if prev_step_to_display is not None:
                    prev_html_path = self._step_data_map[prev_step_to_display][active_group_id]["html_path"]
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.caption(f"上一步 (S{prev_step_to_display})")
                        if prev_html_path.exists():
                            with open(prev_html_path, "r", encoding="utf-8") as f: components.html(f.read(), height=self.ui_config.html_height, scrolling=True)
                        else: st.error(f"HTML文件未找到: {prev_html_path}")
                    with comp_col2:
                        st.caption(f"当前 (S{step_to_display_current})")
                        if current_html_path.exists():
                            with open(current_html_path, "r", encoding="utf-8") as f: components.html(f.read(), height=self.ui_config.html_height, scrolling=True)
                        else: st.error(f"HTML文件未找到: {current_html_path}")
                else: 
                    st.caption(f"当前 (S{step_to_display_current}) (无上一步可对比)")
                    if current_html_path.exists():
                        with open(current_html_path, "r", encoding="utf-8") as f: components.html(f.read(), height=self.ui_config.html_height, scrolling=True)
                    else: st.error(f"HTML文件未找到: {current_html_path}")
            else: 
                if current_html_path.exists():
                    with open(current_html_path, "r", encoding="utf-8") as f: components.html(f.read(), height=self.ui_config.html_height, scrolling=True)
                else: st.error(f"HTML文件未找到: {current_html_path}")


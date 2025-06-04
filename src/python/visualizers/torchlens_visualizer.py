# flowillower/visualizers/torchlens_visualizer.py
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
import html

# PyTorch, torchlens, treescope imports
import torch
import torch.nn as nn
import torchlens as tl # Assuming torchlens is installed
import treescope      # Assuming treescope is installed

from .base_visualizer import VisualizationComponent, register_visualizer

# Define the data type this visualizer handles.
# This type represents a collection of step-dependent torchlens outputs (PDF graph + Treescope HTML of tensors)
TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE = "torchlens_output_collection"
# This is the data type for an individual set of (PDF, HTML) files for a given step, as logged in the manifest.
INDIVIDUAL_TORCHLENS_OUTPUT_DATA_TYPE = "single_torchlens_output_set"

# Define a simple PyTorch model for example generation
class SimpleCNN(nn.Module):
    def __init__(self, step_influence=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(6 * 7 * 7, 10) # Assuming input 1x28x28 -> 6x7x7 after convs/pools

        with torch.no_grad():
            for param in self.parameters():
                param.mul_(step_influence + 0.5) # Add some base to avoid zeroing out too quickly

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 6 * 7 * 7)
        x = self.fc(x)
        return x

@register_visualizer(name="torchlens_flow_viewer")
class TorchlensFlowVisualizer(VisualizationComponent):
    """
    Visualizes the output of torchlens: a computation graph (PDF) and
    a Treescope HTML representation of tensor contents for a given model and input,
    allowing exploration across different global_steps.
    可视化torchlens的输出：计算图（PDF）和张量内容的Treescope HTML表示，
    允许按不同的global_step进行浏览。
    """

    _step_to_data_paths_map: Dict[int, Dict[str, Path]] # Stores {step: {"pdf_path": Path, "html_path": Path}}
    _display_name_for_panel: str

    def __init__(self,
                 component_instance_id: str,
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]],
                 component_specific_config: Dict[str, Any] = None):
        super().__init__(component_instance_id, trial_root_path, data_sources_map, component_specific_config)
        self._step_to_data_paths_map = {}
        self.load_component_config()

        primary_source_info = self._get_data_asset_info("default")
        self._display_name_for_panel = "Torchlens Flow View"
        if primary_source_info and "display_name" in primary_source_info:
            self._display_name_for_panel = primary_source_info["display_name"]
        
        self.html_height = self.config.get("html_height", 700)
        self.pdf_height = self.config.get("pdf_height", 700)
        self.pdf_width = self.config.get("pdf_width", None) # None for auto-width

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        """
        This visualizer handles a collection of torchlens outputs.
        The main app should group individual 'single_torchlens_output_set' assets
        and pass it as a 'torchlens_output_collection'.
        """
        return TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Torchlens 计算流查看器 (Torchlens Flow Viewer)"

    @classmethod
    def generate_example_data(cls, example_data_path: Path,
                              data_sources_config: Optional[Dict[str, Dict[str, Any]]] = None
                             ) -> Dict[str, Dict[str, Any]]:
        """
        Generates example data: a PDF graph and a Treescope HTML of tensor contents
        for a simple PyTorch model at different steps.
        """
        group_id = data_sources_config.get("group_id", "example_cnn_flow") if data_sources_config else "example_cnn_flow"
        
        # Data will be stored under example_data_path / f"torchlens_{group_id}" / f"step_{step_val}" / (graph.pdf, tensors.html)
        group_data_dir_name = f"torchlens_{group_id}"
        group_data_root_full_path = example_data_path / group_data_dir_name # e.g. .../example_assets_for_ide/torchlens_example_cnn_flow
        group_data_root_full_path.mkdir(parents=True, exist_ok=True)

        items_for_manifest = []
        example_steps = [0, 3, 7] # Fewer steps due to potentially slow generation
        base_decay_factor = 0.90

        for i, step_val in enumerate(example_steps):
            step_output_dir_name = f"step_{step_val}"
            step_output_full_path = group_data_root_full_path / step_output_dir_name # .../torchlens_example_cnn_flow/step_0
            step_output_full_path.mkdir(parents=True, exist_ok=True)

            pdf_file_name = "computation_graph.gv.pdf" # torchlens adds .gv.pdf
            html_file_name = "tensor_contents.html"
            
            pdf_full_path = step_output_full_path / pdf_file_name
            html_full_path = step_output_full_path / html_file_name

            # Paths for manifest (relative to trial_root_path)
            # example_data_path.name is "example_assets_for_ide"
            pdf_path_in_manifest = Path(example_data_path.name) / group_data_dir_name / step_output_dir_name / pdf_file_name
            html_path_in_manifest = Path(example_data_path.name) / group_data_dir_name / step_output_dir_name / html_file_name

            try:
                step_influence = base_decay_factor ** i
                model = SimpleCNN(step_influence=step_influence)
                # Example input: batch of 1, 1 channel, 28x28 image
                example_input_x = torch.randn(1, 1, 28, 28) * step_influence 

                # 1. Use torchlens to log forward pass and generate PDF
                #    The vis_outpath for torchlens should be the base name, it appends .gv and .pdf
                #    So we pass the path to the directory and the base name.
                graph_base_name_for_tl = (step_output_full_path / "computation_graph").as_posix()

                model_history = tl.log_forward_pass(
                    model, example_input_x, layers_to_save='all', 
                    vis_opt='unrolled', # or 'rolled'
                    vis_outpath=graph_base_name_for_tl, # torchlens will append .gv and format
                    vis_save_only=True,
                    vis_fileformat="pdf" # This should ensure .pdf is created
                )
                
                # Check if PDF was created (torchlens might name it slightly differently, e.g., adding .gv)
                # For simplicity, we assume it creates "computation_graph.gv.pdf" at step_output_full_path
                # If torchlens creates "computation_graph.pdf", adjust pdf_file_name accordingly.
                # The path used in manifest should match the actual created file.
                # We named it pdf_file_name = "computation_graph.gv.pdf"
                # If torchlens creates "computation_graph.pdf" instead of "computation_graph.gv.pdf",
                # then pdf_file_name should be "computation_graph.pdf"
                # For now, assume torchlens respects the full vis_outpath for the base and adds .pdf
                # Let's ensure the file exists with the expected name. If not, try to find it.
                
                # If torchlens saves as `graph_base_name_for_tl + ".pdf"`
                # pdf_full_path = Path(graph_base_name_for_tl + ".pdf") # This would be the actual path
                # pdf_path_in_manifest = Path(example_data_path.name) / group_data_dir_name / step_output_dir_name / pdf_full_path.name

                # For now, stick to the assumption that pdf_file_name is correct.
                # If the actual generated PDF is just "computation_graph.pdf", then pdf_file_name should be that.
                # Let's assume torchlens creates "computation_graph.pdf" if format is "pdf"
                # Re-evaluating: `vis_outpath` is a path prefix.
                # If `vis_outpath` is `step_output_full_path / "computation_graph"`,
                # and `vis_fileformat` is `pdf`, it should create `step_output_full_path / "computation_graph.pdf"`.
                # So, let's adjust pdf_file_name.
                
                actual_pdf_file_name = "computation_graph.pdf" # More likely name
                pdf_full_path = step_output_full_path / actual_pdf_file_name
                pdf_path_in_manifest = Path(example_data_path.name) / group_data_dir_name / step_output_dir_name / actual_pdf_file_name


                if not pdf_full_path.exists():
                    # Fallback if the primary expected name isn't found
                    fallback_pdf_path = step_output_full_path / "computation_graph.gv.pdf"
                    if fallback_pdf_path.exists():
                        pdf_full_path = fallback_pdf_path
                        pdf_path_in_manifest = Path(example_data_path.name) / group_data_dir_name / step_output_dir_name / "computation_graph.gv.pdf"
                    else:
                        st.warning(f"Torchlens PDF for step {step_val} not found at expected paths.")
                        # Continue without PDF for this step, or raise error

                # 2. Extract tensor contents and render with Treescope
                tensor_contents_dict = {
                    label: model_history[label].tensor_contents 
                    for label in model_history.layer_labels
                }
                
                with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
                    with treescope.using_expansion_strategy(max_height=9999):
                        html_str = treescope.render_to_html(tensor_contents_dict, compressed=True)
                
                with open(html_full_path, "w", encoding="utf-8") as f:
                    f.write(html_str)
                
                items_for_manifest.append({
                    "asset_id": f"{group_id}_step_{step_val}_output_set",
                    "data_type_original": INDIVIDUAL_TORCHLENS_OUTPUT_DATA_TYPE,
                    "related_step": step_val,
                    "display_name": f"{group_id.replace('_', ' ').title()} (Step {step_val})",
                    "paths": { # Store paths to both files
                        "pdf_graph": str(pdf_path_in_manifest),
                        "tensors_html": str(html_path_in_manifest)
                    }
                })

            except Exception as e:
                st.error(f"为步骤 {step_val} 生成Torchlens/Treescope示例数据失败: {e}")
                st.exception(e) # Print full traceback for debugging
        
        if not items_for_manifest:
            st.warning("未能成功生成任何Torchlens/Treescope示例数据。请确保torch, torchlens, treescope已安装。")

        return {
            "default": { 
                "asset_id": f"collection_torchlens_{group_id}",
                "data_type": TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE,
                "display_name": f"{group_id.replace('_', ' ').title()} - Torchlens Flow",
                "items": items_for_manifest,
                "group_id_source": group_id 
            }
        }

    def load_data(self) -> None:
        self._step_to_data_paths_map = {}
        source_info = self._get_data_asset_info("default")

        if not source_info or source_info.get("data_type") != TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE:
            st.warning(f"组件 {self.component_instance_id}: 数据源 'default' 缺失或类型不是 '{TORCHLENS_OUTPUT_COLLECTION_DATA_TYPE}'.")
            return

        items = source_info.get("items", [])
        if not isinstance(items, list):
            st.warning(f"组件 {self.component_instance_id}: 数据源中的 'items' 不是列表。")
            return
            
        temp_all_steps = set()
        for item_info in items:
            if isinstance(item_info, dict) and "paths" in item_info and isinstance(item_info["paths"], dict) and "related_step" in item_info:
                try:
                    step = int(item_info["related_step"])
                    paths_dict = item_info["paths"]
                    
                    pdf_relative_path = paths_dict.get("pdf_graph")
                    html_relative_path = paths_dict.get("tensors_html")

                    if not pdf_relative_path or not html_relative_path:
                        st.warning(f"组件 {self.component_instance_id}: 步骤 {step} 的item缺少PDF或HTML路径。")
                        continue

                    pdf_full_path = (self.trial_root_path / pdf_relative_path).resolve()
                    html_full_path = (self.trial_root_path / html_relative_path).resolve()
                    
                    # For now, we only check existence. Actual content loading happens in render.
                    # We could add checks here if files are truly valid.
                    if pdf_full_path.exists() and pdf_full_path.is_file() and \
                       html_full_path.exists() and html_full_path.is_file():
                        self._step_to_data_paths_map[step] = {
                            "pdf_path": pdf_full_path,
                            "html_path": html_full_path
                        }
                        temp_all_steps.add(step)
                    else:
                        st.warning(f"组件 {self.component_instance_id}: 步骤 {step} 的一个或多个文件未找到。"
                                   f"PDF: {pdf_full_path} (存在: {pdf_full_path.exists()}), "
                                   f"HTML: {html_full_path} (存在: {html_full_path.exists()})")
                except ValueError:
                    st.warning(f"组件 {self.component_instance_id}: item中的步骤值无效: {item_info.get('related_step')}")
                except Exception as e:
                    st.error(f"组件 {self.component_instance_id}: 处理item {item_info} 出错: {e}")
            else:
                st.warning(f"组件 {self.component_instance_id}: 数据源中的item格式无效: {item_info}")
        
        if not self._all_available_steps and temp_all_steps:
            self._all_available_steps = sorted(list(temp_all_steps))
        elif temp_all_steps and self._all_available_steps is not None:
            if set(self._all_available_steps) != temp_all_steps:
                 self._all_available_steps = sorted(list(temp_all_steps))

    def render_settings_ui(self):
        st.markdown("##### 组件设置 (Component Settings)")
        col1, col2 = st.columns(2)
        with col1:
            new_pdf_height = st.number_input(
                "PDF 显示高度 (PDF Display Height)",
                min_value=300, max_value=2000, step=50,
                value=self.pdf_height,
                key=f"{self.component_instance_id}_pdf_height",
            )
            if new_pdf_height != self.pdf_height:
                self.pdf_height = new_pdf_height
                self.config["pdf_height"] = new_pdf_height
                self.save_component_config()
                st.rerun()
        with col2:
            new_html_height = st.number_input(
                "HTML 显示高度 (HTML Display Height)",
                min_value=300, max_value=2000, step=50,
                value=self.html_height,
                key=f"{self.component_instance_id}_html_height",
            )
            if new_html_height != self.html_height:
                self.html_height = new_html_height
                self.config["html_height"] = new_html_height
                self.save_component_config()
                st.rerun()


    def render(self) -> None:
        st.subheader(self._display_name_for_panel)

        with st.expander("显示设置 (Display Settings)", expanded=False):
            self.render_settings_ui()

        if not self._step_to_data_paths_map:
            if not hasattr(self, '_data_loaded_once_torchlens'): 
                self.load_data()
                self._data_loaded_once_torchlens = True 
            if not self._step_to_data_paths_map:
                st.info(f"组件 {self.component_instance_id}: 没有可显示的Torchlens/Treescope数据。")
                return

        if self._all_available_steps is None or not self._all_available_steps:
            if self._step_to_data_paths_map:
                self._all_available_steps = sorted(list(self._step_to_data_paths_map.keys()))
            else: # Should have been caught by the check above
                st.warning("没有可用的步骤。No available steps.")
                return


        step_to_display = self._get_closest_available_step(self._current_global_step)

        if step_to_display is None or step_to_display not in self._step_to_data_paths_map:
            st.warning(f"组件 {self.component_instance_id}: 在步骤 {self._current_global_step} (或附近) 未找到数据。")
            available_s = ", ".join(map(str, sorted(self._step_to_data_paths_map.keys())))
            st.caption(f"可用步骤 (Available steps): {available_s if available_s else 'None'}")
            return

        data_paths = self._step_to_data_paths_map[step_to_display]
        pdf_path = data_paths.get("pdf_path")
        html_path = data_paths.get("html_path")

        if self._current_global_step is not None and step_to_display != self._current_global_step:
            st.caption(f"显示最接近步骤 {self._current_global_step} 的数据 (实际步骤 {step_to_display})。"
                       f"Displaying data for step {step_to_display}, closest to target step {self._current_global_step}.")
        else:
            st.caption(f"显示步骤 {step_to_display} 的数据。Displaying data for step {step_to_display}.")

        col_pdf, col_html = st.columns(2)

        with col_pdf:
            st.markdown("##### 计算图 (Computation Graph - PDF)")
            if pdf_path and pdf_path.exists():
                try:
                    # 确保 streamlit_pdf_viewer 已安装
                    # Ensure streamlit_pdf_viewer is installed
                    from streamlit_pdf_viewer import pdf_viewer # Lazy import
                    # pdf_viewer 需要字节数据或路径字符串
                    # pdf_viewer needs bytes data or path string
                    pdf_viewer(pdf_path.read_bytes(), height=self.pdf_height, width=self.pdf_width) 
                    # 或者使用 st.link_button 提供下载
                    # Or provide download using st.link_button
                    with open(pdf_path, "rb") as f_pdf:
                       st.download_button("下载PDF (Download PDF)", f_pdf, file_name=pdf_path.name, mime="application/pdf")

                except ImportError:
                    st.error("streamlit-pdf-viewer 未安装。请运行 'pip install streamlit-pdf-viewer'。")
                    st.markdown(f"[点击下载PDF (Click to download PDF)]({pdf_path.as_uri()}) (如果浏览器支持)")
                except Exception as e:
                    st.error(f"渲染PDF '{pdf_path}' 失败: {e}")
            else:
                st.warning(f"PDF文件未找到: {pdf_path}")

        with col_html:
            st.markdown("##### 张量内容 (Tensor Contents - Treescope HTML)")
            if html_path and html_path.exists():
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=self.html_height, scrolling=True)
                except Exception as e:
                    st.error(f"渲染HTML文件 '{html_path}' 失败: {e}")
            else:
                st.warning(f"HTML文件未找到: {html_path}")


# src/visualizers/treescope_visualizer.py
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
import html

# 新增导入 (New imports)
import torch
import torch.nn as nn
import os
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
import treescope  

from .base_visualizer import VisualizationComponent, register_visualizer

TREESCOPE_VIEW_COLLECTION_DATA_TYPE = "treescope_view_collection"
INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE = "single_treescope_html"


# 定义一个简单的PyTorch模型用于示例 (Define a simple PyTorch model for example)
class ExamplePyTorchModel(nn.Module):
    def __init__(self, step_influence=1.0):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
        # 用step_influence来改变权重，模拟训练过程
        # Use step_influence to change weights, simulating training
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(
                    step_influence
                )  # 简单地乘以一个因子 Simply multiply by a factor

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@register_visualizer(name="treescope_model_viewer")
class TreescopeModelViewVisualizer(VisualizationComponent):
    """
    Visualizes a collection of Treescope HTML outputs for a specific group of model parameters,
    allowing exploration across different global_steps.
    可视化一组特定模型参数的Treescope HTML输出集合，允许按不同的global_step进行浏览。
    """

    _step_to_html_map: Dict[int, Path]
    _display_name_for_panel: str

    def __init__(
        self,
        component_instance_id: str,
        trial_root_path: Path,
        data_sources_map: Dict[str, Dict[str, Any]],
        component_specific_config: Dict[str, Any] = None,
    ):
        super().__init__(
            component_instance_id,
            trial_root_path,
            data_sources_map,
            component_specific_config,
        )
        self._step_to_html_map = {}
        self.load_component_config()

        primary_source_info = self._get_data_asset_info("default")
        self._display_name_for_panel = "Treescope View"
        if primary_source_info and "display_name" in primary_source_info:
            self._display_name_for_panel = primary_source_info["display_name"]

        self.html_height = self.config.get(
            "html_height", 700
        )  # 增加默认高度 Increase default height

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return TREESCOPE_VIEW_COLLECTION_DATA_TYPE in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Treescope 模型查看器 (Treescope Model Viewer)"

    @classmethod
    def generate_example_data(
        cls,
        example_data_path: Path,
        data_sources_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generates a few HTML files by rendering simple PyTorch models (with varying parameters)
        using the actual `treescope.render_to_html` function.
        使用实际的 `treescope.render_to_html` 函数，通过渲染简单的PyTorch模型（参数变化）来生成一些HTML文件。
        """
        group_id = (
            data_sources_config.get("group_id", "example_pytorch_model")
            if data_sources_config
            else "example_pytorch_model"
        )

        group_data_dir_name = f"treescope_{group_id}"
        group_data_full_path = example_data_path / group_data_dir_name
        group_data_full_path.mkdir(parents=True, exist_ok=True)

        items_for_manifest = []
        example_steps = [0, 5, 10, 15]  # 示例步骤 Example steps
        base_decay_factor = (
            0.95  # 模拟权重衰减的基础因子 Base factor for simulated weight decay
        )

        for i, step_val in enumerate(example_steps):
            file_name = f"view_step_{step_val}.html"
            html_file_full_path = group_data_full_path / file_name

            path_in_manifest = (
                Path(example_data_path.name) / group_data_dir_name / file_name
            )

            try:
                # 1. 创建一个简单的PyTorch模型实例
                # 1. Create a simple PyTorch model instance
                # step_influence 使得每个step的模型参数略有不同
                # step_influence makes model parameters slightly different for each step
                step_influence_factor = base_decay_factor**i
                model_instance = ExamplePyTorchModel(
                    step_influence=step_influence_factor
                )

                # 2. 使用 treescope 将模型渲染为 HTML
                # 2. Use treescope to render the model to HTML
                # 注意：如果模型很大，这可能会比较慢。对于示例，我们保持模型简单。
                # Note: This might be slow if the model is large. For examples, we keep the model simple.
                with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
                    with treescope.using_expansion_strategy(max_height=9999):
                        html_str = treescope.render_to_html(
                            model_instance, compressed=True, roundtrip_mode=False
                        )

                with open(html_file_full_path, "w", encoding="utf-8") as f:
                    f.write(html_str)

                items_for_manifest.append(
                    {
                        "asset_id": f"{group_id}_step_{step_val}",
                        "path": str(path_in_manifest),
                        "related_step": step_val,
                        "data_type_original": INDIVIDUAL_TREESCOPE_HTML_DATA_TYPE,
                        "display_name": f"{group_id.replace('_', ' ').title()} (Step {step_val})",
                    }
                )
            except Exception as e:
                st.error(f"为步骤 {step_val} 生成Treescope HTML失败: {e}")

        if not items_for_manifest:
            st.warning(
                "未能成功生成任何示例Treescope HTML。请确保treescope和torch已安装。No example Treescope HTML items were successfully generated. Ensure treescope and torch are installed."
            )

        return {
            "default": {
                "asset_id": f"collection_{group_id}",
                "data_type": TREESCOPE_VIEW_COLLECTION_DATA_TYPE,
                "display_name": f"{group_id.replace('_', ' ').title()} Views",
                "items": items_for_manifest,
                "group_id_source": group_id,
            }
        }

    def load_data(self) -> None:
        self._step_to_html_map = {}
        source_info = self._get_data_asset_info("default")

        if (
            not source_info
            or source_info.get("data_type") != TREESCOPE_VIEW_COLLECTION_DATA_TYPE
        ):
            st.warning(
                f"组件 {self.component_instance_id}: "
                f"数据源 'default' 缺失或类型不是 '{TREESCOPE_VIEW_COLLECTION_DATA_TYPE}'。"
            )
            return

        items = source_info.get("items", [])
        if not isinstance(items, list):
            st.warning(
                f"组件 {self.component_instance_id}: 数据源中的 'items' 不是列表。"
            )
            return

        temp_all_steps = set()
        for item_info in items:
            if (
                isinstance(item_info, dict)
                and "path" in item_info
                and "related_step" in item_info
            ):
                try:
                    step = int(item_info["related_step"])
                    relative_path_str = item_info["path"]
                    full_path = (self.trial_root_path / relative_path_str).resolve()

                    if full_path.exists() and full_path.is_file():
                        self._step_to_html_map[step] = full_path
                        temp_all_steps.add(step)
                    else:
                        st.warning(
                            f"组件 {self.component_instance_id}: HTML文件未找到或不是文件: {full_path} "
                            f"(步骤 {step}, 原始路径: {relative_path_str})"
                        )
                except ValueError:
                    st.warning(
                        f"组件 {self.component_instance_id}: item中的步骤值无效: {item_info.get('related_step')}"
                    )
                except Exception as e:
                    st.error(
                        f"组件 {self.component_instance_id}: 处理item {item_info} 出错: {e}"
                    )
            else:
                st.warning(
                    f"组件 {self.component_instance_id}: 数据源中的item格式无效: {item_info}"
                )

        if not self._all_available_steps and temp_all_steps:
            self._all_available_steps = sorted(list(temp_all_steps))
        elif temp_all_steps and self._all_available_steps is not None:
            if set(self._all_available_steps) != temp_all_steps:
                self._all_available_steps = sorted(list(temp_all_steps))

    def render_settings_ui(self):
        st.markdown("##### 组件设置 (Component Settings)")
        new_html_height = st.number_input(
            "HTML 显示高度 (HTML Display Height)",
            min_value=200,
            max_value=3000,
            step=100,  # 增加最大高度和步长 Increased max height and step
            value=self.html_height,
            key=f"{self.component_instance_id}_html_height",
            help="设置内嵌HTML内容的高度（像素）。Set the height (in pixels) for the embedded HTML content.",
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

        if not self._step_to_html_map:
            if not hasattr(self, "_data_loaded_once"):
                self.load_data()
                self._data_loaded_once = True

            if not self._step_to_html_map:
                st.info(
                    f"组件 {self.component_instance_id}: 没有可显示的Treescope视图数据。"
                )
                st.caption("请确保已生成示例数据或实际数据已正确记录并在清单中声明。")
                return

        if self._all_available_steps is None or not self._all_available_steps:
            if self._step_to_html_map:
                self._all_available_steps = sorted(list(self._step_to_html_map.keys()))

        step_to_display = self._get_closest_available_step(self._current_global_step)

        if step_to_display is None or step_to_display not in self._step_to_html_map:
            st.warning(
                f"组件 {self.component_instance_id}: 在步骤 {self._current_global_step} (或附近) 未找到Treescope视图。"
            )
            available_s = ", ".join(map(str, sorted(self._step_to_html_map.keys())))
            st.caption(
                f"可用步骤 (Available steps): {available_s if available_s else 'None'}"
            )
            return

        html_file_path = self._step_to_html_map[step_to_display]

        if html_file_path.exists():
            try:
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                if (
                    self._current_global_step is not None
                    and step_to_display != self._current_global_step
                ):
                    st.caption(
                        f"显示最接近步骤 {self._current_global_step} 的视图 (步骤 {step_to_display})。"
                        f"Displaying view for step {step_to_display}, closest to target step {self._current_global_step}."
                    )
                else:
                    st.caption(
                        f"显示步骤 {step_to_display} 的视图。Displaying view for step {step_to_display}."
                    )

                st.components.v1.html(
                    html_content, height=self.html_height, scrolling=True
                )
            except Exception as e:
                st.error(
                    f"组件 {self.component_instance_id}: 渲染HTML文件 '{html_file_path}' 失败: {e}"
                )
        else:
            st.error(
                f"组件 {self.component_instance_id}: HTML文件未找到: {html_file_path}"
            )

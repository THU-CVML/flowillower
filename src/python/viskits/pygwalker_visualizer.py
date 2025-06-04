# flowillower/visualizers/pygwalker_visualizer.py
from gradio import render
import streamlit as st
import pandas as pd
import tomli
import tomli_w
from pathlib import Path
from typing import Dict, Any, Optional, List

# 导入Pygwalker Streamlit渲染器
# Import Pygwalker Streamlit renderer
try:
    from pygwalker.api.streamlit import StreamlitRenderer
except ImportError:
    # 在无法导入Pygwalker时提供一个替代方案或明确的错误提示
    # Provide a fallback or clear error if Pygwalker cannot be imported
    st.error("Pygwalker库未安装或无法导入。请运行 'pip install pygwalker'。Pygwalker library not installed or could not be imported. Please run 'pip install pygwalker'.")
    # 定义一个虚拟的StreamlitRenderer以避免后续代码在导入失败时崩溃
    # Define a dummy StreamlitRenderer to prevent subsequent code from crashing if import fails
    class StreamlitRenderer:
        def __init__(self, df, spec=None, spec_io_mode='rw', **kwargs):
            self.df = df
            st.warning("Pygwalker 未加载，将显示原始数据表格。Pygwalker not loaded, displaying raw data table instead.")
        def explorer(self, width: int = None, height: int = None, theme_key: str = None):
            st.dataframe(self.df)


from .base_viskit import VisualizationComponent, register_visualizer

@register_visualizer(name="pygwalker_interactive_dashboard")
class PygwalkerDashboardVisualizer(VisualizationComponent):
    """
    使用Pygwalker为多指标、多track的标量数据提供交互式探索界面的组件。
    A component that uses Pygwalker to provide an interactive exploration interface
    for multi-metric, multi-track scalar data.
    """

    _raw_data_df: Optional[pd.DataFrame] = None # 用于存储加载的原始DataFrame
                                             # To store the loaded raw DataFrame

    def __init__(self,
                 component_instance_id: str,
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]],
                 component_specific_config: Dict[str, Any] = None):
        super().__init__(component_instance_id, trial_root_path, data_sources_map, component_specific_config)
        self._raw_data_df = None
        self.load_component_config() # 加载任何已保存的组件特定配置

        # Pygwalker的配置可以存储在component_specific_config中，例如默认的图表规格
        # Pygwalker's config (e.g., default chart spec) can be stored in component_specific_config
        self.pyg_spec:str = self.config.get("pyg_spec", (trial_root_path/"pyg_spec.json").as_posix()) # Pygwalker的图表规格字符串
        self.pyg_spec_io_mode = self.config.get("pyg_spec_io_mode", "rw") # 'r', 'w', or 'rw'

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        """
        此组件处理与ScalarMetricsDashboardVisualizer相同的数据类型。
        This component handles the same data type as ScalarMetricsDashboardVisualizer.
        """
        return "multi_metric_multi_track_scalars" in data_type_names

    @classmethod
    def get_display_name(cls) -> str:
        return "Pygwalker 交互式仪表盘 (Pygwalker Interactive Dashboard)"

    @classmethod
    def generate_example_data(cls, example_data_path: Path,
                              data_sources_config: Optional[Dict[str, Dict[str, Any]]] = None
                             ) -> Dict[str, Dict[str, Any]]:
        """
        生成与ScalarMetricsDashboardVisualizer类似的示例TOML数据。
        Generates example TOML data similar to ScalarMetricsDashboardVisualizer.
        """
        example_data_path.mkdir(parents=True, exist_ok=True)
        metrics_file_name = "example_pygwalker_scalar_metrics.toml" # 使用不同的文件名以避免冲突 Use a different filename to avoid conflicts
        metrics_file_full_path = example_data_path / metrics_file_name

        data_points = []
        from math import sin  # Add import for sin function
        for step in range(25): # 生成更多数据点以更好地利用Pygwalker Generate more data points for better Pygwalker utilization
            # 指标1: 性能 (train, val, test tracks)
            # Metric 1: Performance (train, val, test tracks)
            data_points.append({"global_step": step, "track": "train", "cpu_usage": 60 + step * 0.5 + sin(step/3)*5, "memory_gb": 4 + (step % 5)*0.2})
            data_points.append({"global_step": step, "track": "validation", "cpu_usage": 55 + step * 0.3 - sin(step/4)*3, "memory_gb": 3.8 + (step % 6)*0.25})
            if step % 2 == 0: # test track记录频率较低 test track logged less frequently
                data_points.append({"global_step": step, "track": "test", "cpu_usage": 50 + step * 0.2, "memory_gb": 3.5 + (step % 4)*0.15})
            
            # 指标2: 错误率 (仅train track)
            # Metric 2: Error Rate (train track only)
            if step % 3 == 0:
                data_points.append({"global_step": step, "track": "validation", "error_rate": 0.4 - step * 0.006})

            data_points.append({"global_step": step, "track": "train", "error_rate": 0.1 - step * 0.003})

            # 模拟多任务学习中的权衡
            data_points.append({
                "global_step": step, "track": "train", "acc_1": 0.5 + step * 0.02 + sin(step/5)*0.1,
            })
            data_points.append({
                "global_step": step, "track": "train", "acc_2": 0.6 + step * 0.015 - sin(step/6)*0.05,
            })


        try:
            with open(metrics_file_full_path, "wb") as f:
                tomli_w.dump({"metrics": data_points}, f)
        except Exception as e:
            st.error(f"为Pygwalker生成示例数据失败: {e}")
            raise

        path_for_manifest = Path(example_data_path.name) / metrics_file_name

        return {
            "main_metrics_source": { # 逻辑数据源名称 Logical data source name
                "asset_id": "example_pygwalker_dashboard_data",
                "data_type": "multi_metric_multi_track_scalars", # 声明的数据类型 Declared data type
                "path": str(path_for_manifest), 
                "display_name": "Pygwalker的示例综合指标 (Example Comprehensive Metrics for Pygwalker)"
            }
        }

    def load_data(self) -> None:
        """
        从TOML文件加载标量数据到Pandas DataFrame。
        Loads scalar data from a TOML file into a Pandas DataFrame.
        """
        data_asset_path = self._get_data_asset_path("main_metrics_source")
        if data_asset_path is None or not data_asset_path.exists():
            st.warning(f"组件 {self.component_instance_id}: 未找到数据源 'main_metrics_source' 或路径无效: {data_asset_path}")
            self._raw_data_df = pd.DataFrame()
            return

        try:
            with open(data_asset_path, "rb") as f:
                data = tomli.load(f)
            
            metrics_list = data.get("metrics", [])
            if not metrics_list:
                self._raw_data_df = pd.DataFrame()
            else:
                self._raw_data_df = pd.DataFrame(metrics_list)
            
            # Pygwalker可以直接处理包含NaN的DataFrame，但确保global_step是数值类型
            # Pygwalker can handle DataFrames with NaNs, but ensure global_step is numeric
            if "global_step" in self._raw_data_df.columns:
                self._raw_data_df["global_step"] = pd.to_numeric(self._raw_data_df["global_step"], errors='coerce')
            
            # 更新所有可用步骤 (如果尚未由外部设置)
            # Update all available steps (if not already set externally)
            if self._all_available_steps is None and not self._raw_data_df.empty and "global_step" in self._raw_data_df.columns:
                valid_steps = pd.to_numeric(self._raw_data_df["global_step"], errors='coerce').dropna().astype(int).unique()
                self._all_available_steps = sorted(list(valid_steps))


        except Exception as e:
            st.error(f"组件 {self.component_instance_id}: 加载Pygwalker数据 '{data_asset_path}' 失败: {e}")
            self._raw_data_df = pd.DataFrame()
            
    def render_settings_ui(self):
        """渲染此组件的设置UI (例如Pygwalker的spec_io_mode)。"""
        st.markdown("##### Pygwalker 设置 (Pygwalker Settings)")
        # spec_io_mode: 'r' (只读), 'w' (只写，每次重置), 'rw' (读写)
        # spec_io_mode: 'r' (read-only), 'w' (write-only, resets each time), 'rw' (read-write)
        new_spec_io_mode = st.selectbox(
            "图表规格交互模式 (Chart Spec I/O Mode)",
            options=['rw', 'r', 'w'],
            index=['rw', 'r', 'w'].index(self.pyg_spec_io_mode),
            key=f"{self.component_instance_id}_pyg_spec_io_mode",
            help="控制Pygwalker图表配置的读写行为。'rw'会保存并加载您的配置。"
                 "Controls read/write behavior of Pygwalker chart configurations. 'rw' saves and loads your config."
        )
        if new_spec_io_mode != self.pyg_spec_io_mode:
            self.pyg_spec_io_mode = new_spec_io_mode
            self.config["pyg_spec_io_mode"] = new_spec_io_mode
            self.save_component_config()
            st.rerun()

        theme_key = st.selectbox(
            "图表主题 Theme type for the GraphicWalker.", 
            options=['vega', 'g2'],
            index=['vega', 'g2'].index(self.config.get("theme_key", "vega")),
            key=f"{self.component_instance_id}_pyg_theme_key",
            help="选择图表的主题类型。Select the theme type for the chart."
        )
        if theme_key != self.config.get("theme_key", "vega"):
            self.config["theme_key"] = theme_key
            self.save_component_config()
            st.rerun()

    def render(self) -> None:
        """
        使用Pygwalker渲染数据探索界面。
        Renders the data exploration interface using Pygwalker.
        """
        st.subheader(self.get_display_name()) # 使用类方法获取显示名称 Use class method for display name

        with st.expander("Pygwalker 显示设置 (Pygwalker Display Settings)", expanded=False):
            self.render_settings_ui()

        if self._raw_data_df is None:
            self.load_data() # 尝试加载数据 Try to load data

        if self._raw_data_df is None or self._raw_data_df.empty:
            st.info(f"组件 {self.component_instance_id}: 没有数据可供Pygwalker显示。")
            st.caption("请确保已生成示例数据或实际数据已正确记录。")
            return
        
        # Pygwalker 通常不需要直接与外部的 global_step 滑块交互来进行其核心渲染，
        # 因为它提供了自己的过滤和探索UI。
        # 但是，如果需要，我们可以基于 self._current_global_step 预先筛选DataFrame。
        # Pygwalker usually doesn't need to interact directly with an external global_step slider
        # for its core rendering, as it provides its own filtering and exploration UI.
        # However, we could pre-filter the DataFrame based on self._current_global_step if desired.

        # st.caption(f"当前模拟全局步骤 (参考): {self._current_global_step}")
        # st.caption(f"Pygwalker将使用完整数据集进行探索。Pygwalker will use the full dataset for exploration.")

        try:
            # 为Pygwalker创建一个唯一的key，以便它能正确管理自己的状态/图表规格
            # Create a unique key for Pygwalker so it can manage its state/chart spec correctly
            pyg_key = f"pygwalker_{self.component_instance_id}"

            # StreamlitRenderer会使用这个key来持久化图表规格（如果spec_io_mode允许）
            # StreamlitRenderer will use this key to persist chart spec (if spec_io_mode allows)
            # 我们不需要显式地从self.config加载或保存pyg_spec，Pygwalker会处理
            # We don't need to explicitly load/save pyg_spec from self.config, Pygwalker handles it

            # https://github.com/Kanaries/pygwalker
            @st.cache_resource
            def get_pyg_renderer() -> "StreamlitRenderer":
                renderer = StreamlitRenderer(
                    self._raw_data_df,
                    spec=self.pyg_spec, # 图表保存的位置
                    spec_io_mode=self.pyg_spec_io_mode,
                    theme_key=self.config.get("theme_key", "vega"), # 使用Streamlit主题 Use Streamlit theme
                    key=pyg_key # 非常重要，用于状态持久化 Very important for state persistence
                    ,kernel_computation=True # 通过设置 kernel_computation=True，将启用由 DuckDB 提供动力的 pygwalker 的新计算引擎。
                )
                return renderer
            # 获取Pygwalker渲染器实例
            # Get the Pygwalker renderer instance
            renderer = get_pyg_renderer()
            # https://docs.kanaries.net/zh/pygwalker/api-reference/streamlit
            # explorer() 方法会渲染Pygwalker的UI
            # The explorer() method renders Pygwalker's UI
            # renderer.explorer(width=None, height=1000, scrolling=False, default_tab="vis")
            renderer.explorer()
            
            # 如果spec_io_mode允许写入，Pygwalker会自动将会话中的spec保存到Streamlit的会话状态或类似机制中
            # If spec_io_mode allows writing, Pygwalker automatically saves the spec from the session
            # into Streamlit's session state or similar mechanism, associated with the key.
            # 如果我们想手动获取并保存到我们自己的组件配置中，会更复杂一些。
            # It's more complex if we want to manually fetch and save it to our own component config.
            # 目前，我们依赖Pygwalker自身的持久化（基于key）。
            # For now, we rely on Pygwalker's own persistence (based on the key).

        except NameError as ne: # 处理StreamlitRenderer未定义的情况 (如果Pygwalker导入失败)
            if "StreamlitRenderer" in str(ne):
                 st.error("Pygwalker 渲染器不可用。请检查安装。Pygwalker renderer is unavailable. Please check installation.")
                 st.dataframe(self._raw_data_df) # 回退到显示DataFrame Fallback to displaying DataFrame
            else:
                raise ne # 重新抛出其他NameError Re-raise other NameErrors
        except Exception as e:
            st.error(f"渲染Pygwalker组件时出错: {e}")
            st.dataframe(self._raw_data_df) # 作为回退，显示原始数据 As a fallback, display raw data
